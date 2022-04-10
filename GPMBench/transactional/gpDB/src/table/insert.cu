//#define EMULATE_NVM_BW
extern "C" 
{
#include "change-ddio.h"
}
#include "virginian.h"
#include "libgpmlog.cuh"
#include "bandwidth_analysis.cuh"
#include <chrono>

void *PMEM_START_HOST; 
__constant__ char *PMEM_START_DEV;
#ifdef OUTPUT_NVM_DETAILS
__device__ uint64_t nvm_write;
__device__ uint64_t nvm_read;
#endif
char *gpm_start_alloc; // Point from which memory allocation can begin

bool gpm_init_complete = false;

/**
 * @ingroup table
 * @brief Insert a row into a table
 *
 * Insert a new row by adding it to the end of a table. This function locates
 * the tablet where we have set the write_cursor. If the tablet is full, we
 * attempt to add more row space with virg_tablet_addrows(), and if we can't,
 * then we move onto the next tablet in the tablet string. The key and data
 * arguments are passed as pointers to their buffer because the size of their
 * variable types is unknown. The data buffer should contain all the columns in
 * order immediately adjacent to each other. For example, the following code
 * adds a row to a table with an integer key, and an integer, double, and float
 * column:
 *
 * @code
 * int i = 150;
 * char buff[sizeof(int) + sizeof(double) + sizeof(float)];
 * char *x = &buff[0];
 * ((int*)x)[0] = 100;
 * x += sizeof(int);
 * ((double*)x)[0] = 100.0;
 * x += sizeof(double);
 * ((float*)x)[0] = 100.0;
 * virg_table_insert(v, table_id, &i, &buff[0], NULL);
 * @endcode
 *
 * @param v 		Pointer to the state struct of the database system
 * @param table_id 	Table in which to insert the row
 * @param key 		Buffer containing the key value for this row
 * @param data		Buffer containing the row data
 * @param blob		Buffer containing variable size data associated with the
 * key (unimplemented)
 * @return VIRG_SUCCESS or VIRG_FAIL depending on errors during the function
 * call
 */
#define NUM_VARS 6
static __global__ void logMetadataKernel(gpmlog *log, unsigned *log_data)
{
	gpmlog_insert(log, log_data, sizeof(unsigned) * NUM_VARS, 0);
}

static __global__ void clearLogKernel(gpmlog *log)
{
	gpmlog_clear(log, 0);
}

__global__ void addRowsKernel(virg_tablet_meta *tab, char *key, char *data, int old_row_size, int new_rows, int total_rows)
{
    int row_number = threadIdx.x + blockDim.x * blockIdx.x;
    for(; row_number < new_rows; row_number += blockDim.x * gridDim.x) {
		//if(row_number >= new_rows)
		//    return;
		    
		int data_row = old_row_size + row_number;
		int tab_row = tab->rows + row_number;
		
		// copy key from buffer
		char *fixed_ptr = (char*)tab->base_ptr + tab->fixed_block;

		//assert(tab_row < tab->possible_rows);
		
		int src_offset = 0;

		char *key_dest = (char*)tab->base_ptr + tab->key_block + tab_row * tab->key_stride;
		char *key_src  = key + data_row * tab->key_stride;
		gpm_memcpy_nodrain(key_dest, key_src, tab->key_stride, cudaMemcpyDeviceToDevice);
		BW_DELAY(CALC(61, 22, tab->key_stride));
		
		// copy over all columns from buffer
		for(int i = 0; i < tab->fixed_columns; i++) {
			int stride = tab->fixed_stride[i];
			char *dest = fixed_ptr + tab->fixed_offset[i] + tab_row * stride;
			char *src = data + src_offset + data_row * stride;
			gpm_memcpy_nodrain(dest, src, stride, cudaMemcpyDeviceToDevice);
			BW_DELAY(CALC(61, 22, stride));
			src_offset += stride * total_rows;
		}
	}
	gpm_drain();
}

int virg_table_insert(virginian *v, unsigned table_id, char *key, char *data, int data_size, char *blob)
{
	virg_tablet_meta *tab;

	// load the table's table on which we have a write cursor
	virg_db_load(v, v->db.write_cursor[table_id], &tab);

	// check for corruption
	assert(tab->rows <= tab->possible_rows);
	
	char *d_key, *d_data;
    cudaMalloc((void **)&d_key, tab->key_stride);
    cudaMalloc((void **)&d_data, data_size);
	cudaMemcpy(d_key, key, tab->key_stride, cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
	
    // if the current tablet is full
    if(tab->rows == tab->possible_rows) {
	    // if there is room to add more fixed-size rows
	    if(tab->size < VIRG_TABLET_SIZE - tab->row_stride)
		    virg_tablet_addrows(v, tab, VIRG_TABLET_KEY_INCREMENT);
	    // otherwise move on to the next tablet
	    else {
		    virg_db_loadnext(v, &tab);
		    v->db.write_cursor[tab->table_id] = tab->id;
	    }
    }
    
    virg_tablet_meta *d_tab;
    cudaMalloc((void **)&d_tab, sizeof(virg_tablet_meta));
    cudaMemcpy(d_tab, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);
    
    addRowsKernel<<<1, 1>>> (d_tab, d_key, d_data, 0, 0, 1);
    
    cudaMemcpy(tab, d_tab, sizeof(virg_tablet_meta), cudaMemcpyDeviceToHost);
	cudaFree(d_tab);
	
    virg_db_write_tab(v, tab);

	cudaFree(d_data);
	virg_tablet_unlock(v, tab->id);
	
	char *cursor = (char*)&v->db.write_cursor[tab->table_id];
	char *loc = v->dbfd + (cursor - (char *)&v->db);
	// Persist new write_cursor
    gpm_memcpy(loc, cursor, sizeof(v->db.write_cursor[tab->table_id]), cudaMemcpyHostToDevice); 

	return VIRG_SUCCESS;
}

int virg_table_insert_multiple(virginian *v, unsigned table_id, char *key, char *data, int data_size, int rows, 
	char *blob, double &operation_time, double &ddio_time, double &persist_time)
{
    char *log_file = (char *)malloc(sizeof(char) * (4 + strlen(v->file_name) + 1));
	strcpy(log_file, "ilog_");
	strcat(log_file, v->file_name);
	gpmlog *log = gpmlog_create(log_file, sizeof(int) * NUM_VARS, 1);

	//START_BW_MONITOR2("bw_gpm_dbinsert.csv")
    auto start_timer = TIME_NOW;
    
	virg_tablet_meta *tab;
	// load the table's table on which we have a write cursor
	virg_db_load(v, v->db.write_cursor[table_id], &tab);
	// check for corruption
	assert(tab->rows <= tab->possible_rows);
	
	// Log metadata of current tablet
	unsigned *log_data;
	cudaMalloc((void **)&log_data, sizeof(unsigned) * NUM_VARS);
	cudaMemcpy(log_data, &tab->rows, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 1, &v->db.write_cursor[table_id], sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 2, &table_id, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 3, &v->db.table_tablets[tab->table_id], sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 4, &v->db.last_tablet[tab->table_id], sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 5, &tab->size, sizeof(unsigned), cudaMemcpyHostToDevice);
	operation_time += time_val(TIME_NOW - start_timer).count();
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start_timer = TIME_NOW;
    ddio_off(); 
    ddio_time += time_val(TIME_NOW - start_timer).count();
#endif

	start_timer = TIME_NOW;
	logMetadataKernel<<<1, 1>>> (log, log_data);
	cudaFree(log_data);	
	operation_time += time_val(TIME_NOW - start_timer).count();
	
	start_timer = TIME_NOW;
    int inserted = 0;
	char *d_key, *d_data;
    cudaMalloc((void **)&d_key, rows * tab->key_stride);
    cudaMalloc((void **)&d_data, data_size);
	cudaMemcpy(d_key, key, rows * tab->key_stride, cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);

	virg_tablet_meta *d_tab;
	cudaMalloc((void **)&d_tab, sizeof(virg_tablet_meta));
#if defined(NVM_ALLOC_GPU) && !defined(EMULATE_NVM)
    int *tablet;
    cudaMalloc((void **)&tablet, VIRG_TABLET_SIZE);
#endif
	operation_time += time_val(TIME_NOW - start_timer).count();
    while(inserted < rows) {
		start_timer = TIME_NOW;
	    // if the current tablet is full
	    if(tab->rows == tab->possible_rows) {
		    // if there is room to add more fixed-size rows
		    if(tab->size < VIRG_TABLET_SIZE - tab->row_stride)
			    virg_tablet_addrows(v, tab, VIRG_TABLET_KEY_INCREMENT);
		    // otherwise move on to the next tablet
		    else {
			    virg_db_loadnext(v, &tab);
			    v->db.write_cursor[tab->table_id] = tab->id;
		    }
	    }
	    int added_rows = min(rows - inserted, tab->possible_rows - tab->rows);
	    cudaMemcpy(d_tab, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);

	    int blocks = added_rows / 1024 + 1;
	    addRowsKernel<<<blocks, 1024>>> (d_tab, d_key, d_data, inserted, added_rows, rows);
	    tab->rows += added_rows;
	    virg_db_write_tab(v, tab);
	    inserted += added_rows;
		operation_time += time_val(TIME_NOW - start_timer).count();
#ifdef GPM_WDP
		start_timer = TIME_NOW;
		pmem_mt_persist(tab->base_ptr, tab->size);
		persist_time += time_val(TIME_NOW - start_timer).count();
#endif
	}
	start_timer = TIME_NOW;
	char *cursor = (char*)&v->db.write_cursor[tab->table_id];
	char *loc = v->dbfd + (cursor - (char *)&v->db);
	// Persist new write_cursor
    gpm_memcpy(loc, cursor, sizeof(v->db.write_cursor[tab->table_id]), cudaMemcpyHostToDevice);
	cudaFree(d_tab);
	cudaFree(d_data);
	virg_tablet_unlock(v, tab->id);
#ifndef RESTORE_FLAG
	clearLogKernel<<<1, 1>>>(log);
#endif
	cudaDeviceSynchronize();
	operation_time += time_val(TIME_NOW - start_timer).count();
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start_timer = TIME_NOW;
    ddio_on(); 
    ddio_time += time_val(TIME_NOW - start_timer).count();
#endif
	//STOP_BW_MONITOR
	//OUTPUT_STATS
	gpmlog_close(log);

	return VIRG_SUCCESS;
}

__global__ void getLogInsertData(gpmlog *log, int *log_data)
{
    size_t size = gpmlog_get_size(log, 0);
    if(size < sizeof(int) * NUM_VARS)
        log_data[2] = -1;
    else
        gpmlog_read(log, log_data, sizeof(int) * 5, 0);
}

void virg_recover_insert(virginian *v, double *timer)
{
    // Create log for metadata
	char *log_file = (char *)malloc(sizeof(char) * (4 + strlen(v->file_name) + 1));
	strcpy(log_file, "ilog_");
	strcat(log_file, v->file_name);
	struct stat sb;
	std::string st = "/pmem/";
	if (stat((st + log_file).c_str(), &sb) != -1) {
	    printf("Found %s log\n", log_file);
	    gpmlog *log = gpmlog_open(log_file);
auto start_timer = std::chrono::steady_clock::now();

        int *log_data;
        cudaMalloc((void **)&log_data, sizeof(int) * NUM_VARS);
        unsigned rows;
        unsigned write_cursor;
        unsigned table_id;
        unsigned tablets;
        unsigned last_tablet;
        unsigned tab_size;
        getLogInsertData<<<1, 1>>>(log, log_data);
        cudaMemcpy(&rows, log_data, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&write_cursor, log_data + 1, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&table_id, log_data + 2, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tablets, log_data + 3, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_tablet, log_data + 4, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tab_size, log_data + 5, sizeof(unsigned), cudaMemcpyDeviceToHost);
        if(table_id == -1) {
        	printf("Empty log, quitting recovery\n");
        	return;
        }
        printf("%u rows, %u write_cursor, %u table_id, tablets: %u, last tablet: %d, tab_size: %u\n", rows, write_cursor, table_id, tablets, last_tablet, tab_size);
        v->db.write_cursor[table_id] = write_cursor;
        v->db.last_tablet[table_id] = last_tablet;
        v->db.table_tablets[table_id] = tablets;
        
        virg_tablet_meta *tab;
	    // load the table's table on which we have a write cursor
	    virg_db_load(v, v->db.write_cursor[table_id], &tab);
        tab->rows = rows;
        tab->size = tab_size;
        tab->last_tablet = 1;
	    virg_db_write_tab(v, tab);
	    
        char *loc = v->dbfd + ((char *)&v->db.write_cursor[table_id] - (char *)&v->db);
        // Persist new write_cursor
        gpm_memcpy(loc, &write_cursor, sizeof(v->db.write_cursor[tab->table_id]), cudaMemcpyHostToDevice);
        
        loc = v->dbfd + ((char *)&v->db.last_tablet[table_id] - (char *)&v->db);
        // Persist new write_cursor
        gpm_memcpy(loc, &last_tablet, sizeof(v->db.last_tablet[tab->table_id]), cudaMemcpyHostToDevice);
        
        loc = v->dbfd + ((char *)&v->db.table_tablets[table_id] - (char *)&v->db);
        // Persist new write_cursor
        gpm_memcpy(loc, &tablets, sizeof(v->db.table_tablets[tab->table_id]), cudaMemcpyHostToDevice);
        
        cudaDeviceSynchronize();
auto end_timer = std::chrono::steady_clock::now();
        //printf("Log time: %f ms, setup time: %f ms, fetch time = %f ms, kernel time = %f ms\n");
        if(timer != NULL)
            *timer = time_val(end_timer - start_timer).count();
        virg_tablet_unlock(v, tab->id);
	    clearLogKernel<<<1, 1>>>(log);
	    gpmlog_close(log);
	}
}

