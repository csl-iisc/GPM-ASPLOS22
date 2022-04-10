#define EMULATE_NVM_BW
//#define CONV_LOG
extern "C" 
{
#include "change-ddio.h"
}
#include "virginian.h"
#include "libgpmlog.cuh"
#include "bandwidth_analysis.cuh"
#include <chrono>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define timediff(a, b) std::chrono::duration_cast<std::chrono::microseconds>(a - b).count()

// UPDATE THIS DESCRIPTION LATER
/**
 * @ingroup table
 * @brief Update rows in a table
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

static __global__ void logMetadataKernel(gpmlog *log, int *log_data, size_t size)
{
	gpmlog_insert(log, log_data, size, 0);
}

static __global__ void clearLogKernel(gpmlog *log, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < size)
	    gpmlog_clear(log, tid);
}

__global__ void updateRangeColKernel(virg_tablet_meta *tab, char *data, int upd_row_start, int upd_row_end, int tab_row_start, int upd_col, gpmlog *dlog)
{
    int row_number = threadIdx.x + blockDim.x * blockIdx.x + upd_row_start;
    if(row_number > upd_row_end || row_number - tab_row_start >= tab->rows)
        return;
    
    char *fixed_ptr = (char*)tab->base_ptr + tab->fixed_block;
    // copy over column from buffer
    char *dest = fixed_ptr + tab->fixed_offset[upd_col] + (row_number - tab_row_start) * tab->fixed_stride[upd_col];
    char *src = data + (row_number - upd_row_start) * tab->fixed_stride[upd_col];
    gpmlog_insert(dlog, dest, tab->fixed_stride[upd_col], (row_number - upd_row_start));
	vol_memcpy(dest, src, tab->fixed_stride[upd_col]);
}

int virg_table_update_range(virginian *v, unsigned table_id, int row_start, int row_end, char *data, int data_size, int column, double *timer)
{
    int total_rows = row_end - row_start + 1;	
	// Create logs for current operation
	char *log_file = (char *)malloc(sizeof(char) * (4 + strlen(v->file_name) + 1));
	strcpy(log_file, "urlog_");
	strcat(log_file, v->file_name);
	gpmlog *log = gpmlog_create(log_file, sizeof(int) * 3, 1);
	
	// Create log for data being updated
	char *data_log_file = (char *)malloc(sizeof(char) * (5 + strlen(v->file_name) + 1));
	strcpy(data_log_file, "urdlog_");
	strcat(data_log_file, v->file_name);
    size_t log_size = data_size;
#if defined(CONV_LOG)
	gpmlog *dlog = gpmlog_create(data_log_file, log_size, total_rows);
#else
	gpmlog *dlog = gpmlog_create_managed(data_log_file, log_size, (total_rows + 511) / 512, 512);
#endif
    if(total_rows <= 0)
        return VIRG_FAIL;

    auto start_timer = std::chrono::steady_clock::now();
	virg_tablet_meta *tab;
	// load the table's first tablet
	virg_db_load(v, v->db.first_tablet[table_id], &tab);
	int start = 0;
	// move tablet until we reach appropriate location
    if(start + tab->rows < row_start) {
        start += tab->rows;
	    virg_db_loadnext(v, &tab);
    }

	// check for corruption
	assert(tab->rows <= tab->possible_rows);

	int *log_data;
	cudaMalloc((void **)&log_data, sizeof(int) * 3);
	cudaMemcpy(log_data, &row_start, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 1, &row_end, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 2, &column, sizeof(int), cudaMemcpyHostToDevice);

	logMetadataKernel<<<1, 1>>> (log, log_data, sizeof(int) * 3);

	cudaFree(log_data);

	char *d_data;
    cudaMalloc((void **)&d_data, data_size);
	cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
	
    while(start < row_end) {
	    virg_tablet_meta *d_tab;
	    cudaMalloc((void **)&d_tab, sizeof(virg_tablet_meta));
	    cudaMemcpy(d_tab, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);
	    
	    int blocks = (total_rows + 511) / 512;
	    updateRangeColKernel<<<blocks, 512>>> (d_tab, d_data, row_start, row_end, start, column, dlog);
		cudaFree(d_tab);
		
	    virg_db_write_tab(v, tab);
	    start += tab->rows;
	    virg_db_loadnext(v, &tab);
	}

	cudaFree(d_data);
	virg_tablet_unlock(v, tab->id);
#ifndef RESTORE_FLAG
	clearLogKernel<<<(total_rows + 511) / 512, 512>>>(dlog, total_rows);
	clearLogKernel<<<1, 1>>>(log, 1);
#endif
	cudaDeviceSynchronize();
	
    auto end_timer = std::chrono::steady_clock::now();
	
	gpmlog_close(dlog);
	gpmlog_close(log);
    
    if(timer != NULL)
        *timer = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count() / 1000000.0f;
    
	return VIRG_SUCCESS;
}

__global__ void updateColKernel(virg_tablet_meta *tab, char *data, int *selected_rows, int num_rows, int tab_row_start, int upd_col, gpmlog *dlog)
{
    int row_index = threadIdx.x + blockDim.x * blockIdx.x;
    while(row_index < num_rows && selected_rows[row_index] < tab_row_start)
        row_index += blockDim.x * gridDim.x;
    if(row_index >= num_rows || selected_rows[row_index] - tab_row_start >= tab->rows)
        return;
    
    char *fixed_ptr = (char*)tab->base_ptr + tab->fixed_block;
    // copy over column from buffer
    char *dest = fixed_ptr + tab->fixed_offset[upd_col] + (selected_rows[row_index] - tab_row_start) * tab->fixed_stride[upd_col];
    char *src = data + row_index * tab->fixed_stride[upd_col];
    gpmlog_insert(dlog, dest, tab->fixed_stride[upd_col], row_index);       // Insert old row value
    BW_DELAY(CALC(73, 24, (tab->fixed_stride[upd_col] + sizeof(size_t))));
    gpmlog_insert(dlog, &selected_rows[row_index], sizeof(int), row_index); // Inform which row is being updated
    BW_DELAY(CALC(73, 24, (sizeof(int) + sizeof(size_t))));
	gpm_memcpy(dest, src, tab->fixed_stride[upd_col], cudaMemcpyDeviceToDevice);
    BW_DELAY(CALC(73, 24, tab->fixed_stride[upd_col]));
}

#define TPB 512
#define TPBM (TPB - 1)

int virg_table_update(virginian *v, unsigned table_id, int *selected_rows, int num_rows, char *data, int data_size, 
	int column, double &operation_time, double &ddio_time, double &persist_time, bool device_mem)
{
	// Create log for metadata
	char *log_file = (char *)malloc(sizeof(char) * (4 + strlen(v->file_name) + 1));
	strcpy(log_file, "ulog_");
	strcat(log_file, v->file_name);
	gpmlog *log = gpmlog_create(log_file, sizeof(int) * 2, 1);
	
	// Create log for data being updated
	char *data_log_file = (char *)malloc(sizeof(char) * (5 + strlen(v->file_name) + 1));
	strcpy(data_log_file, "udlog_");
	strcat(data_log_file, v->file_name);
    size_t log_size = (data_size / num_rows + sizeof(int)) * ((num_rows + TPBM) / TPB) * TPB;
#if defined(CONV_LOG)
    gpmlog *dlog = gpmlog_create(data_log_file, log_size, num_rows);
#else
	gpmlog *dlog = gpmlog_create_managed(data_log_file, log_size, (num_rows + TPBM) / TPB, TPB);
#endif
    cudaDeviceSynchronize();
    //START_BW_MONITOR2("bw_gpm_dbupdate.csv")
    
    auto start_timer = TIME_NOW;
    
    if(num_rows <= 0)
        return VIRG_FAIL;

	virg_tablet_meta *tab;
	// load the table's first tablet
	virg_db_load(v, v->db.first_tablet[table_id], &tab);
	int start = 0;
	int starting;
	if(device_mem)
		cudaMemcpy(&starting, selected_rows, sizeof(int), cudaMemcpyDeviceToHost);
	else
	 	starting = selected_rows[0];
	// move tablet until we reach appropriate location
    if(start + tab->rows < starting) {
        start += tab->rows;
	    virg_db_loadnext(v, &tab);
    }

	// check for corruption
	assert(tab->rows <= tab->possible_rows);
	
	int *log_data;
	cudaMalloc((void **)&log_data, sizeof(int) * 2);
	cudaMemcpy(log_data, &column, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(log_data + 1, &table_id, sizeof(unsigned), cudaMemcpyHostToDevice);
	operation_time += time_val(TIME_NOW - start_timer).count();
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start_timer = TIME_NOW;
    ddio_off(); 
	ddio_time += time_val(TIME_NOW - start_timer).count();
#endif
	start_timer = TIME_NOW;
	logMetadataKernel<<<1, 1>>> (log, log_data, sizeof(int) * 2);
	cudaFree(log_data);
	operation_time += time_val(TIME_NOW - start_timer).count();
	start_timer = TIME_NOW;
	char *d_data;
	int *d_sel_rows;
	cudaMalloc((void **)&d_data, data_size);
	cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
	if(!device_mem) {
		cudaMalloc((void **)&d_sel_rows, num_rows * sizeof(int));
		cudaMemcpy(d_sel_rows, selected_rows, num_rows * sizeof(int), cudaMemcpyHostToDevice);
	}
	else {
		d_sel_rows = selected_rows;
	}
	operation_time += time_val(TIME_NOW - start_timer).count();	
	start_timer = TIME_NOW;
    virg_tablet_meta *d_tab;
    cudaMalloc((void **)&d_tab, sizeof(virg_tablet_meta));
	operation_time += time_val(TIME_NOW - start_timer).count();	
	int ending;
	if(device_mem)
		cudaMemcpy(&ending, &selected_rows[num_rows - 1], sizeof(int), cudaMemcpyDeviceToHost);
	else
	 	ending = selected_rows[num_rows - 1];
    while(start < ending) {
		start_timer = TIME_NOW;
	    cudaMemcpy(d_tab, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);

	    int blocks = (num_rows + TPBM) / TPB;
	    updateColKernel<<<blocks, 512>>> (d_tab, d_data, d_sel_rows, num_rows, start, column, dlog);
		cudaDeviceSynchronize();
	    virg_db_write_tab(v, tab);
	    start += tab->rows;
		operation_time += time_val(TIME_NOW - start_timer).count();	
#ifdef GPM_WDP
		start_timer = TIME_NOW;
		pmem_mt_persist(tab->base_ptr, tab->size);
		persist_time += time_val(TIME_NOW - start_timer).count();
#endif
	    virg_db_loadnext(v, &tab);	    
	}
	start_timer = TIME_NOW;
	cudaFree(d_tab);

	cudaFree(d_data);
	virg_tablet_unlock(v, tab->id);
#ifndef RESTORE_FLAG
	clearLogKernel<<<(num_rows + 511) / 512, 512>>>(dlog, num_rows);
	clearLogKernel<<<1, 1>>>(log, 1);
#endif
	cudaDeviceSynchronize();
	operation_time += time_val(TIME_NOW - start_timer).count();	
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
	start_timer = TIME_NOW;
    ddio_on(); 
	ddio_time += time_val(TIME_NOW - start_timer).count();
#endif
	//STOP_BW_MONITOR
	gpmlog_close(dlog);
	gpmlog_close(log);
	//OUTPUT_STATS

	return VIRG_SUCCESS;
}

__global__ void getLogData(gpmlog *log, int *log_data)
{
    size_t size = gpmlog_get_size(log, 0);
    if(size < sizeof(int) * 2)
        log_data[1] = -1;
    else
        gpmlog_read(log, log_data, sizeof(int) * 2, 0);
}

__global__ void recoverTable(virg_tablet_meta *tab, gpmlog *dlog, int column, int start, bool *cont, int *row_done, int partitions)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= partitions) {
        return;
    }
    if(row_done[id] == -1)
        return;
    // Incomplete log entry, so update did not complete
    if(row_done[id] == 0 && gpmlog_get_size(dlog, id) != tab->fixed_stride[column] + sizeof(int)) {
        return;
    }
    
    int row = 0;
    if(row_done[id] == 0)
        gpmlog_read(dlog, &row_done[id], sizeof(int));
    row = row_done[id];
    // Already done the recovery
    if(row < start)
        return;
    // Row not in current tablet, wait for proper one to come
    if(row >= start + tab->rows) {
        *cont = true;
        return;
    }
    gpmlog_remove(dlog, sizeof(int));
    char *fixed_ptr = (char*)tab->base_ptr + tab->fixed_block;
    char *dest = fixed_ptr + tab->fixed_offset[column] + (row - start) * tab->fixed_stride[column];
    gpmlog_read(dlog, dest, tab->fixed_stride[column]);
    row_done[id] = -1;
}

void virg_recover_update(virginian *v, double *timer)
{
    // Create log for metadata
	char *log_file = (char *)malloc(sizeof(char) * (4 + strlen(v->file_name) + 1));
	strcpy(log_file, "ulog_");
	strcat(log_file, v->file_name);
	struct stat sb;
	std::string st = "/pmem/";
	if (stat((st + log_file).c_str(), &sb) != -1) {
	    printf("Found %s log\n", log_file);
	    gpmlog *log = gpmlog_open(log_file);
auto start_timer = std::chrono::steady_clock::now();
        int *log_data;
        cudaMalloc((void **)&log_data, sizeof(int) * 2);
        int column;
        unsigned table_id;
        getLogData<<<1, 1>>>(log, log_data);
        cudaMemcpy(&column, log_data, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&table_id, log_data + 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
auto end_timer = std::chrono::steady_clock::now();
	    if(timer != NULL)
            *timer += time_val(end_timer - start_timer).count();
                
	    char *data_log_file = (char *)malloc(sizeof(char) * (6 + strlen(v->file_name) + 1));
        strcpy(data_log_file, "udlog_");
        strcat(data_log_file, v->file_name);
        printf("Searching for %s\n", (st + data_log_file).c_str());
	    if(table_id != (unsigned)-1 && stat((st + data_log_file).c_str(), &sb) != -1) {
	        printf("Found %s log\n", data_log_file);
	        gpmlog *dlog = gpmlog_open(data_log_file);
	        int partitions = gpmlog_get_partitions(dlog);
	        int blk = (partitions + TPBM) / TPB;
	        int thd = TPB;
	        printf("Table %u, column %d, with %d partitions\n", table_id, column, partitions);
            virg_tablet_meta *d_tab;
            cudaMalloc((void **)&d_tab, sizeof(virg_tablet_meta));
            int *row_done;
            cudaMalloc((void **)&row_done, sizeof(int) * partitions);
            cudaMemset(row_done, 0, sizeof(int) * partitions);
start_timer = std::chrono::steady_clock::now();
	        virg_tablet_meta *tab;
	        // load the table's first tablet
	        virg_db_load(v, v->db.first_tablet[table_id], &tab);
	        int start = 0;
	        bool cont = false;
            bool *d_continue;
            cudaMalloc((void **)&d_continue, sizeof(bool));
	        do{
	            cont = false;
	            cudaMemcpy(d_continue, &cont, sizeof(bool), cudaMemcpyHostToDevice);
	            cudaMemcpy(d_tab, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);
	            recoverTable<<<blk, thd>>>(d_tab, dlog, column, start, d_continue, row_done, partitions);
	            virg_db_write_tab(v, tab);
	            start += tab->rows;
	            virg_db_loadnext(v, &tab);
	            cudaMemcpy(&cont, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);
	        }while(cont);
	        clearLogKernel<<<blk, thd>>>(dlog, partitions);
	        cudaDeviceSynchronize();
end_timer = std::chrono::steady_clock::now();    
            cudaFree(d_tab);
            cudaFree(row_done);
            if(timer != NULL)
                *timer += time_val(end_timer - start_timer).count();
	        gpmlog_close(dlog);
	        virg_tablet_unlock(v, tab->id);
	    }
	    clearLogKernel<<<1, 1>>>(log, 1);
	    gpmlog_close(log);
	}
}
