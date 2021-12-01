#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <virginian.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <set>
#include <cub/cub.cuh>
double insert_persist_time = 0, insert_operation_time = 0, insert_ddio_time = 0; 
double update_persist_time = 0, update_operation_time = 0, update_ddio_time = 0; 

int main(int argc, char **argv)
{
	if(argc != 3) {
		fprintf(stderr, "%s <database name> <rows>\n", argv[0]);
		exit(1);
	}

	unlink(argv[1]);
	virginian virg;
	virginian *v = &virg;

	virg_init(v);
	virg_db_create(v, argv[1]);
	
	virg_table_create(v, "test", VIRG_INT);
    virg_table_addcolumn(v, 0, "uniformi", VIRG_INT);
    virg_table_addcolumn(v, 0, "normali5", VIRG_INT);
    virg_table_addcolumn(v, 0, "normali20", VIRG_INT);
    virg_table_addcolumn(v, 0, "uniformf", VIRG_FLOAT);
    virg_table_addcolumn(v, 0, "normalf5", VIRG_FLOAT);
    virg_table_addcolumn(v, 0, "normalf20", VIRG_INT);

    long rows = 50000000;//atoi(argv[2]);

    const long ROW_SIZE = 6 * 4;

	int i;
	void *buff = malloc(ROW_SIZE * rows);
	int *buff_i = (int*)buff;
	float *buff_f = (float*)buff;

    int *keys = (int *)malloc(sizeof(int) * rows);

    // insert multiple uses column-major format
	for(i = 0; i < rows; i++) {
		buff_i[i + rows * 0] = i;//(int)gsl_ran_flat(ran, -100, 100);
		buff_i[i + rows * 1] = 2 * i;//(int)gsl_ran_gaussian(ran, 5);
		buff_i[i + rows * 2] = i / 10;//(int)gsl_ran_gaussian(ran, 20);
		buff_f[i + rows * 3] = 0.1f * (float)i;//(float)gsl_ran_flat(ran, -100, 100);
		buff_f[i + rows * 4] = (float)i / 2.0f;//(float)gsl_ran_gaussian(ran, 5);
		buff_i[i + rows * 5] = 0;//(float)gsl_ran_gaussian(ran, 20);
		keys[i] = i;
	}

    const long update_rows = rows / 20;
    int *buff2 = (int *)malloc(sizeof(int) * update_rows);
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,rows);
    std::set<int> row_sel;
    int *selected2 = (int *)malloc(sizeof(int) * update_rows);
    
    // insert multiple uses column-major format
	for(i = 0; i < update_rows; i++) {
	    int val;
	    do{
	        val = distribution(generator);
	    }while(row_sel.find(val) != row_sel.end());
		buff2[i] = 100;
		buff_i[val + rows * 5] = 1;
		row_sel.insert(val);
		selected2[i] = val;
	}
	
	virg_table_insert_multiple(v, 0, (char*)keys, (char*)buff, ROW_SIZE * rows, rows, NULL, 
		insert_operation_time, insert_ddio_time, insert_persist_time);
    
	free(buff);

	// declare reader pointer
	virg_reader *r;
	
	// execute query
	virg_query(v, &r, "select id, uniformi, normali5, normali20, uniformf, normalf5, normalf20 from test where normali20 < 700002 AND normali20 >= 699999");

	// output result column names
	unsigned j;
	for(j = 0; j < r->res->fixed_columns; j++)
		printf("%s\t", r->res->fixed_name[j]);
	printf("\n");

	// output result data
	int a;
	do {
	    a = virg_reader_row(v, r);
	    if(a != VIRG_FAIL) {
		int *results = (int*)r->buffer;
		float *res = (float*)r->buffer;

		printf("%i\t%i\t%i\t%i\t%f\t%f\t%d\n", results[0], results[1], results[2], results[3], res[4], res[5], results[6]);
		}
	}while(a != VIRG_FAIL);
	virg_release(v, r);

	double update_time = 0;
	auto start_time = TIME_NOW;
    int *selected;
    //cudaMalloc((void**)&selected, sizeof(int) * update_rows);
    cudaMallocHost((void**)&selected, sizeof(int) * update_rows);
    unsigned num_rows = 0;
	virg_query(v, &r, "select id from test where normalf20 = 1", selected, &num_rows);
	printf("Found %u rows\n", num_rows);
	
	virg_release(v, r);
	//virg_reader_copy(v, r, selected);
	// Determine temporary device storage requirements
	//void     *d_temp_storage = NULL;
	//size_t   temp_storage_bytes = 0;
	//cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, selected, selected, update_rows);
	// Allocate temporary storage
	//cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sorting operation
	//cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, selected, selected, update_rows);

	std::sort(selected, selected + update_rows);	
	
	update_time += time_val(TIME_NOW - start_time).count();

	virg_table_update(v, 0, selected, update_rows, (char*)buff2, sizeof(int) * update_rows, 1, 
		update_operation_time, update_ddio_time, update_persist_time, false);

	free(buff2);

	virg_query(v, &r, "select id, uniformi, normali5, normali20, uniformf, normalf5, normalf20 from test where normali20 < 700002 AND normali20 >= 699999");
	// output result column names
	for(j = 0; j < r->res->fixed_columns; j++)
		printf("%s\t", r->res->fixed_name[j]);
	printf("\n");

	// output result data
	do {
	    a = virg_reader_row(v, r);
	    if(a != VIRG_FAIL) {
		int *results = (int*)r->buffer;
		float *res = (float*)r->buffer;

		printf("%i\t%i\t%i\t%i\t%f\t%f\t%d\n", results[0], results[1], results[2], results[3], res[4], res[5], results[6]);
		}
	}while(a != VIRG_FAIL);
	virg_release(v, r);
/*
	double update_recov_time = 0;
	virg_recover_update(v, &update_recov_time);
	
	virg_query(v, &r, "select id, uniformi, normali5, normali20, uniformf, normalf5, normalf20 from test where normali20 < 700002 AND normali20 >= 699999");
	// output result column names
	for(j = 0; j < r->res->fixed_columns; j++)
		printf("%s\t", r->res->fixed_name[j]);
	printf("\n");

	// output result data
	do {
	    a = virg_reader_row(v, r);
	    if(a != VIRG_FAIL) {
		int *results = (int*)r->buffer;
		float *res = (float*)r->buffer;

		printf("%i\t%i\t%i\t%i\t%f\t%f\t%f\n", results[0], results[1], results[2], results[3], res[4], res[5], res[6]);
		}
	}while(a != VIRG_FAIL);


	// clean up after query
	virg_release(v, r);

    double insert_recov_time = 0;
    virg_recover_insert(v, &insert_recov_time);
	
	virg_query(v, &r, "select id, uniformi, normali5, normali20, uniformf, normalf5, normalf20 from test where normali20 < 700002 AND normali20 >= 699999");
	// output result column names
	for(j = 0; j < r->res->fixed_columns; j++)
		printf("%s\t", r->res->fixed_name[j]);
	printf("\n");

	// output result data
	do {
	    a = virg_reader_row(v, r);
	    if(a != VIRG_FAIL) {
		    int *results = (int*)r->buffer;
		    float *res = (float*)r->buffer;

		    printf("%i\t%i\t%i\t%i\t%f\t%f\t%f\n", results[0], results[1], results[2], results[3], res[4], res[5], res[6]);
		}
	}while(a != VIRG_FAIL);

	virg_release(v, r);
	*/
	virg_db_close(v);
	virg_close(v);

    printf("Insert %ld rows; update %ld rows\n", rows, update_rows);
        
    //printf("InsertRecovery\t%f\tms\nUpdateRecovery\t%f\tms\n", insert_recov_time / 1000000.0f, update_recov_time / 1000000.0f);
    
    printf("\nInsertTime\t%f\tms\n", insert_operation_time/1000000.0);  
    printf("DDIO time: %f ms\nPersist time: %f\n\n", insert_ddio_time/1000000.0, insert_persist_time/1000000.0);

    printf("UPDATE\nUpdate execution time: %f ms, Search: %f ms\n", update_operation_time/1000000.0, update_time / 1000000.0);  
    printf("UpdateTime\t%f\tms\n", update_operation_time/1000000.0 + update_time / 1000000.0);  
    printf("DDIO time: %f ms\nPersist time: %f\n\n", update_ddio_time/1000000.0, update_persist_time/1000000.0);

	return 0;
}

