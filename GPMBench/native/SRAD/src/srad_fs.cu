// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel_fs.cu"

#include <unistd.h>
#include <thread>
#include <assert.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include "libgpm.cuh"

#define TIME_NOW std::chrono::high_resolution_clock::now()
double operation_time = 0, persist_time = 0, memcpy_time = 0, pwrite_time = 0;

#define CHECK(len1, len2) \
    if(len1 != len2) \
        printf("Line %d: Error, only wrote %lld of %llu (errno=%d)\n", __LINE__, len1, len2, errno);

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    long long rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

    /*#ifdef CPU
    float Jc, G2, L, num, den, qsqr;
    int *iN,*iS,*jE,*jW, k;
    float *dN,*dS,*dW,*dE;
    float cN,cS,cW,cE,D;
    #endif*/

//#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

//#endif

	unsigned long long r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		
	}
    else{
        usage(argc, argv);
    }

	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


    float *dN = (float*)malloc(sizeof(float) * size_I);
    float *dS = (float*)malloc(sizeof(float) * size_I);
    float *dW = (float*)malloc(sizeof(float) * size_I);
    float *dE = (float*)malloc(sizeof(float) * size_I);
    float *C = (float*)malloc(sizeof(float) * size_I);
    


	//Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);

    int fd_J_cuda = open("/pmem/pm_J_cuda",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    int fd_C_cuda = open("/pmem/pm_C_cuda",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    int fd_E_C = open("/pmem/pm_E_C",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    int fd_N_C = open("/pmem/pm_N_C",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    int fd_S_C = open("/pmem/pm_S_C",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    int fd_W_C = open("/pmem/pm_W_C",  O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
	
	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (long long i=r1; i<=r2; i++) {
            for (long long j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);



//#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    

	//Copy data from main memory to device memory
	cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

	//Run kernels
    auto start = TIME_NOW; 
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 
    cudaDeviceSynchronize(); 
    operation_time += (TIME_NOW - start).count(); 

	printf("Iteration %d\n", iter);
	//Copy data from device memory to main memory
    start = TIME_NOW; 
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(dN, N_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(dE, E_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(dS, S_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(dW, W_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, C_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    memcpy_time += (TIME_NOW - start).count(); 
    start = TIME_NOW; 
    long long len;
    len = pmem_write(fd_J_cuda, J, sizeof(float) * size_I, 0);
    CHECK(len, sizeof(float) * size_I);
    len = pmem_write(fd_E_C, dE, sizeof(float) * size_I, 0); 
    CHECK(len, sizeof(float) * size_I);
    len = pmem_write(fd_N_C, dN, sizeof(float) * size_I, 0); 
    CHECK(len, sizeof(float) * size_I);
    len = pmem_write(fd_S_C, dS, sizeof(float) * size_I, 0); 
    CHECK(len, sizeof(float) * size_I);
    len = pmem_write(fd_W_C, dW, sizeof(float) * size_I, 0); 
    CHECK(len, sizeof(float) * size_I);
    len = pmem_write(fd_C_cuda, C, sizeof(float) * size_I, 0); 
    CHECK(len, sizeof(float) * size_I);
    pwrite_time += (TIME_NOW - start).count(); 
    start = TIME_NOW; 
    fsync(fd_J_cuda);
    fsync(fd_E_C);
    fsync(fd_N_C);
    fsync(fd_S_C);
    fsync(fd_W_C);
    fsync(fd_C_cuda);
    persist_time += (TIME_NOW - start).count();  
}

    cudaDeviceSynchronize();

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for(long long i = 0 ; i < rows ; i++){
		for (long long j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");
    printf("\nOperation execution time: %f ms\n", operation_time/1000000.0);
    printf("memcpy time: %f msec \t pwrite time: %f \t persist time: %f \t\n", memcpy_time/1000000.0f, pwrite_time/1000000.0f, persist_time/1000000.0f);
    printf("\nruntime: %f ms\n", (operation_time + memcpy_time + pwrite_time + persist_time)/1000000.0);


	free(I);
	free(J);
//#ifdef GPU
    cudaFree(C_cuda);
	cudaFree(J_cuda);
	cudaFree(E_C);
	cudaFree(W_C);
	cudaFree(N_C);
	cudaFree(S_C);
//#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

