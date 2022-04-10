// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"
#include "bandwidth_analysis.cuh"
#include "libgpm.cuh"
#include "gpm-helper.cuh"
#include <time.h>
extern "C"
{
#include "change-ddio.h"
}


// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"
const bool recovery = false; 

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

__global__ void setMemory(float *arr, size_t size, float value)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = id; i < size; i += blockDim.x * gridDim.x)
        arr[i] = value;
}

double kernel_time = 0, persist_time = 0, ddio_time = 0, runtime = 0, memcpy_time = 0;

void runTest( int argc, char** argv) 
{
    ddio_on();
    long long rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

    /*#ifdef CPU
    float Jc, G2, L, num, den, qsqr;
    int *iN,*iS,*jE,*jW, k;
    float *dN,*dS,*dW,*dE;
    float cN,cS,cW,cE,D;
    #endif*/

    //#ifdef GPU

    float *J_cuda, *J_cuda_out;
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
    c = (float *)malloc(sizeof(float)* size_I) ;

    //E is east, N is north, W is west, S is south 
    //I is mostly the x cordinate? because it is associated with N and S
    //J is mostly the y cordinate? because it is associated with W and E 

    const char *path_j = "persist_j.dat";
    const char *path_c = "persist_c.dat";
    const char *path_e_c = "persist_e_c.dat";
    const char *path_w_c = "persist_w_c.dat";
    const char *path_s_c = "persist_s_c.dat";
    const char *path_n_c = "persist_n_c.dat";
    const char *path_j_out = "persist_j_out.dat";

    size_t len = sizeof(float) * size_I;
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    auto start = std::chrono::high_resolution_clock::now(); 
    cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);
    memcpy_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
    printf("Randomizing the input matrix\n");
    //Generate a random matrix
    random_matrix(I, rows, cols);
    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]);
    }
         
    if(!recovery) {
        //File does not exist : then create new files  
        //Allocate device memory
        printf("Allocating PM\n");  
        //J_cuda = (float*) gpm_map_file(path_j, len, true);
        C_cuda = (float*) gpm_map_file(path_c, len, true);
        E_C    = (float*) gpm_map_file(path_e_c, len, true);
        W_C    = (float*) gpm_map_file(path_w_c, len, true);
        N_C    = (float*) gpm_map_file(path_n_c, len, true);
        S_C    = (float*) gpm_map_file(path_s_c, len, true);
        J_cuda_out = (float*) gpm_map_file(path_j_out, len, true);
       //#endif 
        //Initialized the c_cuda variable to recover from any failure of kernel 1
       
        printf("Start copying to NVM of original data\n");  
        //pmem_j_cuda_out is the j_cuda_output file for volatile 
        //pmem_j_cuda_out is only written to when executing for volatile 
        //pmem_j_cuda is the input file, it keeps the original input which is J
        //pmem_j_cuda is read into j_cuda for both volatile and GPM-far 
        size_t mapped_len; 
        int is_pmemp; 
        float *pm_j_cuda = (float*)pmem_map_file("/pmem/pmem_j_data.dat", len, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
        //cudaHostRegister(pm_j_cuda, len,  0);
#if defined(NVM_ALLOC_GPU) && !defined(EMULATE_NVM)
        float *pm_j_cuda_out = (float*)pmem_map_file("/pmem/pmem_j_out_data.dat", len, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmemp);
        //cudaHostRegister(pm_j_cuda_out, len,  0);
#endif

        //Copying input data to the scratchpad J_cuda from pm_j_cuda 
        //Included for both coarse and fine-grained 
        auto start = std::chrono::high_resolution_clock::now(); 
        //cudaMemcpy(J_cuda, pm_j_cuda, sizeof(float) * size_I, cudaMemcpyHostToDevice);
        kernel_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;

        start = std::chrono::high_resolution_clock::now(); 
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
        ddio_off();
#endif
		ddio_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
        for (iter=0; iter< niter; iter++){
            int sum=0; sum2=0;
            for (int i=r1; i<=r2; i++) {
                for (int j=c1; j<=c2; j++) {
                    tmp   = J[i * cols + j];
                    sum  += tmp ;
                    sum2 += tmp*tmp;
                }
            }
            meanROI = sum / size_R;
            varROI  = (sum2 / size_R) - meanROI * meanROI;
            q0sqr   = varROI / (meanROI * meanROI);
			setMemory<<<1, 1024>>>(C_cuda, size_I, -1.0);
			setMemory<<<1, 1024>>>(J_cuda_out, size_I, -1.0);
            // Reinit for every iteration
            printf("%d: Copy finished\n", iter);
            
            cudaDeviceSynchronize();

            //Currently the input size must be divided by 16 - the block size
            int block_x = cols/BLOCK_SIZE ;
            int block_y = rows/BLOCK_SIZE ;

            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid(block_x , block_y);

            //Copy data from main memory to device memory

            //Run kernels
            //START_BW_MONITOR2("bw_srad_gpm.csv");
        	start = std::chrono::high_resolution_clock::now(); 
            srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
            srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr, J_cuda_out); 
            cudaDeviceSynchronize();
            kernel_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
            printf("Time: %f\n", kernel_time);
    		OUTPUT_STATS
            //STOP_BW_MONITOR
            //OUTPUT_STATS
#ifdef GPM_WDP
            start = std::chrono::high_resolution_clock::now(); 
            pmem_persist(J_cuda_out, sizeof(float) * size_I); 
            pmem_persist(C_cuda, sizeof(float) * size_I); 
            pmem_persist(E_C, sizeof(float) * size_I); 
            pmem_persist(N_C, sizeof(float) * size_I); 
            pmem_persist(S_C, sizeof(float) * size_I); 
            pmem_persist(W_C, sizeof(float) * size_I); 
            persist_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
#endif
            start = std::chrono::high_resolution_clock::now(); 
#if defined(NVM_ALLOC_GPU) && !defined(EMULATE_NVM)
            cudaMemcpy(pm_j_cuda_out, J_cuda_out, sizeof(float) * size_I, cudaMemcpyDeviceToHost);  
#endif
            kernel_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
            //Copy data from device memory to main memory
            start = std::chrono::high_resolution_clock::now(); 
            cudaMemcpy(J_cuda, J_cuda_out, sizeof(float) * size_I, cudaMemcpyDeviceToDevice); // Copy back for next iteration
		    memcpy_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
            //#endif
        }
        start = std::chrono::high_resolution_clock::now(); 
#if defined(NVM_ALLOC_CPU) && !defined(FAKE_NVM) && !defined(GPM_WDP)
        ddio_on(); 
#endif
		ddio_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
#if defined(NVM_ALLOC_GPU) && !defined(EMULATE_NVM)
         //gpm_unmap(path_j, pm_j_cuda, len);
#endif
        gpm_unmap(path_c, C_cuda, len);
        cudaDeviceSynchronize();
        printf("Operation time: %f ms\n", (double)kernel_time);
        printf("DDIO time: %f ms\n", (double)ddio_time);
        printf("PersistTime: %f ms\n", (double)persist_time);
        printf("memcpy_time: %f ms\n", (double)memcpy_time);
        runtime = kernel_time + persist_time;
        printf("runtime: %f ms\n", runtime);
    } 
    else {
        //Files exist : open the existing files 
        size_t lenc = 0, lenec = 0, lenwc = 0, lennc = 0,lensc = 0,lenj = 0; 
        auto start_recover = std::chrono::high_resolution_clock::now(); 
        C_cuda = (float*) gpm_map_file(path_c, lenc, false);
        E_C    = (float*) gpm_map_file(path_e_c, lenec, false);
        W_C    = (float*) gpm_map_file(path_w_c, lenwc, false);
        N_C    = (float*) gpm_map_file(path_n_c, lennc, false);
        S_C    = (float*) gpm_map_file(path_s_c, lensc, false);
        J_cuda_out = (float*) gpm_map_file(path_j_out, lenj, false);
        sum = 0, sum2 = 0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI * meanROI;
        q0sqr   = varROI / (meanROI * meanROI);
        double recover_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_recover).count() / 1000.0;
        printf("Recover time: %f ms\n", recover_time);

        //Call the kernels 
        int block_x = cols/BLOCK_SIZE ;
        int block_y = rows/BLOCK_SIZE ;

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(block_x , block_y);
        srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
        srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr, J_cuda_out);
        cudaMemcpy(J, J_cuda_out, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

#ifdef OUTPUT
    //Printing output	
    printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
        for ( int j = 0 ; j < cols ; j++){
            printf("%.5f ", J[i * cols + j]); 
        }	
        printf("\n"); 
    }
#endif 

    printf("Computation Done\n");

    free(I);
    free(J);
/*#ifdef CPU
    free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif*/
//#ifdef GPU
     cudaFree(J_cuda);
     //cudaFree(C_cuda);
     gpm_unmap(path_e_c, E_C, len);
     gpm_unmap(path_w_c, W_C, len);
     gpm_unmap(path_n_c, N_C, len);
     gpm_unmap(path_s_c, S_C, len);
     gpm_unmap(path_j_out, J_cuda_out, len);
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

