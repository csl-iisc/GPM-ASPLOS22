// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"
#include "libgpm.cuh"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter> <no. of thd>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	fprintf(stderr, "\t<no. of thd>   - number of CPU threads\n");
	
	exit(1);
}

double kernel_time = 0, persist_time = 0, ddio_time = 0;

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


void
runTest( int argc, char** argv) 
{
    long long rows, cols, size_I, size_R, niter = 10, iter, nthread;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
    float *J_cuda_out;
    float *C_cuda;

/*#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

#endif
*/

	unsigned long long r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc >= 10)
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
		if(argc >= 10)
			nthread = atoi(argv[9]);
		else
			nthread = 32;
		
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


    const char *path_j = "persist_j.dat";
    const char *path_c = "persist_c.dat";
    const char *path_e_c = "persist_e_c.dat";
    const char *path_w_c = "persist_w_c.dat";
    const char *path_s_c = "persist_s_c.dat";
    const char *path_n_c = "persist_n_c.dat";
    const char *path_j_out = "persist_j_out.dat";

    printf("Randomizing the input matrix\n");
    //Generate a random matrix
    random_matrix(I, rows, cols);
    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]);
    }

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    

    size_t len = sizeof(float) * size_I;   
    C_cuda = (float*) gpm_map_file(path_c, len, true);
    dE    = (float*) gpm_map_file(path_e_c, len, true);
    dW    = (float*) gpm_map_file(path_w_c, len, true);
    dN    = (float*) gpm_map_file(path_n_c, len, true);
    dS    = (float*) gpm_map_file(path_s_c, len, true);
    J_cuda_out = (float*) gpm_map_file(path_j_out, len, true);
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

//#endif

/*
#ifdef GPU

	//Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);

	
#endif 
*/

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

        
    	auto start = std::chrono::high_resolution_clock::now(); 
    	#pragma omp parallel for num_threads(nthread)
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			    pmem_flush(&dN[k], sizeof(float));
			    pmem_flush(&dS[k], sizeof(float));
			    pmem_flush(&dW[k], sizeof(float));
			    pmem_flush(&dE[k], sizeof(float));
                pmem_drain();
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                C_cuda[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (C_cuda[k] < 0) {C_cuda[k] = 0;}
                else if (C_cuda[k] > 1) {C_cuda[k] = 1;}
			}
		}
		
    	#pragma omp parallel for num_threads(nthread)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J_cuda_out[k] = J[k] + 0.25*lambda*D;
            }
		}        
		memcpy(J, J_cuda_out, sizeof(float) * size_I);
		kernel_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;    
		printf("Iteration %d, time so far: %f\n", iter, kernel_time);

	/*
	#ifdef GPU

		//Currently the input size must be divided by 16 - the block size
		int block_x = cols/BLOCK_SIZE ;
		int block_y = rows/BLOCK_SIZE ;

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(block_x , block_y);
		

		//Copy data from main memory to device memory
		cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

		//Run kernels
		srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
		srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

		//Copy data from device memory to main memory
		cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

	#endif
	*/   
	}

    cudaThreadSynchronize();

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

	printf("runtime: %f\n", kernel_time);
	printf("Computation Done\n");

	free(I);
	free(J);
//#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    //free(dN); free(dS); free(dW); free(dE);
//#endif
/*
#ifdef GPU
    cudaFree(C_cuda);
	cudaFree(J_cuda);
	cudaFree(E_C);
	cudaFree(W_C);
	cudaFree(N_C);
	cudaFree(S_C);
#endif 
*/
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

