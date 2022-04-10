#include "srad.h"
#include <stdio.h>

__global__ void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  long long cols, 
		  long long rows, 
		  float q0sqr
) 
{

  //block id
  long long bx = blockIdx.x;
  long long by = blockIdx.y;

  //thread id
  long long tx = threadIdx.x;
  long long ty = threadIdx.y;
  
  //indices
  long long index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    if (index < cols * rows) {
  long long index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  long long index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  long long index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  long long index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

        //load data to shared memory
        if ( by == 0 ){
            north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
            south[ty][tx] = J_cuda[index_s];
        }
        else if ( by == gridDim.y - 1 ){
            north[ty][tx] = J_cuda[index_n]; 
            south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
        }
        else {
            north[ty][tx] = J_cuda[index_n]; 
            south[ty][tx] = J_cuda[index_s];
        }
        __syncthreads();

        if ( bx == 0 ){
            west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
            east[ty][tx] = J_cuda[index_e]; 
        }
        else if ( bx == gridDim.x - 1 ){
            west[ty][tx] = J_cuda[index_w];
            east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
        }
        else {
            west[ty][tx] = J_cuda[index_w];
            east[ty][tx] = J_cuda[index_e];
        }

        __syncthreads();

        temp[ty][tx] = J_cuda[index];

        __syncthreads();
   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

        // saturate diffusion coefficent
	    if (c < 0){temp_result[ty][tx] = 0;}
	    else if (c > 1) {temp_result[ty][tx] = 1;}
	    else {temp_result[ty][tx] = c;}

            //__syncthreads(); // Seems unneeded?

        C_cuda[index] = temp_result[ty][tx];
	    E_C[index] = e;
	    W_C[index] = w;
	    S_C[index] = s;
	    N_C[index] = n;
    }
}

__global__ void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  long long cols, 
		  long long rows, 
		  float lambda,
		  float q0sqr
) 
{
	//block id
	long long bx = blockIdx.x;
    long long by = blockIdx.y;

	//thread id
    long long tx = threadIdx.x;
    long long ty = threadIdx.y;

	//indices
    long long index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    if (index < cols * rows)
    {
        long long index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
        long long index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
        float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    __syncthreads();

	if ( by == gridDim.y - 1 ){
	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
    else {
        south_c[ty][tx] = C_cuda[index_s];
    }
	__syncthreads();

	if ( bx == gridDim.x - 1 ){
	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
	}
    else {
        east_c[ty][tx] = C_cuda[index_e];
	 }

    __syncthreads();
  
    c_cuda_temp[ty][tx]      = C_cuda[index];

    __syncthreads();

	cc = c_cuda_temp[ty][tx];

       if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	    cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
       } 
       else if ( tx == BLOCK_SIZE -1 ){ //e
	    cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
       }
       else if ( ty == BLOCK_SIZE -1){ //s
	    cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
       }
       else{ //the data elements which are not on the borders 
	    cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
       }

       // divergence (equ 58)
       d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

       // image update (equ 61)
       c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

       //__syncthreads(); // Seems unneeded?
                  
       J_cuda[index] = c_cuda_result[ty][tx];
    }
}
