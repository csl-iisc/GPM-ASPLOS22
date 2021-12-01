/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef _GPU_HASH_H_
#define _GPU_HASH_H_
extern "C" 
{
#include "change-ddio.h"
}
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "libgpm.cuh"

#define TIME_NOW std::chrono::high_resolution_clock::now()

#if defined(__CUDACC__) // NVCC
	#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
	#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
	#define MY_ALIGN(n) __declspec(align(n))
#else
	#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

typedef uint32_t sign_t;
typedef uint64_t loc_t;

typedef uint32_t hash_t;
typedef uint32_t log_t;

/* In non-caching mode, the memory transaction unit is 32B, therefore, we define
the bucket size to be a multiple of 32B, meaning at least 4 elements, ELEM_NUM_P >= 2 */

#define ELEM_SIG_SIZE	8 // FIXME: sizeof(hash_t) + sizeof(sign_t)
#define ELEM_SIZE_P		4 // 2^3 bytes per element
#define ELEM_NUM_P		3 // 2^ELEM_NUM_P elements per bucket
#define ELEM_NUM		(1 << ELEM_NUM_P)
#define UNIT_THREAD_NUM_P 1
#define UNIT_THREAD_NUM (1 << UNIT_THREAD_NUM_P)
#define WARP_SIZE 32


//#define MEM_P			(31) // 2^31, 2GB memory
#define MEM_P			(32) // 2^30, 1GB memory

#define BUC_P			(ELEM_NUM_P + ELEM_SIZE_P) // 2^3 is element size
/* Since 5 bits are for insert bufs, at most 32-5=27 can be used
 * as HASH_MASK. And the maximum is 1<<27 buckets, 1<<34 memory,
 * 16 GB memory */
#define HASH_MASK		((1 << (MEM_P - BUC_P)) - 1) // 2<<22 -1

#define BUC_NUM			(1 << (MEM_P - BUC_P)) 
#define HT_SIZE			((long)1 << (long)(MEM_P))
  
// for insert, it is divided into 8 blocks
#define IBLOCK_P		10  // 2^3 = 8
#define INSERT_BLOCK	(1 << IBLOCK_P) // number
#define BLOCK_HASH_MASK	((1 << (MEM_P - BUC_P - IBLOCK_P)) - 1)


#define HASH_2CHOICE	1
//#define HASH_CUCKOO		1
#define MAX_CUCKOO_NUM	5	/* maximum cuckoo evict number */



typedef MY_ALIGN(128) struct bucket_s {
	sign_t sig[ELEM_NUM];
	loc_t loc[ELEM_NUM];
} bucket_t;

// for search
typedef MY_ALIGN(8) struct selem_s {
	//uint8_t sig[ELEM_SIG_SIZE];
	sign_t	sig;
	hash_t	hash;
	selem_s(int a, int b) {sig = a, hash = b;}
	selem_s() {}
	bool operator < (const selem_s &b) const {
		if(sig != b.sig)
			return sig < b.sig;
		return hash < b.hash;
	}
} selem_t;

typedef MY_ALIGN(8) union selem_shared_s {
	//uint8_t sig[ELEM_SIG_SIZE];
	selem_t elem_s;
	long long elem_u;
} selem_shared_t;

// for insert and delete
typedef struct ielem_s {
	sign_t	sig;
	hash_t	hash;
	loc_t   loc;
} ielem_t;

typedef struct ielem_s delem_t;


void gpu_hash_search(
		selem_t 	*in,
		loc_t		*out,
		bucket_t	*hash_table,
		int			num_elem,
		int			num_thread,
		int			threads_per_blk,
		cudaStream_t stream);

void gpu_hash_insert(
		bucket_t	*hash_table,
		ielem_t		**blk_input,
		int			*blk_elem_num,
		int			num_blks,
		int         num_elems,
		int         fd,
		cudaStream_t stream);

void gpu_hash_delete(
		delem_t 	*in,
		bucket_t	*hash_table,
		int			num_elem,
		int 		num_thread,
		int			threads_per_blk,
		int         fd,
		cudaStream_t stream);

void gpu_delete_insert(
		bucket_t		*hash_table,
		delem_t			*delete_in,
		uint32_t		num_delete_job,
		ielem_t			**insert_blk_input,
		int				*insert_blk_elem_num,
		int				num_insert_blks,
		uint32_t		num_delete_thread,
		uint32_t		threads_per_blk,
		cudaStream_t	stream);

#define CUDA_SAFE_CALL(call) do {                                            \
	cudaError_t err = call;                                                    \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				 __FILE__, __LINE__, cudaGetErrorString( err) );             \
		exit(EXIT_FAILURE);                                                  \
	} } while (0)

#define CUDA_SAFE_CALL_SYNC(call) do {                                       \
	CUDA_SAFE_CALL_NO_SYNC(call);                                            \
	cudaError_t err |= cudaDeviceSynchronize();                                \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				__FILE__, __LINE__, cudaGetErrorString( err) );              \
		exit(EXIT_FAILURE);                                                  \
	} } while (0)

#endif

/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

__global__ void hash_search(
		selem_t			*in,
		loc_t			*out,
		bucket_t		*hash_table,
		int				total_elem_num,
		int				thread_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int id = 0;
	// (1 << ELEM_NUM_P) threads to cooperate for one element
	int step = thread_num >> ELEM_NUM_P;
	int ballot;
	
	// Calculate mask of elements coordinating insert
	unsigned THREAD_MASK = 0;
	for(int i = 0; i < ELEM_NUM; ++i)
	    THREAD_MASK |= ((unsigned)1 << (unsigned)i);
	THREAD_MASK <<= ((idx % WARP_SIZE) / ELEM_NUM) * ELEM_NUM;
	
	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;

	int bit_move;
	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		selem_t *elem = &(in[id]);

		// TODO: force out buffer to be memset zero so that
		// this simultaneous memory write can be avoided
		// out[id << 1] = 0;
		// out[id << 1 + 1] = 0;

		bucket_t *b = &(hash_table[elem->hash & HASH_MASK]);
		if (b->sig[simd_lane] == elem->sig) {
			out[id << 1] = b->loc[simd_lane];
		}
		ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == elem->sig);
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		//if (ballot != 0) {
		//	continue;
		//}

		//b = &(hash_table[(elem->hash ^ elem->sig) & HASH_MASK]);
		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		if (b->sig[simd_lane] == elem->sig) {
			out[(id << 1) + 1] = b->loc[simd_lane];
		}
	}

	return;
}

__global__ void hash_insert_2choice(
		bucket_t		*hash_table,
		ielem_t			**blk_input,
		int				*blk_elem_num)
{
	__shared__ int count[2];
	ielem_t *in = blk_input[blockIdx.x];
	int total_elem_num = blk_elem_num[blockIdx.x];
	// 16 threads to cooperate for one element
	int step = blockDim.x >> ELEM_NUM_P;
	int idx = threadIdx.x;

	// Calculate mask of elements coordinating insert
	unsigned THREAD_MASK = 0;
	for(int i = 0; i < ELEM_NUM; ++i)
	    THREAD_MASK |= ((unsigned)1 << (unsigned)i);
	THREAD_MASK <<= ((idx % WARP_SIZE) / ELEM_NUM) * ELEM_NUM;

	int id = 0, hash;
	bucket_t *b;
	int chosen_simd;
	int ballot, ml_mask;

	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;
	int bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);
	for (id = elem_id; id < total_elem_num; id += step) {
		__syncthreads();
		if(idx == 0) {
			count[0] = 0;
			count[1] = 0;
		}
		__syncthreads();
	
	
		ielem_t *elem = &(in[id]);

		sign_t sig = elem->sig;
		if (elem->sig == 0 && elem->loc == 0) {
			//printf("error, all is zero\n");
			continue;
		}

		bool done = false;
		b = &(hash_table[elem->hash & HASH_MASK]);

		/*=====================================================================
		 * The double __syncthreads() seems useless in else, this is to match the two in
		 * if (chosen_simd == simd_lane). As is stated in the paper <Demystifying GPU 
		 * Microarchitecture through Microbenchmarking>, the __syncthreads() will not go
		 * wrong if not all threads in one wrap reach it, however, the wraps in the same
		 * block need to reach a __syncthreads(), even if they are not on the same line */
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == elem->sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for bottom half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (!done && 0 != ballot) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
	            // Perform operation
				b->loc[simd_lane] = elem->loc;
			}
			done = true;
			atomicAdd(&count[0], 1);
			atomicAdd(&count[1], 1);
			//continue;
		}

		/*=====================================================================*/
		/* Next we try to insert, the while look breaks if ballot == 0, and the 
		 * __syncthreads() in the two loops match if the code path divergent between
		 * the warps in a block. Or some will terminate, or process the next element. 
		 * FIXME: if some wrap go to process next element, some stays here, will this
		 * lead to mismatch in __syncthreads()? If it does, we should launch one thread
		 * for each element. God knows what nVidia GPU will behave. FIXME;
		 * Here we write b->loc, and the above code also write b->loc. This will not
		 * lead to conflicts, because here all the signatures are 0, while the aboves
		 * are all non-zero */

		/* Major Location : use last 4 bits of signature */
		ml_mask = (1 << (elem->sig & ((1 << ELEM_NUM_P) - 1))) - 1;
		bool fail = false;
		/* find the empty slot for insertion */
		while (1) {
			if(!done && !fail) {
				ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == 0);
				ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
				/* 1010|0011 => 0000 0011 1010 0000, 16 bits to 32 bits*/
				ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
				if (ballot != 0) {
					chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
					if (simd_lane == chosen_simd) {
						b->sig[simd_lane] = sig;
					}
				} else {
					fail = true;
					atomicAdd(&count[0], 1);
				}
			}

			__syncthreads();
			if(((volatile int*)count)[0] == blockDim.x)
				break;
			if(!done && !fail) {
				if (ballot != 0) {
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							b->loc[simd_lane] = elem->loc;
						}
						done = true;
						atomicAdd(&count[0], 1);
						atomicAdd(&count[1], 1);
	//					goto finish;
					}
				}
			}
		}


		/* ==== try next bucket ==== */


		hash = (((elem->hash ^ sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		/*=====================================================================*/
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == elem->sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for bottom half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (!done && 0 != ballot) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
				b->loc[simd_lane] = elem->loc;
			}
			done = true;
			atomicAdd(&count[0], 1);
			atomicAdd(&count[1], 1);
			//continue;
		}
		
		fail = false;
		while (1) {
	        if(!done && !fail) {
				ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == 0);
				ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
				ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
				if (ballot != 0) {
					chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
					if (simd_lane == chosen_simd) {
						b->sig[simd_lane] = sig;
					}
				} else {
					/* No available slot.
					 * Get a Major location between 0 and 15 for insertion */
					chosen_simd = elem->sig & ((1 << ELEM_NUM_P) - 1);
					if (simd_lane == chosen_simd) {
						b->sig[simd_lane] = sig;
					}
					/* we only try insert once if there are no empty slots,
					 * because conflicted items on the same chosen_simd will
					 * keep conflicting. 
					 */
					//break;
					fail = true;
					atomicAdd(&count[1], 1);
				}
			}

			//__syncwarp(THREAD_MASK);
			__syncthreads();
			if(((volatile int*)count)[1] == blockDim.x)
				break;
			
			/* chosen_simd controls one thread in a half warp
			 * enters this */
	        if(!done && !fail) {
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						b->loc[simd_lane] = elem->loc;
					}
					done = true;
					atomicAdd(&count[1], 1);
					//goto finish;
				}
			}
		}

finish:
		;
	}

	return;
}

__global__ void hash_insert_cuckoo(
		bucket_t		*hash_table,
		ielem_t			**blk_input,
		int				*blk_elem_num)
{
	ielem_t *in = blk_input[blockIdx.x];
	int total_elem_num = blk_elem_num[blockIdx.x];
	// 16 threads to cooperate for one element
	int step = blockDim.x >> ELEM_NUM_P;
	int idx = threadIdx.x;

	// Calculate mask of elements coordinating insert
	unsigned THREAD_MASK = 0;
	for(int i = 0; i < ELEM_NUM; ++i)
	    THREAD_MASK |= ((unsigned)1 << (unsigned)i);
	THREAD_MASK <<= (idx / ELEM_NUM) * ELEM_NUM;

	hash_t hash, second_hash;
	loc_t loc, new_loc;
	sign_t sig, new_sig;

	int id;
	int cuckoo_num;
	bucket_t *b;
	int chosen_simd;
	int ballot, ml_mask;

	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;
	int bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		ielem_t *elem = &(in[id]);

		if (elem->sig == 0 && elem->loc == 0) {
			printf("error, all is zero\n");
			continue;
		}

		sig = elem->sig;
		hash = elem->hash;
		loc = elem->loc;

		b = &(hash_table[hash & HASH_MASK]);

		/*=====================================================================
		 * The double __syncthreads() seems useless in else, this is to match the two in
		 * if (chosen_simd == simd_lane). As is stated in the paper <Demystifying GPU 
		 * Microarchitecture through Microbenchmarking>, the __syncthreads() will not go
		 * wrong if not all threads in one wrap reach it. However, the wraps in the same
		 * block need to reach a __syncthreads(), even if they are not on the same line */
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for second half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (ballot != 0) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
				b->loc[simd_lane] = loc;
			}
			continue;
		}

		/*=====================================================================*/
		/* Next we try to insert, the while loop breaks if ballot == 0, and the 
		 * __syncthreads() in the two loops match if the code path divergent between
		 * the warps in a block. Or some will terminate, or process the next element. 
		 * FIXME: if some wrap go to process next element, some stays here, will this
		 * lead to mismatch in __syncthreads()? If it does, we should launch one thread
		 * for each element. God knows what nVidia GPU will behave. FIXME;
		 * Here we write b->loc, and the above code also write b->loc. This will not
		 * lead to conflicts, because here all the signatures are 0, while the aboves
		 * are all non-zero */

		/* Major Location : use last 4 bits of signature */
		ml_mask = (1 << (sig & ((1 << ELEM_NUM_P) - 1))) - 1;
		/* find the empty slot for insertion */
		while (1) {
			ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == 0);
			ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
			/* 1010|0011 => 0000 0011 1010 0000, 16 bits to 32 bits*/
			ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
			if (ballot != 0) {
				chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			}

			__syncwarp(THREAD_MASK);

			if (ballot != 0) {
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						b->loc[simd_lane] = loc;
					}
					goto finish;
				}
			} else {
				break;
			}
		}


		/* ==== try next bucket ==== */
		cuckoo_num = 0;

cuckoo_evict:
		second_hash = (((hash ^ sig) & BLOCK_HASH_MASK) 
				| (hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[second_hash]);
		/*=====================================================================*/
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for second half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (0 != ballot) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
				b->loc[simd_lane] = loc;
			}
			continue;
		}

		while (1) {
			ballot = __ballot_sync(THREAD_MASK, b->sig[simd_lane] == 0);
			ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
			ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
			if (ballot != 0) {
				chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			} else {
				/* No available slot.
				 * Get a Major location between 0 and 15 for insertion */
				chosen_simd = elem->sig & ((1 << ELEM_NUM_P) - 1);
				if (cuckoo_num < MAX_CUCKOO_NUM) {
					/* record the signature to be evicted */
					new_sig = b->sig[chosen_simd];
					new_loc = b->loc[chosen_simd];
				}
			}
			
			/* synchronize before the signature is written by others */
			__syncwarp(THREAD_MASK);

			if (ballot != 0) {
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			} else {
				/* two situations to handle: 1) cuckoo_num < MAX_CUCKOO_NUM,
				 * replace one element, and reinsert it into its alternative bucket.
				 * 2) cuckoo_num >= MAX_CUCKOO_NUM.
				 * The cuckoo evict exceed the maximum insert time, replace the element.
				 * In each case, we write the signature first.*/
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			}
			__syncwarp(THREAD_MASK);

			if (ballot != 0) {
				/* write the empty slot or try again when conflict */
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						b->loc[simd_lane] = loc;
					}
					goto finish;
				}
			} else {
				if (cuckoo_num < MAX_CUCKOO_NUM) {
					cuckoo_num ++;
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							b->loc[simd_lane] = loc;
						}
						sig = new_sig;
						loc = new_loc;
						goto cuckoo_evict;
					} else {
						/* if there is conflict when writing the signature,
						 * it has been replaced by another one. Reinserting
						 * the element is meaningless, because it will evict
						 * the one that is just inserted. Only one will survive,
						 * we just give up the failed one */
						goto finish;
					}
				} else {
					/* exceed the maximum insert time, evict one */
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							b->loc[simd_lane] = loc;
						}
					}
					/* whether or not succesfully inserted, finish */
					goto finish;
				}
			}
		}

finish:
		;
		//now we get to the next element
	}

	return;
}

__global__ void hash_delete(
		delem_t			*in,
		bucket_t		*hash_table,
		int				total_elem_num,
		int				thread_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int id = 0;
	// 16 threads to cooperate for one element
	int step = thread_num >> ELEM_NUM_P;
	int ballot;
	
	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;
	bucket_t *b;

	int bit_move;
	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		delem_t *elem = &(in[id]);

		b = &(hash_table[elem->hash & HASH_MASK]);
		/* first perform ballot */
		ballot = __ballot_sync(__activemask(), b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc);

		if (b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc) {
			b->sig[simd_lane] = 0;
			
		}

		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (ballot != 0) {
			continue;
		}

		//b = &(hash_table[(elem->hash ^ elem->sig) & HASH_MASK]);
		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		if (b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc) {
			b->sig[simd_lane] = 0;
		}
	}

	return;
}

void persist_data(void *file, void *d_table, double &mem_time, double &persist_time)
{
    auto start = TIME_NOW;
    cudaMemcpy(file, d_table, HT_SIZE, cudaMemcpyDeviceToHost);
    mem_time += (TIME_NOW - start).count();
    start = TIME_NOW;
    printf("MM writes:\t%ld\n", HT_SIZE);
    if(pmem_is_pmem(file, HT_SIZE))
        pmem_mt_persist(file, HT_SIZE);
    else
        pmem_msync(file, HT_SIZE);
    persist_time += (TIME_NOW - start).count();
}

void gpu_hash_search(
		selem_t 	*in,
		loc_t		*out,
		bucket_t	*hash_table,
		int			num_elem,
		int 		num_thread,
		int			threads_per_blk,
		cudaStream_t stream)
{
	int num_blks = (num_thread + threads_per_blk - 1) / threads_per_blk;
	assert(num_thread > threads_per_blk);
	assert(threads_per_blk <= 1024);
	//assert(num_thread <= num_elem);
	if (num_thread % 32 != 0) {
		num_thread = (num_thread + 31) & 0xffe0;
	}
	assert(num_thread % 32 == 0);

	/* prefer L1 cache rather than shared memory,
	   the other is cudaFuncCachePreferShared
	*/
	//void (*funcPtr)(selem_t *, loc_t *, bucket_t *, int, int);
	//funcPtr = hash_search;
	//cudaFuncSetCacheConfig(*funcPtr, cudaFuncCachePreferL1);
	

	//printf("stream=%d, threads_per_blk =%d, num_blks = %d\n", stream, threads_per_blk, num_blks);
	if (stream == 0) {
		hash_search<<<num_blks, threads_per_blk>>>(
			in, out, hash_table, num_elem, num_thread);
	} else  {
		hash_search<<<num_blks, threads_per_blk, 0, stream>>>(
			in, out, hash_table, num_elem, num_thread);
	}

	return;
}

/* num_blks is the array size of blk_input and blk_output */
void gpu_hash_insert(
		bucket_t	*hash_table,
		ielem_t		**blk_input,
		int			*blk_elem_num,
		int			num_blks,
		int         num_elems,
		void        *imkv_file,
		cudaStream_t stream,
		double &op_time,
		double &memcpy_time,
		double &persist_time)
{
    auto start = TIME_NOW;
	int threads_per_blk = min(512, (num_elems + INSERT_BLOCK - 1) / INSERT_BLOCK * ELEM_NUM);
	//printf("hash_insert: num_blks %d, threads_per_blk %d\n", num_blks, threads_per_blk);

	// prefer L1 cache rather than shared cache
	//void (*funcPtr)(bucket_t *, ielem_t **, loc_t **, int *);
	//funcPtr = hash_insert;
	//cudaFuncSetCacheConfig(*funcPtr, cudaFuncCachePreferL1);
	assert(ELEM_NUM_P < 5 && ELEM_NUM_P > 0);

#if defined(HASH_2CHOICE)
    printf("Using insert2choice\n");
	if (stream == 0) {
		hash_insert_2choice<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
	} else {
		hash_insert_2choice<<<num_blks, threads_per_blk, 0, stream>>>(
			hash_table, blk_input, blk_elem_num);
	}
#elif defined(HASH_CUCKOO)
    printf("Using cuckoo\n");
	if (stream == 0) {
		hash_insert_cuckoo<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
	} else {
		hash_insert_cuckoo<<<num_blks, threads_per_blk, 0, stream>>>(
			hash_table, blk_input, blk_elem_num);
	}
#endif
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    op_time += (TIME_NOW - start).count();
    persist_data(imkv_file, hash_table, memcpy_time, persist_time);
	return;
}

void gpu_hash_delete(
		delem_t 	*in,
		bucket_t	*hash_table,
		int			num_elem,
		int 		num_thread,
		int			threads_per_blk,
		void        *imkv_file,
		cudaStream_t stream,
		double &op_time,
		double &memcpy_time,
		double &persist_time)
{
    auto start = TIME_NOW;
	int num_blks = (num_thread + threads_per_blk - 1) / threads_per_blk;
	assert(num_thread >= threads_per_blk);
	assert(threads_per_blk <= 1024);
	//assert(num_thread <= num_elem);
	if (num_thread % 32 != 0) {
		num_thread = (num_thread + 31) & 0xffe0;
	}
	assert(num_thread % 32 == 0);

	/* prefer L1 cache rather than shared memory,
	   the other is cudaFuncCachePreferShared
	*/
	//void (*funcPtr)(selem_t *, loc_t *, bucket_t *, int, int);
	//funcPtr = hash_search;
	//cudaFuncSetCacheConfig(*funcPtr, cudaFuncCachePreferL1);

	//printf("stream=%d, threads_per_blk =%d, num_blks = %d\n", stream, threads_per_blk, num_blks);
	if (stream == 0) {
		hash_delete<<<num_blks, threads_per_blk>>>(
			in, hash_table, num_elem, num_thread);
	} else  {
		hash_delete<<<num_blks, threads_per_blk, 0, stream>>>(
			in, hash_table, num_elem, num_thread);
	}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    op_time += (TIME_NOW - start).count();
    persist_data(imkv_file, hash_table, memcpy_time, persist_time);

	return;
}
