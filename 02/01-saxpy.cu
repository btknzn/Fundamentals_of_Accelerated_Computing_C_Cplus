#include <math.h>
#include <stdio.h>

#define N 2048*2048  // Number of elements in each vector
#define rowcol2idx(num, r, c) ((r)*(num)+(c))

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nvprof to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */
__global__ void saxpy(int *a, int *b, int *c, int base)
{
    /*
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (; tid < N; tid += stride)
      c[tid] = (a[tid]<<1) + b[tid];
    */
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = rowcol2idx(gridDim.x*blockDim.x, row, col)+base;
    if (tid < N)
     c[tid] = (a[tid]<<1) + b[tid];
}

/*
__global__ void init_array(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; tid < N; tid += stride) {
      a[tid] = 2;
      b[tid] = 1;
      c[tid] = 0;
    }
}
*/

__global__ void init_array(int *a, int target, int stride)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += stride)
      a[tid] = target;
}

int main()
{
    int *a, *b, *c;

    int size = N*sizeof(int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

   int deviceId;
   cudaDeviceProp props;
   cudaGetDevice(&deviceId);
   cudaGetDeviceProperties(&props, deviceId);
    int multiProcessorCount = props.multiProcessorCount;

   size_t threadsPerBlock = 1024;
   size_t numberOfBlocks = ((N>>10)/multiProcessorCount+1)*multiProcessorCount;

   // i first prefetch pages...
   cudaMemPrefetchAsync(a, size, deviceId);
   cudaMemPrefetchAsync(b, size, deviceId);
   init_array<<<numberOfBlocks, threadsPerBlock>>>(a,2,threadsPerBlock*numberOfBlocks);
   init_array<<<numberOfBlocks, threadsPerBlock>>>(b,1,threadsPerBlock*numberOfBlocks);
   // we have no need to initialize array c, because of default value is 0.
   //init_array<<<numberOfBlocks, threadsPerBlock>>>(c,0,threadsPerBlock*numberOfBlocks);
   cudaDeviceSynchronize();

   //printf("sm numer = %d\n", multiProcessorCount);
   cudaMemPrefetchAsync(c, size, deviceId);
   //saxpy <<<numberOfBlocks, threadsPerBlock>>>(a,b,c,threadsPerBlock*numberOfBlocks);
   int len = 40;
   dim3 threads_per_block(32, 32, 1);
   dim3 number_of_blocks(len, len, 1);
   saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c, 0);
   saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c, 1600*1024);
   saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c, 3200*1024);
   cudaDeviceSynchronize();
   
   cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
