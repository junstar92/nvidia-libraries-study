/*****************************************************************************
 * File:        reduce_shfl.cu
 * Description: Implement parallel reduction kernel by using warp shuffle
 *              instruction.
 *              A thread block size of a grid is 256.
 *                  - reduceGmem
 *                  - reduceSmem
 *                  - reduceGmemUnroll (unroll factor 4)
 *                  - reduceSmemUnroll (unroll factor 4)
 *                  - reduceShfl
 *                  - reduceShflUnroll (unroll factor 4)
 *              
 * Compile:     nvcc -O3 -o reduce_shfl reduce_shfl.cu
 * Run:         ./reduce_shfl
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define DIM 256
#define SMEMDIM (DIM / 32)

// recursive implementation of interleaved pair approach
int recursiveReduce(int* data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// unrolling warp + gmem
__global__
void reduceGmem(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    
    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

// unrolling 4 thread blocks + unrolling warp + gmem
__global__
void reduceGmemUnroll(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 4;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 3 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

// reduction using shared memory + unrolling warp
__global__
void reduceSmem(int* g_in, int* g_out, unsigned int const n)
{
    __shared__ int smem[DIM];
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;

    // set to smem by each threads
    smem[tid] = in[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = smem[0];
}

// shared memory + unrolling 4 thread blocks + unrolling warp
__global__
void reduceSmemUnroll(int* g_in, int* g_out, unsigned int const n)
{
    __shared__ int smem[DIM];
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    
    // unrolling 4 blocks
    int tmp_sum = 0;
    if (idx + blockDim.x * 3 <= n) {
        tmp_sum += g_in[idx];
        tmp_sum += g_in[idx + blockDim.x];
        tmp_sum += g_in[idx + blockDim.x * 2];
        tmp_sum += g_in[idx + blockDim.x * 3];
    }
    smem[tid] = tmp_sum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = smem[0];
}

// using warp shuffle instruction
__inline__ __device__
int warpReduce(int local_sum)
{
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 16, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 8, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 4, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1, warpSize);

    return local_sum;
}

__global__
void reduceShfl(int* g_in, int* g_out, unsigned int const n)
{
    // shared memory for each warp sum in a thread block
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    // block-wide warp reduce
    int local_sum = warpReduce(g_in[idx]);

    // save warp sum to shared memory
    if (lane_idx == 0)
        smem[warp_idx] = local_sum;
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize)
        local_sum = (threadIdx.x < SMEMDIM) ? smem[lane_idx] : 0;
    
    if (warp_idx == 0)
        local_sum = warpReduce(local_sum);
    
    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = local_sum;
}

__global__
void reduceShflUnroll(int* g_in, int* g_out, unsigned int const n)
{
    // shared memory for each warp sum in a thread block
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int tid = threadIdx.x;
    unsigned int idx = 4 * blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    // block-wide warp reduce
    int local_sum = warpReduce(g_in[idx]);
    local_sum += warpReduce(g_in[idx + blockDim.x]);
    local_sum += warpReduce(g_in[idx + blockDim.x * 2]);
    local_sum += warpReduce(g_in[idx + blockDim.x * 3]);

    // save warp sum to shared memory
    if (lane_idx == 0)
        smem[warp_idx] = local_sum;
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize)
        local_sum = (threadIdx.x < SMEMDIM) ? smem[lane_idx] : 0;
    
    if (warp_idx == 0)
        local_sum = warpReduce(local_sum);
    
    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = local_sum;
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp dev_prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&dev_prop, dev));
    printf("> Starting reduction at device %d: %s\n", dev, dev_prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // array size
    int num_elements = 1 << 24;
    printf("> Array size: %d\n", num_elements);

    // execution configuration
    int block_size = DIM;

    dim3 block(block_size);
    dim3 grid((num_elements + block.x - 1) / block.x);

    // allocate host memory
    size_t bytes = num_elements * sizeof(int);
    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // init the input array
    for (int i = 0; i < num_elements; i++)
        h_in[i] = (int)(rand() & 0xFF);
    (void*)memcpy(tmp, h_in, bytes);

    // cuda event
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // allocate device memory
    int *d_in, *d_out;
    CUDA_ERROR_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_out, grid.x * sizeof(int)));

    float msec = 0.f;
    int gpu_sum = 0;
    
    // cpu reduction
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    int cpu_sum = recursiveReduce(tmp, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("cpu reduce             : %.4f ms     cpu sum: %d\n", msec, cpu_sum);

    // kernel 1: reduceGmem
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceGmem<<<grid, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("reduceGmem             : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);

    // kernel 2: reduceSmem
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceSmem<<<grid, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("reduceSmem             : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);

    // kernel 3: reduceGmemUnroll
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceGmemUnroll<<<grid.x / 4, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_out[i];
    printf("reduceGmemUnroll       : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x / 4, block.x);

    // kernel 4: reduceSmemUnroll
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_out[i];
    printf("reduceSmemUnroll       : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x / 4, block.x);

    // kernel 5: reduceShfl
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceShfl<<<grid.x, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("reduceShfl             : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);

    // kernel 6: reduceShflUnroll
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceShflUnroll<<<grid.x / 4, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_out[i];
    printf("reduceShflUnroll       : %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x / 4, block.x);


    // free host memory
    free(h_in);
    free(h_out);
    free(tmp);

    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_in));
    CUDA_ERROR_CHECK(cudaFree(d_out));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    // reset device
    CUDA_ERROR_CHECK(cudaDeviceReset());

    return 0;
}