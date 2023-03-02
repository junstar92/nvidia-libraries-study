/*****************************************************************************
 * File:        smem_rectangle.cu
 * Description: This is an example to transpose rectangle thread coordinates of
 *              a CUDA grid using shared memory into a global memory array.
 *              Different kernels demonstrate performing reads and writes with
 *              different ordering and optimizaing using memory padding.
 *              
 *              Test Kernels:
 *              - setColReadCol         : set by columns, read by columns
 *              - setRowReadRow         : set by rows, read by rows
 *              - setRowReadCol         : set by rows, read by columns
 *              - setRowReadColDyn      : set by rows, read by columns,
 *                                        using dynamic allocation
 *              - setRowReadColPad      : set by rows, read by columns,
 *                                        using memory paddings
 *              - setRowReadColDynPad   : set by rows, read by columns,
 *                                        using dynamice allocation and paddings
 *              
 * Compile:     nvcc -O3 -o smem_rectangle smem_rectangle.cu
 * Run:         ./smem_rectangle
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define BDIMX 32
#define BDIMY 16
#define PADDING 1

void printData(char* msg, int* in, int const num_elements)
{
    printf("%s:\n", msg);
    for (int i = 0; i < num_elements; i++) {
        printf("%5d", in[i]);
        if ((i+1) % BDIMX == 0)
            printf("\n");
    }
    printf("\n");
}

__global__
void setRowReadRow(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__
void setColReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__
void setRowReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__
void setRowReadColDyn(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx];
}

__global__
void setRowReadColPad(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + PADDING];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__
void setRowReadColDynPad(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int g_idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;
    
    unsigned int row_idx = (blockDim.x + PADDING) * threadIdx.y + threadIdx.x;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * (blockDim.x + PADDING) + irow;

    // shared memory store operation
    tile[row_idx] = g_idx;
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    cudaSharedMemConfig smem_config = cudaSharedMemBankSizeEightByte;
    CUDA_ERROR_CHECK(cudaDeviceGetSharedMemConfig(&smem_config));
    printf("> At device %d: %s with Bank Size: %s\n", dev, prop.name,
        smem_config == cudaSharedMemBankSizeFourByte ? "4-Byte" : "8-Byte");

    bool print = false;
    if (argc > 1) print = static_cast<bool>(atoi(argv[1]));

    // setup array size
    int nx = BDIMX;
    int ny = BDIMY;
    size_t bytes = nx * ny * sizeof(int);

    // execution congfiguration
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    // allocate host and device memory
    int* d_out;
    CUDA_ERROR_CHECK(cudaMalloc(&d_out, bytes));
    int* gpu_ref = (int*)malloc(bytes);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setColReadCol<<<grid, block>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set col read col   ", gpu_ref, nx * ny);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setRowReadRow<<<grid, block>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set row read row   ", gpu_ref, nx * ny);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setRowReadCol<<<grid, block>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set row read col   ", gpu_ref, nx * ny);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setRowReadColDyn<<<grid, block, bytes>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set row read col dyn", gpu_ref, nx * ny);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setRowReadColPad<<<grid, block>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set row read col pad", gpu_ref, nx * ny);

    CUDA_ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + PADDING) * BDIMY * sizeof(int)>>>(d_out);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));
    if (print) printData("set row read col DP ", gpu_ref, nx * ny);

    // free host and device memory
    CUDA_ERROR_CHECK(cudaFree(d_out));
    free(gpu_ref);

    return 0;
}