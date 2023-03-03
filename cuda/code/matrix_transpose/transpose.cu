/*****************************************************************************
 * File:        transpose.cu
 * Description: Implement various memory access optimizations for matrix transpose.
 *              
 * Compile:     nvcc -O3 -o transpose transpose.cu
 * Run:         ./transpose <blockx> <blocky> <nx> <ny>
 *                  <blockx> : x dim of a thread block (default: 16)
 *                  <blocky> : y dim of a thread block (default: 16)
 *                  <nx> : the row number of a matrix (default: 2048)
 *                  <ny> : the column number of a matrix (default: 2048)
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

void initMatrix(float* in, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        in[i] = rand() / (float)RAND_MAX;
    }
}

void transposeHost(float* out, float const* in, int const nx, int const ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

void checkResult(float const* host_ref, float const* gpu_ref, int const num_elements)
{
    double eps = 1e-8;
    
    for (int i = 0; i < num_elements; i++) {
        if (host_ref[i] != gpu_ref[i]) {
            printf("different on %dth element, host: %f / gpu: %f\n", i, host_ref[i], gpu_ref[i]);
            break;
        }
    }
}

__global__
void warmup(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// 0 copy kernel: read by rows, write by rows
__global__
void copyRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// 1 copy kernel: read by columns, write by columns
__global__
void copyCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

// 2 transpose kernel: read by rows, write by columns
__global__
void transposeNaiveRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// 3 transpose kernel: read by columns, write by rows
__global__
void transposeNaiveCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// 4 transpose kernel: read by rows, write by columns + unroll 4 blocks
__global__
void transposeUnroll4Row(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in [ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// 5 transpose kernel: read by columns, write by rows + unroll 4 blocks
__global__
void transposeUnroll4Col(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[ti] = in [to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

// 6 transpose kernel: read by rows, write by columns + diagonal coordinate transform
__global__
void transposeDiagonalRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int block_y = blockIdx.y;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.y;

    unsigned int ix = blockDim.x * block_x + threadIdx.x;
    unsigned int iy = blockDim.y * block_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// 7 transpose kernel: read by columns, write by rows + diagonal coordinate transform
__global__
void transposeDiagonalCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int block_y = blockIdx.y;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.y;

    unsigned int ix = blockDim.x * block_x + threadIdx.x;
    unsigned int iy = blockDim.y * block_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> Matrix transpose at device %d: %s\n", dev, prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // setup matrix dimensions (2048 x 2048)
    int nx = 1 << 11;
    int ny = 1 << 11;

    // setup kernel and block size
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) blockx = atoi(argv[1]);
    if (argc > 2) blocky = atoi(argv[2]);
    if (argc > 3) nx = atoi(argv[3]);
    if (argc > 4) ny = atoi(argv[4]);

    printf("> with matrix %d x %d\n", nx, ny);
    size_t bytes = nx * ny * sizeof(float);

    // setup execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* host_ref = (float*)malloc(bytes);
    float* gpu_ref = (float*)malloc(bytes);

    // initialize host matrix
    initMatrix(h_A, nx * ny);

    // transpose at host side
    transposeHost(host_ref, h_A, nx, ny);

    // allocate device memory
    float *d_A, *d_B;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, bytes));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // CUDA event to estimate elapsed time
    cudaEvent_t start, end;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // warmup to avoid startup overhead
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    warmup<<<grid, block>>>(d_B, d_A, nx, ny);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("warmup         elapsed time: %f ms\n", msec);

    // setup kernel
    void(*kernel_func)(float*, float const*, int const, int const);
    char* kernel_name;

    for (int i = 0; i < 8; i++) {
        switch (i) {
        default:
        case 0:
            kernel_func = &copyRow;
            kernel_name = "CopyRow       ";
            break;

        case 1:
            kernel_func = &copyCol;
            kernel_name = "CopyCol       ";
            break;

        case 2:
            kernel_func = &transposeNaiveRow;
            kernel_name = "NaiveRow      ";
            break;

        case 3:
            kernel_func = &transposeNaiveCol;
            kernel_name = "NaiveCol      ";
            break;
        
        case 4:
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            kernel_func = &transposeUnroll4Row;
            kernel_name = "Unroll4Row    ";
            break;
        
        case 5:
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            kernel_func = &transposeUnroll4Col;
            kernel_name = "Unroll4Col    ";
            break;
        
        case 6:
            grid.x = (nx + block.x - 1) / block.x;
            kernel_func = &transposeDiagonalRow;
            kernel_name = "DiagonalRow   ";
            break;
        
        case 7:
            grid.x = (nx + block.x - 1) / block.x;
            kernel_func = &transposeDiagonalCol;
            kernel_name = "DiagonalCol   ";
            break;
        }

        // run kernel
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        kernel_func<<<grid, block>>>(d_B, d_A, nx, ny);
        CUDA_ERROR_CHECK(cudaEventRecord(end));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

        // calculate effective bandwidth
        float bandwidth = 2 * nx * ny * sizeof(float) / 1e9 / (msec * 1.0e-3);
        printf("%s elapsed time: %f ms <<< grid(%d,%d) block(%d,%d)>>> effective bandwidth: %f GB/s\n",
            kernel_name, msec, grid.x, grid.y, block.x, block.y, bandwidth);
        CUDA_ERROR_CHECK(cudaGetLastError());

        // check results
        if (i > 1) {
            CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_B, bytes, cudaMemcpyDeviceToHost));
            checkResult(host_ref, gpu_ref, nx * ny);
        }
    }

    // free host & device memory
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    free(h_A); free(host_ref); free(gpu_ref);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}