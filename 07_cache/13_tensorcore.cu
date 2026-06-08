#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <chrono>
using namespace std;
using namespace nvcuda;
/*
Computes the matrix product C = A * B using the GPU's tensor cores (16x16x16) via WMMA API. The output is tiled: each thread block computes one 256x128 tile of C. To do so it walks the shared K dimension in chunks of 32, loading the matching slices of A and B into shared memory so all threads in the block can reuse them. Those slices are double-buffered -- while the tensor cores multiply the current chunk, the next one is already being loaded, which reduces global memory latency. Inputs are converted FP32 -> FP16 on the way into shared memory and the products accumulate in FP32. Within a block, each of the 8 warps computes a 64x64 sub-tile as a 4x4 grid of 16x16 WMMA fragments.
*/
__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       float *d_a, float *d_b, float *d_c) {
  const int BLOCK_M = 256, BLOCK_N = 128, BLOCK_K = 32;
  const int A_STRIDE = BLOCK_M + 8;   // shared A is [k][m], +8 padding for bank conflicts
  const int B_STRIDE = BLOCK_K + 8;   // shared B is [n][k], +8 padding
  const int THREADS = 256;
  const int A_LOADS = BLOCK_M * BLOCK_K / 4 / THREADS;   // float4 loads of A per thread = 8
  const int B_LOADS = BLOCK_N * BLOCK_K / 4 / THREADS;   // float4 loads of B per thread = 4

  int offset_m = BLOCK_M * blockIdx.x;
  int offset_n = BLOCK_N * blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int warp_m = (warp_id / 2) * 64;   // this warp's row in the C tile
  int warp_n = (warp_id % 2) * 64;   // this warp's col in the C tile

  extern __shared__ half smem[];
  half (*block_a)[BLOCK_K][A_STRIDE] = (half (*)[BLOCK_K][A_STRIDE]) smem;
  half (*block_b)[BLOCK_N][B_STRIDE] = (half (*)[BLOCK_N][B_STRIDE]) (smem + 2 * BLOCK_K * A_STRIDE);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  float4 ra[A_LOADS], rb[B_LOADS];

  // load the first K chunk and convert it to FP16 in shared memory
  for (int i = 0; i < A_LOADS; i++) {
    int idx = tid + i * THREADS;
    int k = idx / (BLOCK_M / 4), m = idx % (BLOCK_M / 4);
    ra[i] = *(float4 *)&d_a[k * dim_m + offset_m + m * 4];
  }
  for (int i = 0; i < B_LOADS; i++) {
    int idx = tid + i * THREADS;
    int n = idx / (BLOCK_K / 4), k = idx % (BLOCK_K / 4);
    rb[i] = *(float4 *)&d_b[(offset_n + n) * dim_k + k * 4];
  }
  for (int i = 0; i < A_LOADS; i++) {
    int idx = tid + i * THREADS;
    int k = idx / (BLOCK_M / 4), m = idx % (BLOCK_M / 4);
    *(half2 *)&block_a[0][k][m * 4 + 0] = __floats2half2_rn(ra[i].x, ra[i].y);
    *(half2 *)&block_a[0][k][m * 4 + 2] = __floats2half2_rn(ra[i].z, ra[i].w);
  }
  for (int i = 0; i < B_LOADS; i++) {
    int idx = tid + i * THREADS;
    int n = idx / (BLOCK_K / 4), k = idx % (BLOCK_K / 4);
    *(half2 *)&block_b[0][n][k * 4 + 0] = __floats2half2_rn(rb[i].x, rb[i].y);
    *(half2 *)&block_b[0][n][k * 4 + 2] = __floats2half2_rn(rb[i].z, rb[i].w);
  }
  __syncthreads();

  int buf = 0;
  int numK = dim_k / BLOCK_K;
  for (int t = 0; t < numK; t++) {
    // prefetch the next K chunk into registers
    int kbase = (t + 1) * BLOCK_K;
    if (t + 1 < numK) {
      for (int i = 0; i < A_LOADS; i++) {
        int idx = tid + i * THREADS;
        int k = idx / (BLOCK_M / 4), m = idx % (BLOCK_M / 4);
        ra[i] = *(float4 *)&d_a[(kbase + k) * dim_m + offset_m + m * 4];
      }
      for (int i = 0; i < B_LOADS; i++) {
        int idx = tid + i * THREADS;
        int n = idx / (BLOCK_K / 4), k = idx % (BLOCK_K / 4);
        rb[i] = *(float4 *)&d_b[(offset_n + n) * dim_k + kbase + k * 4];
      }
    }

    // multiply the current tile (two k=16 steps, 4x4 fragments per warp)
    for (int ks = 0; ks < BLOCK_K / 16; ks++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];
      for (int r = 0; r < 4; r++)
        wmma::load_matrix_sync(a_frag[r], &block_a[buf][ks * 16][warp_m + r * 16], A_STRIDE);
      for (int c = 0; c < 4; c++)
        wmma::load_matrix_sync(b_frag[c], &block_b[buf][warp_n + c * 16][ks * 16], B_STRIDE);
      for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }

    // store the prefetched chunk into the other shared buffer
    if (t + 1 < numK) {
      buf = 1 - buf;
      for (int i = 0; i < A_LOADS; i++) {
        int idx = tid + i * THREADS;
        int k = idx / (BLOCK_M / 4), m = idx % (BLOCK_M / 4);
        *(half2 *)&block_a[buf][k][m * 4 + 0] = __floats2half2_rn(ra[i].x, ra[i].y);
        *(half2 *)&block_a[buf][k][m * 4 + 2] = __floats2half2_rn(ra[i].z, ra[i].w);
      }
      for (int i = 0; i < B_LOADS; i++) {
        int idx = tid + i * THREADS;
        int n = idx / (BLOCK_K / 4), k = idx % (BLOCK_K / 4);
        *(half2 *)&block_b[buf][n][k * 4 + 0] = __floats2half2_rn(rb[i].x, rb[i].y);
        *(half2 *)&block_b[buf][n][k * 4 + 2] = __floats2half2_rn(rb[i].z, rb[i].w);
      }
    }
    __syncthreads();
  }

  // write this warp's 64x64 result back to C 
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_m + warp_m + r * 16;
      int c_n = offset_n + warp_n + c * 16;
      if (c_m < dim_m && c_n < dim_n)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int BLOCK_M = 256, BLOCK_N = 128, BLOCK_K = 32;
  int smem = (2 * BLOCK_K * (BLOCK_M + 8) + 2 * BLOCK_N * (BLOCK_K + 8)) * sizeof(half);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 block = dim3(256);
  dim3 grid = dim3(m / BLOCK_M, n / BLOCK_N);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block, smem >>>(m,
				    n,
				    k,
				    A,
				    B,
				    C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
