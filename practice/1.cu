#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void MatAddKernel(int *A, int *B, int *C, int n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

void vecAdd(int *A_h, int *B_h, int *C_h, int n) {
    int *A_d, *B_d, *C_d;

    cudaError_t err = cudaMalloc((void **)&A_d, n * n * sizeof(int));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&B_d, n * n * sizeof(int));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&C_d, n * n * sizeof(int));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(A_d, A_h, n * n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(B_d, B_h, n * n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    dim3 grid((n + 16 - 1) / 16, (n + 16 - 1) / 16);
    dim3 block(16, 16, 1);

    MatAddKernel<<<grid, block>>>(A_d, B_d, C_d, n);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(C_h, C_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaFree(A_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *A, *B, *C;
    int n = 5;

    A = (int *)malloc(n * n * sizeof(int));
    B = (int *)malloc(n * n * sizeof(int));
    C = (int *)malloc(n * n * sizeof(int));

    if (!A || !B || !C) {
        printf("Cannot assign memory\n");
        return 1;
    }

    for (int i = 0; i < n * n; ++i) {
        A[i] = i;
        B[i] = i;
    }

    vecAdd(A, B, C, n);

    for (int i = 0; i < n * n; ++i) {
        printf("%d ", C[i]);
    }

    printf("\nDone\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
