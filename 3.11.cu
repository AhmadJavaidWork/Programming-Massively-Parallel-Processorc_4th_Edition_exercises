#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void MatrixMulKernel(float *A, float *B, float *C, int m, int n, int p) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < p) {
        float pValue = 0;
        for (int i = 0; i < n; ++i) {
            pValue += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = pValue;
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int m, int n, int p) {
    float *A_d, *B_d, *C_d;

    cudaError_t err = cudaMalloc((void **)&A_d, m * n * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&B_d, n * p * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&C_d, m * p * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(B_d, B_h, n * p * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    dim3 dimGrid(ceil(p / 16.0), ceil(p / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, p);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(C_h, C_d, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    err = cudaFree(A_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int m = 2;
    int n = 3;
    int p = 2;
    float *A_h;
    float *B_h;
    float *C_h;

    A_h = (float *)malloc(m * n * sizeof(float));
    B_h = (float *)malloc(n * p * sizeof(float));
    C_h = (float *)malloc(m * p * sizeof(float));

    if (!A_h || !B_h || !C_h) {
        printf("Memory allocation failed\n");
        return 1;
    }

    A_h[0] = 1.0;
    A_h[1] = 2.0;
    A_h[2] = 3.0;
    A_h[3] = 4.0;
    A_h[4] = 5.0;
    A_h[5] = 6.0;

    B_h[0] = 7.0;
    B_h[1] = 8.0;
    B_h[2] = 9.0;
    B_h[3] = 10.0;
    B_h[4] = 11.0;
    B_h[5] = 12.0;

    vecAdd(A_h, B_h, C_h, m, n, p);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < p; ++col) {
            printf("%f ", C_h[row * m + col]);
        }
    }

    free(A_h);
    free(B_h);
    free(C_h);

    printf("Done\n");
    return 0;
}
