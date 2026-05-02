#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecMatMulKernel(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; ++row) {
        float pValue = 0;
        for (int col = 0; col < n; ++col) {
            pValue += A[row * n + col] * B[col];
        }
        C[row] = pValue;
    }
}

void vecMatMul(float *A_h, float *B_h, float *C_h, int n) {
    float *A_d, *B_d, *C_d;

    cudaError_t err = cudaMalloc((void **)&A_d, n * n * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&B_d, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&C_d, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(A_d, A_h, n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    dim3 grid((n + 16 - 1) / 16.0);
    dim3 block(16.0);

    vecMatMulKernel<<<grid, block>>>(A_d, B_d, C_d, n);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(C_h, C_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

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
    float *A, *B, *C;
    int n = 2;

    A = (float *)malloc(n * n * sizeof(float));
    B = (float *)malloc(n * sizeof(float));
    C = (float *)malloc(n * sizeof(float));

    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;

    B[0] = 5.0;
    B[1] = 6.0;

    if (!A || !B) {
        printf("Cannot assign memory\n");
        return 1;
    }

    vecMatMul(A, B, C, n);

    for (int i = 0; i < n; i++) {
        printf("%f ", C[i]);
    }

    printf("Done\n");

    free(A);
    free(B);
    free(C);

    return 0;
}
