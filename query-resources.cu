#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
    int devCount;

    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    printf("Available cuda devices: %d\n", devCount);
    return 0;
}
