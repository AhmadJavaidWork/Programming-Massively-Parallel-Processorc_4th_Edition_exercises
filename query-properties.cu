#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>

void getDeviceProperties(cudaDeviceProp *devProp, int devNum) {
    cudaError_t err = cudaGetDeviceProperties_v2(devProp, devNum);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void getDevCount(int *devCount) {
    cudaError_t err = cudaGetDeviceCount(devCount);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int devCount;
    getDevCount(&devCount);

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp devProp;
        getDeviceProperties(&devProp, i);

        printf("Name: %s\n", devProp.name);
        printf("UUID: ");
        for (int j = 0; j < 16; ++j) {
            printf("%02x", devProp.uuid.bytes[j]);
        }
        printf("\n");

        size_t globalMemB = devProp.totalGlobalMem;
        double globalMemKB = (double)globalMemB / 1024.0;
        double globalMemMB = (double)globalMemKB / 1024.0;
        double globalMemGB = (double)globalMemMB / 1024.0;

        printf("Total Global Memory: %zu B or %.2f KB or %.2f MB or %.2f GB\n", globalMemB,
               globalMemKB, globalMemMB, globalMemGB);

        size_t sharedMemoryPerBlockB = devProp.sharedMemPerBlock;
        double sharedMemoryPerBlockKB = (double)sharedMemoryPerBlockB / 1024.0;
        printf("Shared Memory Per Block: %zu B or %.2f KB\n", sharedMemoryPerBlockB,
               sharedMemoryPerBlockKB);

        printf("Registers per block: %d\n", devProp.regsPerBlock);
        printf("Warp size: %d threads\n", devProp.warpSize);

        printf("Max Threads Per Block: %d\n", devProp.maxThreadsPerBlock);

        printf("Max Threads Per Dim:\n");
        printf("  - x: %d\n", devProp.maxThreadsDim[0]);
        printf("  - y: %d\n", devProp.maxThreadsDim[1]);
        printf("  - z: %d\n", devProp.maxThreadsDim[2]);

        printf("Max Grid Size:\n");
        printf("  - x: %d\n", devProp.maxGridSize[0]);
        printf("  - y: %d\n", devProp.maxGridSize[1]);
        printf("  - z: %d\n", devProp.maxGridSize[2]);

        printf("Clock Rate: %.2fMHz\n", devProp.clockRate / 1000.0);

        printf("Multi Processor Count: %d\n", devProp.multiProcessorCount);

        printf("Is Integrated: %s\n", devProp.integrated == 1 ? "Yes" : "No");

        printf("Is ECC Enabled: %s\n", devProp.ECCEnabled == 1 ? "Yes" : "No");

        printf("PCI Bus Id: %d\n", devProp.pciBusID);

        printf("PCI Device Id: %d\n", devProp.pciDeviceID);

        printf("Memory Clock Rate: %.2fMHz\n", devProp.memoryClockRate / 1000.0);

        printf("Memory Bus Width: %d\n", devProp.memoryBusWidth);

        printf("L2 Cache size: %.2fKB\n", devProp.l2CacheSize / 1024.0);

        printf("Max Threads Per Multi Processor: %d\n", devProp.maxThreadsPerMultiProcessor);

        printf("Global L1 Cache Supported: %s\n",
               devProp.globalL1CacheSupported == 1 ? "Yes" : "No");

        printf("Local L1 Cache Supported: %s\n", devProp.localL1CacheSupported == 1 ? "Yes" : "No");

        printf("Shared Memory Per Multi Processor: %0.2fKB\n",
               devProp.sharedMemPerMultiprocessor / 1024.0);

        printf("Registers Per Multi Processor: %d\n", devProp.regsPerMultiprocessor);

        printf("Is Mult GPU Board: %s\n", devProp.isMultiGpuBoard == 1 ? "Yes" : "No");
    }

    return 0;
}
