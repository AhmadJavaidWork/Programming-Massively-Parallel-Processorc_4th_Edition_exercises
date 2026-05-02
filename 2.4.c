#include <stdio.h>
#include <stdlib.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    const int size = 50000000;
    float *A_h = (float *)malloc(size * sizeof(float));
    float *B_h = (float *)malloc(size * sizeof(float));
    float *C_h = (float *)malloc(size * sizeof(float));

    if (!A_h || !B_h || !C_h) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < size; i++) {
        A_h[i] = i;
        B_h[i] = 49999999.0 - i;
    }

    vecAdd(A_h, B_h, C_h, size);

    printf("Done \n");

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
