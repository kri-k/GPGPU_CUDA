#include <cstdio>
#include <iostream>
#include <limits>
#include <string>


using namespace std;


#define CSC(call) do { cudaError_t res = call; if (res != cudaSuccess) { fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); exit(0); } } while (0)


typedef long long ll;


const int BLOCK_SIZE = 512; // must be power of 2
const int GRID_SIZE = 32768;

const ll INF = numeric_limits<ll>::max();


__device__ void swap(ll *a, ll *b) {
    ll tmp = *a;
    *a = *b;
    *b = tmp;
}


__global__ void oddEvenBlockSort(ll *arr, int len) {
    int arrOffset = blockIdx.x * BLOCK_SIZE;
    if (arrOffset >= len) {
        return;
    }

    __shared__ ll block[BLOCK_SIZE];
    int idx = threadIdx.x;
    int sortIndx = 2 * idx;

    for (int i = 0; i < 2; i++) {
        block[sortIndx + i] = arr[arrOffset + sortIndx + i];
    }

    for (int k = 0; k < BLOCK_SIZE / 2; k++) {
        __syncthreads();
        if (sortIndx + 1 < BLOCK_SIZE) {
            if (block[sortIndx] > block[sortIndx + 1]) {
                swap(block + sortIndx, block + sortIndx + 1);
            }
        }

        __syncthreads();
        if (sortIndx + 2 < BLOCK_SIZE) {
            if (block[sortIndx + 1] > block[sortIndx + 2]) {
                swap(block + sortIndx + 1, block + sortIndx + 2);
            }
        }
    }

    __syncthreads();
    for (int i = 0; i < 2; i++) {
        arr[arrOffset + sortIndx + i] = block[sortIndx + i];
    }
}


__global__ void bitonicMerge(ll *arr, int len, bool oddPhase) {
    int arrOffset = blockIdx.x * BLOCK_SIZE * 2;

    if (oddPhase) {
        arrOffset += BLOCK_SIZE;
    }

    if (arrOffset + BLOCK_SIZE * 2 > len) {
        return;
    }

    __shared__ ll block[BLOCK_SIZE * 2];
    int idx = threadIdx.x;
    int sortIndx = 2 * idx;

    for (int i = 0; i < 2; i++) {
        block[sortIndx + i] = arr[arrOffset + sortIndx + i];
    }

    __syncthreads();
    if (idx < BLOCK_SIZE && block[idx] > block[BLOCK_SIZE * 2 - idx - 1]) {
        swap(block + idx, block + BLOCK_SIZE * 2 - idx - 1);
    }

    int tmpIdx;
    int step = BLOCK_SIZE / 2;
    while (step != 0) {
        __syncthreads();
        if ((idx / step) % 2 == 0) {
            tmpIdx = idx;
        }
        else {
            tmpIdx = idx - step + BLOCK_SIZE;
        }
        if (block[tmpIdx] > block[tmpIdx + step]) {
            swap(block + tmpIdx, block + tmpIdx + step);
        }
        step /= 2;
    }

    __syncthreads();
    for (int i = 0; i < 2; i++) {
        arr[arrOffset + sortIndx + i] = block[sortIndx + i];
    }
}


int main() {
    ios_base::sync_with_stdio(false);

    int n;
    fread(&n, sizeof(int), 1, stdin);

    int len = n;
    if (n % BLOCK_SIZE != 0) {
        len += BLOCK_SIZE - n % BLOCK_SIZE;
    }

    ll *arr = new ll[len];
    int elem;
    for (int i = 0; i < n; i++) {
        fread(&elem, sizeof(int), 1, stdin);
        arr[i] = elem;
    }

    for (int i = n; i < len; i++) {
        arr[i] = INF;
    }

    ll *devArr;
    CSC(cudaMalloc(&devArr, sizeof(ll) * len));
    CSC(cudaMemcpy(devArr, arr, sizeof(ll) * len, cudaMemcpyHostToDevice));

    oddEvenBlockSort<<<GRID_SIZE, BLOCK_SIZE / 2>>>(devArr, len);
    CSC(cudaGetLastError());

    if (len > BLOCK_SIZE) {
        for (int step = 0; step < len / BLOCK_SIZE; step++) {
            bitonicMerge<<<GRID_SIZE, BLOCK_SIZE>>>(devArr, len, step & 1);
            CSC(cudaGetLastError());
        }
    }

    CSC(cudaMemcpy(arr, devArr, sizeof(ll) * len, cudaMemcpyDeviceToHost));
    CSC(cudaFree(devArr));

    for (int i = 0; i < n; i++) {
        elem = (int)arr[i];
        fwrite(&elem, sizeof(int), 1, stdout);
    }

    delete[] arr;

    return 0;
}
