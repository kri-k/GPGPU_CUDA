#include <iostream>
#include <cstdio>
#include <iomanip>


#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)


using namespace std;


__global__ void kernel(double *da, double *db, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	while (idx < n) {
		da[idx] -= db[idx];
		idx += offset;
	}
}


__host__ double* scanDoubleArray(int size) {
	double *arr = new double[size];
	for (int i = 0; i < size; i++) {
		cin >> arr[i];
	}
	return arr;
}


__host__ double* getDeviceDoubleArray(double *arr, int size) {
	double *d_arr;
	CSC(cudaMalloc(&d_arr, sizeof(double) * size));
	CSC(cudaMemcpy(d_arr, arr, sizeof(double) * size, cudaMemcpyHostToDevice));
	return d_arr;
}


int main() {
	ios_base::sync_with_stdio(false);

	int n;
	cin >> n;

	double *a = scanDoubleArray(n);
	double *b = scanDoubleArray(n);

	double *da = getDeviceDoubleArray(a, n);
	double *db = getDeviceDoubleArray(b, n);

	delete[] b;

	kernel<<<256, 256>>>(da, db, n);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(a, da, sizeof(double) * n, cudaMemcpyDeviceToHost));

	CSC(cudaFree(da));
	CSC(cudaFree(db));

	cout.precision(10);
	cout.setf(ios::scientific);

	for (int i = 0; i < n; i++) {
		cout << a[i] << ' ';
	}
	cout << endl;

	delete[] a;

	return 0;
}
