#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>


using namespace std;


#define CSC(call) do { cudaError_t res = call; if (res != cudaSuccess) { fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); exit(0); } } while (0)


__global__ void kernelSwap(int n, double *cols, double *vec, int step, int maxPos) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	double tmp;
	
	for (int col = idx; col <= n; col += offsetx) {
		if (col < step) {
			continue;
		}
		if (col == n) {
			tmp = vec[step];
			vec[step] = vec[maxPos];
			vec[maxPos] = tmp;
		}
		else {
			tmp = cols[col * n + step];
			cols[col * n + step] = cols[col * n + maxPos];
			cols[col * n + maxPos] = tmp;
		}
	}
}


__global__ void kernelModify(int n, double *cols, double *vec, int step) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	double coef;

	for (int row = idx; row < n; row += offsetx) {
		for (int col = idy; col <= n; col += offsety) {
			if (row <= step || col <= step) {
				continue;
			}

			coef = cols[step * n + row] / cols[step * n + step];

			if (col == n) {
				vec[row] = vec[row] - coef * vec[step];
			}
			else {
				cols[col * n + row] = cols[col * n + row] - coef * cols[col * n + step];
			}
		}
	}
}


struct compareKeyValue {
	__host__ __device__
	bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};


int main() {
	ios_base::sync_with_stdio(false);

	int n;
	cin >> n;

	double *cols = new double[n * n];
	double *vec = new double[n];

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			cin >> cols[col * n + row];
		}
	}

	for (int row = 0; row < n; row++) {
		cin >> vec[row];
	}

	double *devCols, *devVec;
	CSC(cudaMalloc(&devCols, sizeof(double) * n * n));
	CSC(cudaMalloc(&devVec, sizeof(double) * n));

	CSC(cudaMemcpy(devCols, cols, sizeof(double) * n * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(devVec, vec, sizeof(double) * n, cudaMemcpyHostToDevice));

	int maxPos;
	for (int step = 0; step < n - 1; step++) {
		thrust::device_ptr<double> devPtr = thrust::device_pointer_cast(devCols + step * (n + 1));
		thrust::device_ptr<double> maxPtr = thrust::max_element(devPtr, devPtr + (n - step), compareKeyValue());
		maxPos = &maxPtr[0] - &devPtr[0] + step;
		kernelSwap<<<256, 256>>>(n, devCols, devVec, step, maxPos);
		CSC(cudaGetLastError());
		kernelModify<<<dim3(16, 16), dim3(16, 16)>>>(n, devCols, devVec, step);
		CSC(cudaGetLastError());
	}

	CSC(cudaMemcpy(cols, devCols, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	CSC(cudaMemcpy(vec, devVec, sizeof(double) * n, cudaMemcpyDeviceToHost));

	CSC(cudaFree(devCols));
	CSC(cudaFree(devVec));

	vector<double> res(n);
	double tmp;
	for (int row = n - 1; row >= 0; row--) {
		tmp = 0;
		for (int col = row + 1; col < n; col++) {
			tmp += cols[col * n + row] * res[col];
		}
		res[row] = (vec[row] - tmp) / cols[row * n + row];
	}

	cout.precision(10);
	cout.setf(ios::scientific);
	for (int i = 0; i < n; i++) {
		cout << res[i] << " ";
	}
	cout << endl;

	delete[] cols;
	delete[] vec;

	return 0;
}
