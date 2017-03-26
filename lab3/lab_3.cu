#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <cstring>


using namespace std;


typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


#define CSC(call) do { cudaError_t res = call; if (res != cudaSuccess) { fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); exit(0); } } while (0)


__constant__ double GPU_AVG[32][3];
__constant__ double GPU_INVERT_COV[32][3][3];


struct Point {
	int x, y;
	Point(int x, int y) : x(x), y(y) {}
	Point() : x(0), y(0) {}
};


class BinaryImage {
public:
	uint w;
	uint h;
	uchar4 *data;

	BinaryImage() : w(0), h(0), data(NULL) {}
	BinaryImage(string path) {
		FILE *fin = fopen(path.c_str(), "rb");
		if (!fin) {
			printf("File %s not found\n", path.c_str());
			return;
		}
		fread(&w, sizeof(uint), 1, fin);
		fread(&h, sizeof(uint), 1, fin);
		data = new uchar4[w * h];
		fread(data, sizeof(uchar4), w * h, fin);
		fclose(fin);
	}
	~BinaryImage() {
		if (data != NULL) {
			delete[] data;
		}
	}

	void toFile(string path) {
		FILE *fout = fopen(path.c_str(), "wb");
		if (!fout) {
			printf("File %s not found\n", path.c_str());
			return;
		}
		fwrite(&w, sizeof(uint), 1, fout);
		fwrite(&h, sizeof(uint), 1, fout);
		fwrite(data, sizeof(uchar4), w * h, fout);
		fclose(fout);
	}

	size_t size() {
		return w * h;
	}

	uchar4& getPixel(int x, int y) {
		return data[w * y + x];
	}

	uchar4& getPixel(const Point &p) {
		return getPixel(p.x, p.y);
	}
};


double minorOf3x3(double mtrx[3][3], int i, int j) {
	double arr[4];
	int len = 0;
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			if (row == i || col == j) {
				continue;
			}
			arr[len++] = mtrx[row][col];
		}
	}
	return arr[0] * arr[3] - arr[1] * arr[2];
}


double cofactorOf3x3(double mtrx[3][3], int i, int j) {
	return ((i + j) % 2 == 0 ? 1 : -1) * minorOf3x3(mtrx, i, j);
}


double determinantOf3x3(double mtrx[3][3]) {
	double res = 0;
	for (int col = 0; col < 3; col++) {
		res += mtrx[0][col] * cofactorOf3x3(mtrx, 0, col);
	}
	return res;
}


__device__ double mahalanobisDistance(uchar4 px, int cls) {
	double v[3] = { px.x - GPU_AVG[cls][0], px.y - GPU_AVG[cls][1], px.z - GPU_AVG[cls][2] };
	double tmp[3] = { 0, 0, 0 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			tmp[i] += v[j] * GPU_INVERT_COV[cls][j][i];
		}
	}
	double res = 0;
	for (int i = 0; i < 3; i++) {
		res += -tmp[i] * v[i];
	}
	return res;
}


__device__ int classify(uchar4 px, int clsNum) {
	double maxVal = mahalanobisDistance(px, 0);
	int resCls = 0;
	double tmp;
	for (int cls = 1; cls < clsNum; cls++) {
		tmp = mahalanobisDistance(px, cls);
		if (tmp > maxVal) {
			maxVal = tmp;
			resCls = cls;
		}
	}
	return resCls;
}


__global__ void kernel(uchar4 *dst, uint w, uint h, int clsNum) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int x = idx; x < w; x += offsetx) {
		for (int y = idy; y < h; y += offsety) {
			dst[y * w + x].w = classify(dst[y * w + x], clsNum);
		}
	}
}


int main() {
	ios_base::sync_with_stdio(false);

	string in;
	string out;
	int nc;
	int np;
	int x, y;
	cin >> in >> out >> nc;

	BinaryImage img(in);

	vector<vector<Point> > classes(nc);
	uchar4 px;

	vector<vector<int> > pixelSum(nc, vector<int>(3, 0));
	for (int i = 0; i < nc; i++) {
		cin >> np;
		classes[i] = vector<Point>(np);
		for (int j = 0; j < np; j++) {
			cin >> x >> y;
			classes[i][j] = Point(x, y);
			px = img.getPixel(x, y);
			pixelSum[i][0] += px.x;
			pixelSum[i][1] += px.y;
			pixelSum[i][2] += px.z;
		}
	}

	double arrAvg[32][3];
	for (int i = 0; i < nc; i++) {
		for (int j = 0; j < 3; j++) {
			arrAvg[i][j] = (double)pixelSum[i][j] / classes[i].size();
		}
	}
	CSC(cudaMemcpyToSymbol(GPU_AVG, arrAvg, sizeof(double) * 32 * 3));

	double arrCov[32][3][3];
	memset(arrCov, 0, sizeof(double) * 32 * 3 * 3);
	for (int cls = 0; cls < nc; cls++) {
		for (int i = 0; i < (int)classes[cls].size(); i++) {
			px = img.getPixel(classes[cls][i]);
			double coeff[3] = { px.x - arrAvg[cls][0], px.y - arrAvg[cls][1], px.z - arrAvg[cls][2] };
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					arrCov[cls][row][col] += coeff[row] * coeff[col];
				}
			}
		}
		np = classes[cls].size();
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				arrCov[cls][row][col] /= np - 1;
			}
		}
	}

	double arrInvertCov[32][3][3];
	memset(arrInvertCov, 0, sizeof(double) * 32 * 3 * 3);
	double det;
	for (int cls = 0; cls < nc; cls++) {
		det = determinantOf3x3(arrCov[cls]);
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				if (det == 0) {
					arrInvertCov[cls][col][row] = (row == col ? 1 : 0);
				}
				else {
					arrInvertCov[cls][col][row] = cofactorOf3x3(arrCov[cls], row, col) / det;
				}
			}
		}
	}
	CSC(cudaMemcpyToSymbol(GPU_INVERT_COV, arrInvertCov, sizeof(double) * 32 * 3 * 3));

	uchar4 *devData;
	CSC(cudaMalloc(&devData, sizeof(uchar4) * img.size()));
	CSC(cudaMemcpy(devData, img.data, sizeof(uchar4) * img.size(), cudaMemcpyHostToDevice));

	kernel<<<dim3(16, 16), dim3(16, 16)>>>(devData, img.w, img.h, nc);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(img.data, devData, sizeof(uchar4) * img.size(), cudaMemcpyDeviceToHost));

	CSC(cudaFree(devData));

	img.toFile(out);
	return 0;
}
