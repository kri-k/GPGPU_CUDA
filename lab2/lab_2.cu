#include <cstdio>
#include <string>
#include <cassert>
#include <iostream>
#include <cstddef>


using namespace std;


typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)


texture<uchar4, 2, cudaReadModeElementType> tex;


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
};


__device__ uchar getMedian(ushort *cnt, int mid) {
	int curNum = 0;
	for (int i = 0; i < 256; i++) {
		curNum += cnt[i];
		if (curNum > mid) {
			return i;
		}
	}
	return 255;
}


__device__ uchar4 getPixelColor(int x, int y, int radius, uint w, uint h) {
	uchar4 p;
	ushort cntR[256];
	ushort cntG[256];
	ushort cntB[256];
	uchar r, g, b;

	for (int i = 0; i < 256; i++) {
		cntR[i] = cntG[i] = cntB[i] = 0;
	}

	int mid = 0;

	for (int i = x - radius; i <= x + radius; i++) {
		for (int j = y - radius; j <= y + radius; j++) {
			if (i < 0 || j < 0 || i >= w || j >= h) {
				continue;
			}
			p = tex2D(tex, i, j);
			cntR[p.x]++;
			cntG[p.y]++;
			cntB[p.z]++;
			mid++;
		}
	}

	mid /= 2;
	
	r = getMedian(cntR, mid);
	g = getMedian(cntG, mid);
	b = getMedian(cntB, mid);

	return make_uchar4(r, g, b, tex2D(tex, x, y).w);
}


__global__ void kernel(uchar4 *dst, uint w, uint h, int radius) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int x = idx; x < w; x += offsetx) {
		for (int y = idy; y < h; y += offsety) {
			dst[y * w + x] = getPixelColor(x, y, radius, w, h);
		}
	}
}


int main() {
	ios_base::sync_with_stdio(false);

	string in;
	string out;
	int radius;
	cin >> in >> out >> radius;

	BinaryImage img(in);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, img.w, img.h));
	CSC(cudaMemcpyToArray(arr, 0, 0, img.data, sizeof(uchar4) * img.size(), cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *devData;
	CSC(cudaMalloc(&devData, sizeof(uchar4) * img.size()));

	kernel<<<dim3(16, 16), dim3(16, 16)>>>(devData, img.w, img.h, radius);
	CSC(cudaGetLastError());
	CSC(cudaMemcpy(img.data, devData, sizeof(uchar4) * img.size(), cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(devData));

	img.toFile(out);
	return 0;
}
