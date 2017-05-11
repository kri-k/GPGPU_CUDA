#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <curand_kernel.h>


#define CSC(call) {cudaError err = call; if(err != cudaSuccess) {fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);}} while (0)

#define GLOBAL_INT(name) fscanf(f, "%s", buf); \
                         fscanf(f, "%d", &name); \
                         CSC(cudaMemcpyToSymbol(D_ ## name, &name, sizeof(int)))

#define GLOBAL_FLOAT(name) fscanf(f, "%s", buf); \
                           fscanf(f, "%f", &name); \
                           CSC(cudaMemcpyToSymbol(D_ ## name, &name, sizeof(float)))

#define UPDATE_GLOBAL(name, type) CSC(cudaMemcpyToSymbol(D_ ## name, &name, sizeof(type)))

#define SKIP_LINE fscanf(f, "%s", buf)


const int WIDTH = 800;
const int HEIGHT = 800;

long long TIME = 0;
float TIME_STD_DEVIATION = 300;
bool PLAY = false;

dim3 blocks(32, 32), threads(32, 32);
dim3 lin_blocks(32 * 32), lin_threads(32 * 32);

#define EPS_NULL 0.1

int P_NUM;

float GEN_X_MIN;
float GEN_X_MAX;
float GEN_Y_MIN;
float GEN_Y_MAX;

float X_MIN;
float X_MAX;
float Y_MIN;
float Y_MAX;

float MIN_HEAT_VAL;
float MAX_HEAT_VAL;

float RADIUS;

float WEIGHT_LOCAL;
float WEIGHT_GLOBAL;
float WEIGHT_INERTIA;

int BLOCK_NUM_X;
int BLOCK_NUM_Y;

float FORCE_FACTOR;
float FORCE_POW;

float TIME_STEP;

__constant__ int D_P_NUM;

__constant__ float D_GEN_X_MIN;
__constant__ float D_GEN_X_MAX;
__constant__ float D_GEN_Y_MIN;
__constant__ float D_GEN_Y_MAX;

__constant__ float D_X_MIN;
__constant__ float D_X_MAX;
__constant__ float D_Y_MIN;
__constant__ float D_Y_MAX;

__constant__ float D_MIN_HEAT_VAL;
__constant__ float D_MAX_HEAT_VAL;

__constant__ float D_RADIUS;

__constant__ float D_WEIGHT_LOCAL;
__constant__ float D_WEIGHT_GLOBAL;
__constant__ float D_WEIGHT_INERTIA;

__constant__ int D_BLOCK_NUM_X;
__constant__ int D_BLOCK_NUM_Y;

__constant__ float D_FORCE_FACTOR;
__constant__ float D_FORCE_POW;

__constant__ float D_TIME_STEP;

float X_CENTER = 0;
float Y_CENTER = 0;
float DELTA = 800;

__constant__ float D_X_CENTER;
__constant__ float D_Y_CENTER;
__constant__ float D_DELTA;

__constant__ float2 *points;
__constant__ int *pointsIndexes;
__constant__ float2 *velocity;
__constant__ float3 *localBest;
__constant__ float3 globalBest;
__constant__ int *blockPrefixSum;
__constant__ float *rawFuncData;


void initGlobalVars() {
    FILE *f = fopen("globals.conf", "r");
    if (!f) {
        printf("Cant't open file globals.conf\n");
        exit(1);
    }
    char buf[100];

    GLOBAL_INT(P_NUM);
    SKIP_LINE;
    GLOBAL_FLOAT(GEN_X_MIN);
    GLOBAL_FLOAT(GEN_X_MAX);
    GLOBAL_FLOAT(GEN_Y_MIN);
    GLOBAL_FLOAT(GEN_Y_MAX);
    SKIP_LINE;
    GLOBAL_FLOAT(X_MIN);
    GLOBAL_FLOAT(X_MAX);
    GLOBAL_FLOAT(Y_MIN);
    GLOBAL_FLOAT(Y_MAX);
    SKIP_LINE;
    GLOBAL_FLOAT(MIN_HEAT_VAL);
    GLOBAL_FLOAT(MAX_HEAT_VAL);
    SKIP_LINE;
    GLOBAL_FLOAT(RADIUS);
    SKIP_LINE;
    GLOBAL_FLOAT(WEIGHT_LOCAL);
    GLOBAL_FLOAT(WEIGHT_GLOBAL);
    GLOBAL_FLOAT(WEIGHT_INERTIA);
    SKIP_LINE;
    GLOBAL_INT(BLOCK_NUM_X);
    GLOBAL_INT(BLOCK_NUM_Y);
    SKIP_LINE;
    GLOBAL_FLOAT(FORCE_FACTOR);
    GLOBAL_FLOAT(FORCE_POW);
    SKIP_LINE;
    GLOBAL_FLOAT(TIME_STEP);

    fclose(f);

    CSC(cudaMemcpyToSymbol(D_X_CENTER, &X_CENTER, sizeof(float)));
    CSC(cudaMemcpyToSymbol(D_Y_CENTER, &Y_CENTER, sizeof(float)));
    CSC(cudaMemcpyToSymbol(D_DELTA, &DELTA, sizeof(float)));
}


void createInitGlobalArrays() {
    srand(time(0));
    float2 *p = new float2[P_NUM];
    float2 *v = new float2[P_NUM];
    int *indx = new int[P_NUM];

    for (int i = 0; i < P_NUM; i++) {
        p[i].x = ((float)rand() / (float)(RAND_MAX)) * (GEN_X_MAX - GEN_X_MIN) + GEN_X_MIN;
        p[i].y = ((float)rand() / (float)(RAND_MAX)) * (GEN_Y_MAX - GEN_Y_MIN) + GEN_Y_MIN;
        v[i] = make_float2(0, 0);
        indx[i] = i;
    }

    void *tmp;

    CSC(cudaMalloc(&tmp, sizeof(float2) * P_NUM));
    CSC(cudaMemcpy(tmp, p, sizeof(float2) * P_NUM, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(points, &tmp, sizeof(float2*)));
    delete[] p;

    CSC(cudaMalloc(&tmp, sizeof(float2) * P_NUM));
    CSC(cudaMemcpy(tmp, v, sizeof(float2) * P_NUM, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(velocity, &tmp, sizeof(float2*)));
    delete[] v;

    CSC(cudaMalloc(&tmp, sizeof(int) * P_NUM));
    CSC(cudaMemcpy(tmp, indx, sizeof(int) * P_NUM, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(pointsIndexes, &tmp, sizeof(int*)));
    delete[] indx;

    CSC(cudaMalloc(&tmp, sizeof(float3) * P_NUM));
    CSC(cudaMemcpyToSymbol(localBest, &tmp, sizeof(float3*)));

    int lastElem = P_NUM;
    CSC(cudaMalloc(&tmp, sizeof(int) * (BLOCK_NUM_X * BLOCK_NUM_Y + 1)));
    CSC(cudaMemcpy((int*)tmp + BLOCK_NUM_X * BLOCK_NUM_Y, &lastElem, sizeof(int), cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(blockPrefixSum, &tmp, sizeof(int*)));

    CSC(cudaMalloc(&tmp, sizeof(float) * WIDTH * HEIGHT));
    CSC(cudaMemcpyToSymbol(rawFuncData, &tmp, sizeof(float*)));
}


void deleteGlobalArrays() {
    void *ptr;
    cudaMemcpyFromSymbol(&ptr, points, sizeof(float2*));
    CSC(cudaFree(ptr));
    cudaMemcpyFromSymbol(&ptr, velocity, sizeof(float2*));
    CSC(cudaFree(ptr));
    cudaMemcpyFromSymbol(&ptr, localBest, sizeof(float3*));
    CSC(cudaFree(ptr));
    cudaMemcpyFromSymbol(&ptr, blockPrefixSum, sizeof(int*));
    CSC(cudaFree(ptr));
    cudaMemcpyFromSymbol(&ptr, rawFuncData, sizeof(float*));
    CSC(cudaFree(ptr));
}
