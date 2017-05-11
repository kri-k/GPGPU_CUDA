#include "globals.h"
#include "keyboard.h"

float getWeightDecr() {
    //return pow(2.718281828, -pow(TIME, 2) / pow(TIME_STD_DEVIATION, 2));
    return 1.0;
}

struct compareLocalBest {
    __device__
    bool operator()(float3 a, float3 b) {
        return a.z < b.z;
    }
};

__device__
int pointToBlockNum(float2 p) {
    int n = min((int)((p.y - D_Y_MIN) * D_BLOCK_NUM_Y) / (int)(D_Y_MAX - D_Y_MIN), D_BLOCK_NUM_X - 1) * D_BLOCK_NUM_X;
    n += min((int)((p.x - D_X_MIN) * D_BLOCK_NUM_X) / (int)(D_X_MAX - D_X_MIN), D_BLOCK_NUM_Y - 1);
    return n;
}

struct comparePoints {
    __device__
    bool operator()(float2 a, float2 b) {
        int na = pointToBlockNum(a);
        int nb = pointToBlockNum(b);
        return na < nb;
    }
};

__host__ __device__
float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float convertItoX(int i) {
    return 2 * D_DELTA / WIDTH * i + D_X_CENTER - D_DELTA;
}

__device__ float convertJtoY(int j) {
    return 2 * D_DELTA / HEIGHT * j + D_Y_CENTER - D_DELTA;
}

__device__ int convertXtoI(float x) {
    return (x - D_X_CENTER + D_DELTA) * WIDTH / D_DELTA / 2.0;
}

__device__ int convertYtoJ(float y) {
    return (y - D_Y_CENTER + D_DELTA) * HEIGHT / D_DELTA / 2.0;
}

__device__ float fun(float x, float y) {
    return -x * sin(sqrt(fabs(x))) - y * sin(sqrt(fabs(y)));
}

__device__ float fun(float2 p) {
    return fun(p.x, p.y);
}

__global__ void drawHeatMap(uchar4* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    float x, y, f;
    for (int i = idx; i < WIDTH; i += offsetx) {
        x = convertItoX(i);
        for (int j = idy; j < HEIGHT; j += offsety) {
            y = convertJtoY(j);
            f = fun(x, y);
            if (D_X_MIN <= x && x <= D_X_MAX && D_Y_MIN <= y && y <= D_Y_MAX) {
                float ratio = 2 * (f - D_MIN_HEAT_VAL) / (D_MAX_HEAT_VAL - D_MIN_HEAT_VAL);
                int b = max(0, (int)(255 * (1 - ratio)));
                int r = max(0, (int)(255 * (ratio - 1)));
                data[j * WIDTH + i] = make_uchar4(r, 255 - b - r, b, 255);
                rawFuncData[j * WIDTH + i] = f;
            }
            else {
                data[j * WIDTH + i] = make_uchar4(176, 224, 255, 230);
                rawFuncData[j * WIDTH + i] = 0.0;
            }
        }
    }
}

__global__ void drawPoints(uchar4* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    float x, y;
    int xs, xe;
    int ys, ye;
    float r = D_RADIUS;
    for (int p = idx; p < D_P_NUM; p += offsetx) {
        x = points[p].x;
        y = points[p].y;
        if (x <= D_X_CENTER - D_DELTA || x >= D_X_CENTER + D_DELTA ||
            y <= D_Y_CENTER - D_DELTA || y >= D_Y_CENTER + D_DELTA)
        {
            continue;
        }
        xs = max(0, convertXtoI(x - r));
        xe = min(WIDTH - 1, convertXtoI(x + r));
        ys = max(0, convertYtoJ(y - r));
        ye = min(HEIGHT - 1, convertYtoJ(y + r));
        for (int i = xs; i <= xe; i++) {
            for (int j = ys; j <= ye; j++) {
                data[j * WIDTH + i] = make_uchar4(0, 0, 0, 255);
            }
        }
    }
}

__global__ void updateLocalExtremums() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    float fVal;
    for (int p = idx; p < D_P_NUM; p += offsetx) {
        int realP = pointsIndexes[p];
        fVal = fun(points[p]);
        if (fVal < localBest[realP].z) {
            localBest[realP] = make_float3(points[p].x, points[p].y, fVal);
        }
    }
}

void updateGlobalExtremum() {
    float3 *hostLocalBestPtr;
    cudaMemcpyFromSymbol(&hostLocalBestPtr, localBest, sizeof(float3*));
    thrust::device_ptr<float3> devPtrLocals = thrust::device_pointer_cast(hostLocalBestPtr);
    float3 extrm = thrust::min_element(devPtrLocals, devPtrLocals + P_NUM, compareLocalBest())[0];

    float3 hostGlobalBest;
    cudaMemcpyFromSymbol(&hostGlobalBest, globalBest, sizeof(float3));
    if (extrm.z < hostGlobalBest.z) {
        CSC(cudaMemcpyToSymbol(globalBest, &extrm, sizeof(float3)));
        printf("Global min = %f in (%f, %f)\n", extrm.z, extrm.x, extrm.y);
    }
}

__device__ float2 forceFromBlock(int blockNum, int pn) {
    float2 res = make_float2(0.0, 0.0);
    if (blockNum < 0 || blockNum >= D_BLOCK_NUM_X * D_BLOCK_NUM_Y) {
        return res;
    }
    float2 tmp;
    float2 target = points[pn];
    float len;
    for (int p = blockPrefixSum[blockNum]; p < blockPrefixSum[blockNum + 1]; p++) {
        if (p == pn) {
            continue;
        }
        tmp = points[p];
        tmp.x = target.x - tmp.x;
        tmp.y = target.y - tmp.y;
        len = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2));
        if (len < EPS_NULL) {
            len += EPS_NULL;
        }
        res.x += (tmp.x * D_FORCE_FACTOR) / (pow(len, (float)D_FORCE_POW + 1) + 1e-7);
        res.y += (tmp.y * D_FORCE_FACTOR) / (pow(len, (float)D_FORCE_POW + 1) + 1e-7);
    }
    return res;
}

__global__ void processPoints(unsigned long long seed, float weightDecr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    float rLx, rGx, rLy, rGy;
    seed *= idx;
    curandState s;
    curand_init(seed, 0, 0, &s);
    int curBlock;
    float2 blockForce;

    for (int p = idx; p < D_P_NUM; p += offsetx) {
        int realP = pointsIndexes[p];

        rLx = curand_uniform(&s);
        rGx = curand_uniform(&s);
        rLy = curand_uniform(&s);
        rGy = curand_uniform(&s);
        curBlock = pointToBlockNum(points[p]);

        float dt = D_TIME_STEP;

        velocity[realP].x = D_WEIGHT_INERTIA * velocity[realP].x +
            (D_WEIGHT_LOCAL * weightDecr * rLx * (localBest[realP].x - points[p].x) +
            D_WEIGHT_GLOBAL * weightDecr * rGx * (globalBest.x - points[p].x)) * dt;
        velocity[realP].y = D_WEIGHT_INERTIA * velocity[realP].y +
            (D_WEIGHT_LOCAL * weightDecr * rLy * (localBest[realP].y - points[p].y) +
            D_WEIGHT_GLOBAL * weightDecr * rGy * (globalBest.y - points[p].y)) * dt;

        for (int deltaRow = -1; deltaRow < 2; deltaRow++) {
            for (int deltaCol = -1; deltaCol < 2; deltaCol++) {
                blockForce = forceFromBlock(curBlock + deltaRow * D_BLOCK_NUM_X + deltaCol, p);
                velocity[realP].x += blockForce.x * dt;
                velocity[realP].y += blockForce.y * dt;
            }
        }

        points[p].x += velocity[realP].x * dt;
        points[p].y += velocity[realP].y * dt;
        points[p].x = min(max(D_X_MIN, points[p].x), D_X_MAX);
        points[p].y = min(max(D_Y_MIN, points[p].y), D_Y_MAX);
    }
}

__global__ void buildBlocksHistogram(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int n;
    for (int p = idx; p < D_P_NUM; p += offsetx) {
        n = pointToBlockNum(points[p]);
        if (n < 0 || n >= D_BLOCK_NUM_X * D_BLOCK_NUM_Y) {
            printf("(%f, %f) -> %d - fail\n", points[p].x, points[p].y, n);
            return;
        }
        //assert(((n < 0) || (n >= BLOCK_NUM_X * BLOCK_NUM_Y)));
        atomicAdd(&blockPrefixSum[n], 1);
    }
}

void sortPointsToBlocks() {
    void *tmp;
    void *tmp2;

    cudaMemcpyFromSymbol(&tmp, blockPrefixSum, sizeof(int*));
    thrust::device_ptr<int> devBlockPSum = thrust::device_pointer_cast((int*)tmp);
    thrust::fill(devBlockPSum, devBlockPSum + BLOCK_NUM_X * BLOCK_NUM_Y, (int)0);

    buildBlocksHistogram<<<256, 256>>>();
    CSC(cudaGetLastError());

    thrust::exclusive_scan(devBlockPSum, devBlockPSum + BLOCK_NUM_X * BLOCK_NUM_Y, devBlockPSum);

    cudaMemcpyFromSymbol(&tmp, points, sizeof(float2*));
    thrust::device_ptr<float2> devPtrPoints = thrust::device_pointer_cast((float2*)tmp);

    cudaMemcpyFromSymbol(&tmp2, pointsIndexes, sizeof(int*));
    thrust::device_ptr<int> devPtrPointsIndexes = thrust::device_pointer_cast((int*)tmp2);

    thrust::sort_by_key(devPtrPoints, devPtrPoints + P_NUM, devPtrPointsIndexes, comparePoints());
}

void updateCenter() {
    void *tmp;
    cudaMemcpyFromSymbol(&tmp, points, sizeof(float2*));
    thrust::device_ptr<float2> devPtrPoints = thrust::device_pointer_cast((float2*)tmp);
    float2 newCenter = thrust::reduce(devPtrPoints, 
        devPtrPoints + P_NUM,
        make_float2(0.0, 0.0),
        thrust::plus<float2>()
    );
    X_CENTER = newCenter.x / P_NUM;
    Y_CENTER = newCenter.y / P_NUM;
    UPDATE_GLOBAL(X_CENTER, float);
    UPDATE_GLOBAL(Y_CENTER, float);
}

void updateHeatMapMaxMin() {
    void *tmp;
    cudaMemcpyFromSymbol(&tmp, rawFuncData, sizeof(float*));
    thrust::device_ptr<float> devPtrRawFunc = thrust::device_pointer_cast((float*)tmp);
    auto p = thrust::minmax_element(devPtrRawFunc, devPtrRawFunc + WIDTH * HEIGHT);
    MIN_HEAT_VAL = p.first[0];
    MAX_HEAT_VAL = p.second[0];
    UPDATE_GLOBAL(MIN_HEAT_VAL, float);
    UPDATE_GLOBAL(MAX_HEAT_VAL, float);
}

struct cudaGraphicsResource *res;

void update() {
    uchar4* dev_data;
    size_t size;
    CSC(cudaGraphicsMapResources(1, &res, 0));
    CSC(cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res));

    drawHeatMap<<<blocks, threads>>>(dev_data);
    CSC(cudaGetLastError());

    drawPoints<<<lin_blocks, lin_threads>>>(dev_data);
    CSC(cudaGetLastError());

    updateHeatMapMaxMin();

    if (PLAY) {
        updateLocalExtremums<<<lin_blocks, lin_threads>>>();
        CSC(cudaGetLastError());

        updateGlobalExtremum();

        sortPointsToBlocks();

        processPoints<<<lin_blocks, lin_threads>>>(rand(), max(getWeightDecr(), 0.001));
        CSC(cudaGetLastError());

        updateCenter();

        TIME++;
    }

    CSC(cudaDeviceSynchronize());
    CSC(cudaGraphicsUnmapResources(1, &res, 0));

    glutPostRedisplay();
}

void display() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}


int main(int argc, char** argv) {
    initGlobalVars();
    createInitGlobalArrays();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Course project - Krivov Kirill, 80-408");

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(processNormalKeys);
    glutSpecialFunc(processSpecialKeys);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)WIDTH, 0.0, (GLdouble)HEIGHT);

    glewInit();

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

    CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

    glutMainLoop();

    CSC(cudaGraphicsUnregisterResource(res));

    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);

    deleteGlobalArrays();
    return 0;
}
