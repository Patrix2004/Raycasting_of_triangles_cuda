#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__host__ __device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator*(const float3& a, float b) { return f3(a.x * b, a.y * b, a.z * b); }
__host__ __device__ inline float3 operator*(float b, const float3& a) { return a * b; }
__host__ __device__ inline float3 operator/(const float3& a, float b) { return f3(a.x / b, a.y / b, a.z / b); }
__host__ __device__ inline float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 cross3(const float3& a, const float3& b) {
    return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float len3(const float3& a) { return sqrtf(dot3(a, a)); }
__host__ __device__ inline float3 norm3(const float3& a) {
    float l = len3(a);
    return (l > 0.0f) ? a / l : f3(0, 0, 0);
}
__host__ __device__ inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }


struct Triangle {
    float3 v0, v1, v2;
    float3 n; 
};
struct Light {
    float3 pos;
    float3 color; 
    float intensity;
};


static bool loadOBJ_Triangles(const std::string& path, std::vector<Triangle>& tris, float3& bbMin, float3& bbMax) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Cannot open file: " << path << "\n";
        return false;
    }

    std::vector<float3> verts;
    verts.reserve(20000);

    bbMin = f3(1e30f, 1e30f, 1e30f);
    bbMax = f3(-1e30f, -1e30f, -1e30f);

    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "v") {
            float x, y, z; iss >> x >> y >> z;
            if (iss.fail()) continue;
            verts.push_back(f3(x, y, z));
            bbMin.x = std::min(bbMin.x, x); bbMin.y = std::min(bbMin.y, y); bbMin.z = std::min(bbMin.z, z);
            bbMax.x = std::max(bbMax.x, x); bbMax.y = std::max(bbMax.y, y); bbMax.z = std::max(bbMax.z, z);
        }
        else if (tag == "f") {
            auto parseIndex = [](const std::string& s)->int {
                size_t p = s.find('/');
                int idx = std::stoi(p == std::string::npos ? s : s.substr(0, p));
                return idx;
                };

            std::string a, b, c;
            iss >> a >> b >> c;
            if (a.empty() || b.empty() || c.empty()) continue;

            int i0 = parseIndex(a), i1 = parseIndex(b), i2 = parseIndex(c);
            if (i0 <= 0 || i1 <= 0 || i2 <= 0) continue;
            if (i0 > (int)verts.size() || i1 > (int)verts.size() || i2 > (int)verts.size()) continue;

            float3 v0 = verts[i0 - 1];
            float3 v1 = verts[i1 - 1];
            float3 v2 = verts[i2 - 1];
            float3 n = norm3(cross3(v1 - v0, v2 - v0));
            tris.push_back({ v0,v1,v2,n });
        }
    }
    return !tris.empty();
}

static bool loadLights(const std::string& path, std::vector<Light>& lights) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        Light L;
        iss >> L.pos.x >> L.pos.y >> L.pos.z
            >> L.color.x >> L.color.y >> L.color.z
            >> L.intensity;
        if (!iss.fail()) lights.push_back(L);
    }
    return !lights.empty();
}


// Moller-Trumbore algorithm

__device__ inline bool rayTri(const float3& O, const float3& D, const Triangle& T, float& tOut) {
    const float3 v0 = T.v0, v1 = T.v1, v2 = T.v2;
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 p = cross3(D, e2);
    const float det = dot3(e1, p);
    if (fabsf(det) < 1e-8f) return false;
    const float invDet = 1.0f / det;
    const float3 s = O - v0;
    const float u = dot3(s, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    const float3 q = cross3(s, e1);
    const float v = dot3(D, q) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;
    const float t = dot3(e2, q) * invDet;
    if (t <= 0.0f) return false;
    tOut = t;
    return true;
}

__device__ inline uint32_t packRGBA8(float r, float g, float b) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b);
    uint8_t R = (uint8_t)lrintf(r * 255.0f);
    uint8_t G = (uint8_t)lrintf(g * 255.0f);
    uint8_t B = (uint8_t)lrintf(b * 255.0f);
    uint8_t A = 255;
    return (A << 24) | (B << 16) | (G << 8) | (R);
}


__global__ void renderKernel(
    uint32_t* outRGBA, int W, int H,
    const Triangle* tris, int triCount,
    const Light* lights, int lightCount,
    float3 center,
    float worldHalfSize,
    float camZ,
    float3 Ka, float3 Kd, float3 Ks, float shininess,
    float angleX, float angleY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float fx = ((x + 0.5f) / (float)W) * 2.0f - 1.0f;
    float fy = ((y + 0.5f) / (float)H) * 2.0f - 1.0f;
    float aspect = (float)W / (float)H;
    fx *= aspect;

    float3 O = f3(fx * worldHalfSize, fy * worldHalfSize, camZ);
    float3 D = f3(0, 0, -1);


    float cy = cosf(angleY), sy = sinf(angleY);
    float cx = cosf(angleX), sx = sinf(angleX);

    float ox = O.x - center.x;
    float oy = O.y - center.y;
    float oz = O.z - center.z;

    float ox1 = cy * ox + sy * oz;
    float oz1 = -sy * ox + cy * oz;
    float oy1 = oy;

    float oy2 = cx * oy1 - sx * oz1;
    float oz2 = sx * oy1 + cx * oz1;
    float ox2 = ox1;

    O = f3(ox2 + center.x, oy2 + center.y, oz2 + center.z);

    float dx = D.x, dy = D.y, dz = D.z;

    float dx1 = cy * dx + sy * dz;
    float dz1 = -sy * dx + cy * dz;
    float dy1 = dy;

    float dy2 = cx * dy1 - sx * dz1;
    float dz2 = sx * dy1 + cx * dz1;
    float dx2 = dx1;

    D = norm3(f3(dx2, dy2, dz2));

    float bestT = 1e30f;
    int bestIdx = -1;

    for (int i = 0; i < triCount; i++) {
        float t;
        if (rayTri(O, D, tris[i], t)) {
            if (t < bestT) { bestT = t; bestIdx = i; }
        }
    }

    float3 col;

    if (bestIdx >= 0) {
        const Triangle T = tris[bestIdx];
        const float3 N = norm3(T.n);
        const float3 P = O + D * bestT;

        const float3 V = f3(0, 0, 1);

        col = Ka;

        for (int li = 0; li < lightCount; li++) {
            const Light L = lights[li];
            const float3 Ldir = norm3(L.pos - P);

            float NdotL = fmaxf(0.0f, dot3(N, Ldir));
            float3 diffuse = Kd * NdotL;

            float3 Rr = norm3(N * (2.0f * dot3(N, Ldir)) - Ldir);
            float spec = powf(fmaxf(0.0f, dot3(Rr, V)), shininess);
            float3 specular = Ks * spec;

            float3 lightRGB = L.color * L.intensity;
            col = col + f3(
                (diffuse.x + specular.x) * lightRGB.x,
                (diffuse.y + specular.y) * lightRGB.y,
                (diffuse.z + specular.z) * lightRGB.z
            );
        }
    }
    else {
        col = f3(0.02f, 0.02f, 0.03f);
    }

    outRGBA[(H - 1 - y) * W + x] = packRGBA8(col.x, col.y, col.z);
}


static GLuint gPBO = 0;
static cudaGraphicsResource* gCudaPBO = nullptr;

static float gLastRenderMs = 0.0f;

static void createPBO(int W, int H) {
    if (gPBO) { glDeleteBuffers(1, &gPBO); gPBO = 0; }

    glGenBuffers(1, &gPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)W * H * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (gCudaPBO) { cudaGraphicsUnregisterResource(gCudaPBO); gCudaPBO = nullptr; }
    cudaGraphicsGLRegisterBuffer(&gCudaPBO, gPBO, cudaGraphicsRegisterFlagsWriteDiscard);
}

static void drawPBO(int W, int H) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gPBO);
    glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


static void renderToPBO(
    int W, int H,
    Triangle* dTris, int triCount,
    Light* dLights, int lightCount,
    float3 center,
    float worldHalfSize,
    float camZ,
    float3 Ka, float3 Kd, float3 Ks, float shininess,
    float angleX, float angleY
) {
    
    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    cudaEventRecord(evStart, 0);

    cudaGraphicsMapResources(1, &gCudaPBO);

    uint32_t* dOut = nullptr;
    size_t sizeBytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&dOut, &sizeBytes, gCudaPBO);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    renderKernel <<<grid, block >>> (
        dOut, W, H,
        dTris, triCount,
        dLights, lightCount,
        center,
        worldHalfSize,
        camZ,
        Ka, Kd, Ks, shininess,
        angleX, angleY
        );


	//test for 1 light only
    //renderKernel << <grid, block >> > (
    //    dOut, W, H,
    //    dTris, triCount,
    //    dLights, 1,
    //    center,
    //    worldHalfSize,
    //    camZ,
    //    Ka, Kd, Ks, shininess,
    //    angleX, angleY
    //    );

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &gCudaPBO);

    cudaEventRecord(evStop, 0);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gLastRenderMs, evStart, evStop);

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
}

static void rotateLightsY(std::vector<Light>& lights, float deltaAngle, float3 center) {
    float c = cosf(deltaAngle), s = sinf(deltaAngle);
    for (auto& L : lights) {
        float x = L.pos.x - center.x;
        float z = L.pos.z - center.z;
        float xr = c * x + s * z;
        float zr = -s * x + c * z;
        L.pos.x = xr + center.x;
        L.pos.z = zr + center.z;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " model.obj lights.txt\n";
        return 1;
    }

    std::vector<Triangle> hTris;
    std::vector<Light> hLights;
    float3 bbMin, bbMax;

    std::cout << "Trying to open model: " << argv[1] << "\n";

    if (!loadOBJ_Triangles(argv[1], hTris, bbMin, bbMax)) {
        std::cerr << "Failed to load OBJ triangles: " << argv[1] << "\n";
        return 1;
    }
    if (!loadLights(argv[2], hLights)) {
        std::cerr << "Failed to load lights: " << argv[2] << "\n";
        return 1;
    }

    std::cout << "Triangles: " << hTris.size() << "\n";
    std::cout << "Lights: " << hLights.size() << "\n";

    float3 center = f3(
        0.5f * (bbMin.x + bbMax.x),
        0.5f * (bbMin.y + bbMax.y),
        0.5f * (bbMin.z + bbMax.z)
    );
    float3 extent = bbMax - bbMin;
    float sceneRadius = 0.5f * len3(extent);
    if (sceneRadius < 1e-3f) sceneRadius = 1.0f;

    float camZ = center.z + sceneRadius * 2.5f;
    float worldHalfSize = sceneRadius * 1.2f;

    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    int W = 640, H = 360;

    GLFWwindow* win = glfwCreateWindow(W, H, "CUDA Raycasting - Phong (WASD scene, R lights)", nullptr, nullptr);
    if (!win) {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "glewInit failed: " << glewGetErrorString(glewErr) << "\n";
        return 1;
    }

    glViewport(0, 0, W, H);

    createPBO(W, H);

    Triangle* dTris = nullptr;
    Light* dLights = nullptr;

    cudaMalloc(&dTris, hTris.size() * sizeof(Triangle));
    cudaMemcpy(dTris, hTris.data(), hTris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dLights, hLights.size() * sizeof(Light));
    cudaMemcpy(dLights, hLights.data(), hLights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	// Phong parameters
    float3 Ka = f3(0.10f, 0.10f, 0.10f);
    float3 Kd = f3(0.70f, 0.70f, 0.70f);
    float3 Ks = f3(0.35f, 0.35f, 0.35f);
    float shininess = 32.0f;

    // Rotation state
    float angleX = 0.0f;
    float angleY = 0.0f;

    renderToPBO(W, H, dTris, (int)hTris.size(), dLights, (int)hLights.size(),
        center, worldHalfSize, camZ, Ka, Kd, Ks, shininess, angleX, angleY);

    
    {
        float fps = (gLastRenderMs > 0.0f) ? (1000.0f / gLastRenderMs) : 0.0f;
        char title[256];
        std::snprintf(title, sizeof(title),
            "CUDA Raycasting - render %.2f ms | %.1f FPS (WASD scene, R lights)",
            gLastRenderMs, fps);
        glfwSetWindowTitle(win, title);
        std::cout << "Render: " << gLastRenderMs << " ms | FPS: " << fps << "\n";
    }

    bool lastR = false;

    while (!glfwWindowShouldClose(win)) {
        bool changed = false;

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(win, 1);
        }

        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) { angleY += 0.13f; changed = true; }
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) { angleY -= 0.13f; changed = true; }
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) { angleX += 0.13f; changed = true; }
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) { angleX -= 0.13f; changed = true; }

    
        bool nowR = (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS);
        if (nowR && !lastR) {
            rotateLightsY(hLights, 0.25f, center);
            cudaMemcpy(dLights, hLights.data(), hLights.size() * sizeof(Light), cudaMemcpyHostToDevice);
            changed = true;
        }
        lastR = nowR;

        if (changed) {
            renderToPBO(W, H, dTris, (int)hTris.size(), dLights, (int)hLights.size(),
                center, worldHalfSize, camZ, Ka, Kd, Ks, shininess, angleX, angleY);

            float fps = (gLastRenderMs > 0.0f) ? (1000.0f / gLastRenderMs) : 0.0f;
            char title[256];
            std::snprintf(title, sizeof(title),
                "CUDA Raycasting - render %.2f ms | %.1f FPS (WASD scene, R lights)",
                gLastRenderMs, fps);
            glfwSetWindowTitle(win, title);

            std::cout << "Render: " << gLastRenderMs << " ms | FPS: " << fps << "\n";
        }

        
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2f(-1.f, -1.f);
        drawPBO(W, H);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

   
    cudaFree(dTris);
    cudaFree(dLights);

    if (gCudaPBO) cudaGraphicsUnregisterResource(gCudaPBO);
    if (gPBO) glDeleteBuffers(1, &gPBO);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
