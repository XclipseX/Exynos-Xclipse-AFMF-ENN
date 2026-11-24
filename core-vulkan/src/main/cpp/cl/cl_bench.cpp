// Bench kernel "warp" OpenCL trên cùng dữ liệu tổng hợp với bench Vulkan (bench_pattern.h).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../bench_pattern.h"
#include "cl_loader.h"

#include "cl_kernels_abi.h"

// Arg warp theo clabi::kWarpArgs: 0 pc (WarpPC by value) | 1 uPrev | 2 uCur | 3 uFlow | 4 outImg | 5 m | 6 uAux.
// Kernel có reqd_work_group_size(8,8,1): global = RoundUp8(out), local 8x8.
static_assert(sizeof(clabi::WarpPC) == 64, "WarpPC phải 64 byte");

namespace ffcl {
namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Giữ mọi handle, giải phóng ở destructor (mọi nhánh lỗi đều return sớm an toàn).
struct Res {
    const ClApi* cl = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel k = nullptr;
    std::vector<cl_mem> mems;
    std::vector<cl_event> evs;
    ~Res() {
        if (!cl) return;
        for (cl_event e : evs) if (e) cl->ReleaseEvent(e);
        for (cl_mem m : mems) if (m) cl->ReleaseMemObject(m);
        if (k) cl->ReleaseKernel(k);
        if (prog) cl->ReleaseProgram(prog);
        if (q) cl->ReleaseCommandQueue(q);
        if (ctx) cl->ReleaseContext(ctx);
    }
};

std::string Err(const char* what, cl_int e) { return std::string("OpenCL warp: lỗi ") + what + " " + ClErrStr(e); }

cl_mem MakeImage2D(Res& r, cl_mem_flags flags, cl_channel_type type, int w, int h, cl_int* e) {
    cl_image_format fmt{CL_RGBA, type};
    cl_image_desc d{};
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = static_cast<size_t>(w);
    d.image_height = static_cast<size_t>(h);
    cl_mem m = r.cl->CreateImage(r.ctx, flags, &fmt, &d, nullptr, e);
    if (m) r.mems.push_back(m);
    return m;
}

bool EventMs(const ClApi* cl, cl_event ev, double* ms) {
    cl_ulong s = 0, en = 0;
    if (cl->GetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr) != CL_SUCCESS) return false;
    if (cl->GetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr) != CL_SUCCESS) return false;
    *ms = static_cast<double>(en - s) * 1e-6;
    return true;
}

}  // namespace

std::string clBenchWarp(const std::string& clSource, int outW, int outH, int procW, int procH, int iters) {
    if (outW <= 0 || outH <= 0 || procW < 2 || procH < 2) return "OpenCL warp: tham số kích thước sai";
    if (iters <= 0) iters = 1;
    if (clSource.empty()) return "OpenCL warp: source .cl rỗng";
    const ClApi* cl = ClLoad();
    if (!cl) return "OpenCL warp: không nạp được libOpenCL (" + ClStatus().error + ")";

    Res r;
    r.cl = cl;
    cl_int e = CL_SUCCESS;

    // Platform/GPU đầu tiên.
    cl_uint np = 0;
    if ((e = cl->GetPlatformIDs(0, nullptr, &np)) != CL_SUCCESS || np == 0) return Err("clGetPlatformIDs", e);
    std::vector<cl_platform_id> plats(np);
    cl->GetPlatformIDs(np, plats.data(), nullptr);
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (cl->GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) == CL_SUCCESS && nd > 0) { plat = p; break; }
        dev = nullptr;
    }
    if (!dev) return "OpenCL warp: không có GPU device";

    const cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(plat), 0};
    r.ctx = cl->CreateContext(props, 1, &dev, nullptr, nullptr, &e);
    if (!r.ctx) return Err("clCreateContext", e);
    r.q = cl->CreateCommandQueue(r.ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!r.q) return Err("clCreateCommandQueue", e);

    // Build.
    const char* src = clSource.c_str();
    const size_t srcLen = clSource.size();
    r.prog = cl->CreateProgramWithSource(r.ctx, 1, &src, &srcLen, &e);
    if (!r.prog) return Err("clCreateProgramWithSource", e);
    const auto tBuild = Clock::now();
    e = cl->BuildProgram(r.prog, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
    const double buildMs = MsSince(tBuild);
    if (e != CL_SUCCESS) {
        size_t n = 0;
        cl->GetProgramBuildInfo(r.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) cl->GetProgramBuildInfo(r.prog, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        while (!log.empty() && log.back() == '\0') log.pop_back();
        if (log.size() > 4000) log.resize(4000);
        return Err("clBuildProgram", e) + "\n" + log;
    }
    r.k = cl->CreateKernel(r.prog, clabi::kWarpName, &e);
    if (!r.k) return Err("clCreateKernel(warp)", e);

    // Tài nguyên, dữ liệu dùng chung với bench Vulkan.
    const int flowW = procW / 2, flowH = procH / 2;
    cl_mem prev = MakeImage2D(r, CL_MEM_READ_ONLY, CL_UNORM_INT8, outW, outH, &e);
    if (!prev) return Err("clCreateImage(prev)", e);
    cl_mem cur = MakeImage2D(r, CL_MEM_READ_ONLY, CL_UNORM_INT8, outW, outH, &e);
    if (!cur) return Err("clCreateImage(cur)", e);
    cl_mem flow = MakeImage2D(r, CL_MEM_READ_ONLY, CL_HALF_FLOAT, flowW, flowH, &e);
    if (!flow) return Err("clCreateImage(flow RGBA16F)", e);
    cl_mem aux = MakeImage2D(r, CL_MEM_READ_ONLY, CL_HALF_FLOAT, flowW, flowH, &e);
    if (!aux) return Err("clCreateImage(aux RGBA16F)", e);
    cl_mem out = MakeImage2D(r, CL_MEM_WRITE_ONLY, CL_UNORM_INT8, outW, outH, &e);
    if (!out) return Err("clCreateImage(out)", e);
    std::vector<uint32_t> metricsHost(ff::bench::kMetricsUints, 0u);
    cl_mem metrics = cl->CreateBuffer(r.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      metricsHost.size() * sizeof(uint32_t), metricsHost.data(), &e);
    if (!metrics) return Err("clCreateBuffer(metrics)", e);
    r.mems.push_back(metrics);

    std::vector<uint8_t> prevHost(static_cast<size_t>(outW) * outH * 4);
    std::vector<uint8_t> curHost(prevHost.size());
    std::vector<uint16_t> flowHost(static_cast<size_t>(flowW) * flowH * 4);
    std::vector<uint16_t> auxHost(flowHost.size());
    ff::bench::FillRgba8(prevHost.data(), outW, outH, false);
    ff::bench::FillRgba8(curHost.data(), outW, outH, true);
    ff::bench::FillFlowRgba16f(flowHost.data(), flowW, flowH, outW, outH);
    ff::bench::FillAuxRgba16f(auxHost.data(), flowW, flowH);

    const size_t origin[3] = {0, 0, 0};
    const size_t regOut[3] = {static_cast<size_t>(outW), static_cast<size_t>(outH), 1};
    const size_t regFlow[3] = {static_cast<size_t>(flowW), static_cast<size_t>(flowH), 1};
    if ((e = cl->EnqueueWriteImage(r.q, prev, CL_TRUE, origin, regOut, 0, 0, prevHost.data(), 0, nullptr, nullptr)) != CL_SUCCESS)
        return Err("write prev", e);
    if ((e = cl->EnqueueWriteImage(r.q, cur, CL_TRUE, origin, regOut, 0, 0, curHost.data(), 0, nullptr, nullptr)) != CL_SUCCESS)
        return Err("write cur", e);
    if ((e = cl->EnqueueWriteImage(r.q, flow, CL_TRUE, origin, regFlow, 0, 0, flowHost.data(), 0, nullptr, nullptr)) != CL_SUCCESS)
        return Err("write flow", e);
    if ((e = cl->EnqueueWriteImage(r.q, aux, CL_TRUE, origin, regFlow, 0, 0, auxHost.data(), 0, nullptr, nullptr)) != CL_SUCCESS)
        return Err("write aux", e);

    // Push constants giống đường NN thật (GpuAsync/NnPost).
    clabi::WarpPC pc{};
    pc.srcW = outW; pc.srcH = outH;
    pc.lowW = procW; pc.lowH = procH;
    pc.flowW = flowW; pc.flowH = flowH;
    pc.outW = outW; pc.outH = outH;
    pc.t = 0.5f;
    pc.mode = 1;
    pc.useNN = 1;
    pc.pad0 = 0;
    pc.flowScaleX = 1.0f / static_cast<float>(procW);
    pc.flowScaleY = 1.0f / static_cast<float>(procH);
    pc.staticThr = 0.02f;
    pc.occThr = 0.14f;

    struct Arg { size_t size; const void* ptr; const char* name; };
    const Arg args[] = {
        {sizeof(pc), &pc, "pc"},          {sizeof(cl_mem), &prev, "uPrev"}, {sizeof(cl_mem), &cur, "uCur"},
        {sizeof(cl_mem), &flow, "uFlow"}, {sizeof(cl_mem), &out, "outImg"}, {sizeof(cl_mem), &metrics, "m"},
        {sizeof(cl_mem), &aux, "uAux"},
    };
    static_assert(sizeof(args) / sizeof(args[0]) == clabi::kWarpArgCount, "số arg warp lệch ABI");
    for (cl_uint i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        if ((e = cl->SetKernelArg(r.k, i, args[i].size, args[i].ptr)) != CL_SUCCESS)
            return Err((std::string("clSetKernelArg ") + args[i].name).c_str(), e);
    }

    const size_t local[2] = {clabi::kLocalX, clabi::kLocalY};
    const size_t global[2] = {static_cast<size_t>(clabi::RoundUp8(outW)), static_cast<size_t>(clabi::RoundUp8(outH))};

    // Warm-up.
    for (int i = 0; i < 5; ++i) {
        if ((e = cl->EnqueueNDRangeKernel(r.q, r.k, 2, nullptr, global, local, 0, nullptr, nullptr)) != CL_SUCCESS)
            return Err("clEnqueueNDRangeKernel(warm-up)", e);
    }
    if ((e = cl->Finish(r.q)) != CL_SUCCESS) return Err("clFinish(warm-up)", e);

    // Đo: mỗi lần 1 dispatch + clFinish (giống Vulkan submit+wait mỗi khung).
    double gpuSum = 0.0, wallSum = 0.0, gpuMin = 1e30;
    int gpuN = 0;
    for (int i = 0; i < iters; ++i) {
        cl_event ev = nullptr;
        const auto t0 = Clock::now();
        e = cl->EnqueueNDRangeKernel(r.q, r.k, 2, nullptr, global, local, 0, nullptr, &ev);
        if (e != CL_SUCCESS) return Err("clEnqueueNDRangeKernel", e);
        r.evs.push_back(ev);
        if ((e = cl->Finish(r.q)) != CL_SUCCESS) return Err("clFinish", e);
        wallSum += MsSince(t0);
        double ms = 0.0;
        if (EventMs(cl, ev, &ms)) {
            gpuSum += ms;
            if (ms < gpuMin) gpuMin = ms;
            ++gpuN;
        }
        cl->ReleaseEvent(ev);
        r.evs.back() = nullptr;
    }

    // Chi phí chép 1 khung RGBA8 host->device (nếu không có interop AHB thì phải chép mỗi khung).
    double uploadGpu = -1.0;
    cl_event upEv = nullptr;
    const auto tUp = Clock::now();
    e = cl->EnqueueWriteImage(r.q, cur, CL_FALSE, origin, regOut, 0, 0, prevHost.data(), 0, nullptr, &upEv);
    if (e != CL_SUCCESS) return Err("upload", e);
    r.evs.push_back(upEv);
    cl->Finish(r.q);
    const double uploadWall = MsSince(tUp);
    EventMs(cl, upEv, &uploadGpu);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "OpenCL warp: gpu %.2f ms (min %.2f), wall %.2f ms, upload %.2f ms (gpu %.2f) (%dx%d, proc %dx%d, %d iters, build %.0f ms)",
                  gpuN ? gpuSum / gpuN : -1.0, gpuN ? gpuMin : -1.0, wallSum / iters, uploadWall, uploadGpu,
                  outW, outH, procW, procH, iters, buildMs);
    return buf;
}

}  // namespace ffcl
