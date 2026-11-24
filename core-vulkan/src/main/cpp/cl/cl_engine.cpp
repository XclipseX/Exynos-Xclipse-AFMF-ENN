// ClEngine — xem cl_engine.h. Đường import dma-buf giống cl_stress.cpp (đã chạy được trên máy:
// clImportMemoryARM(DMA_BUF) + cl_khr_image2d_from_buffer, pitchAlign 1 px).
#include "cl_engine.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "cl_kernels_abi.h"

#ifndef CL_IMPORT_TYPE_ARM
#define CL_IMPORT_TYPE_ARM 0x40B2
#define CL_IMPORT_TYPE_DMA_BUF_ARM 0x40B4
#endif
#ifndef CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
#define CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR 0x2067
#endif
#ifndef CL_DEVICE_IMAGE_PITCH_ALIGNMENT
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT 0x104A
#endif

namespace ffcl {
namespace {

typedef intptr_t ImportPropsArm;  // cl_import_properties_arm
typedef cl_mem (*PFN_clImportMemoryARM)(cl_context, cl_mem_flags, const ImportPropsArm*, void*, size_t, cl_int*);
typedef cl_ulong MemProps;        // cl_mem_properties
typedef cl_mem (*PFN_clCreateBufferWithProperties)(cl_context, const MemProps*, cl_mem_flags, size_t, void*, cl_int*);

static_assert(sizeof(clabi::FramePC) == 64, "pc64 phải đúng 64 byte (PushConstants Vulkan)");

std::string Fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string Fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}

std::string Err(const std::string& what, cl_int e) { return "ClEngine: lỗi " + what + " " + ClErrStr(e); }

std::string DevStr(const ClApi* cl, cl_device_id d, cl_device_info k) {
    size_t n = 0;
    if (cl->GetDeviceInfo(d, k, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    cl->GetDeviceInfo(d, k, n, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

std::string PlatStr(const ClApi* cl, cl_platform_id p, cl_platform_info k) {
    size_t n = 0;
    if (cl->GetPlatformInfo(p, k, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    cl->GetPlatformInfo(p, k, n, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

}  // namespace

struct ClEngine::Slot {
    bool ready = false;
    int w = 0, h = 0, flowW = 0, flowH = 0;
    size_t nnFlowBytes = 0;
    int fd[3] = {-1, -1, -1};          // bản dup() của prev/cur/out, đóng sau khi release buffer
    cl_mem buf[3] = {nullptr, nullptr, nullptr};  // buffer import từ dma-buf
    cl_mem img[3] = {nullptr, nullptr, nullptr};  // image2d_from_buffer: prev, cur, out
    cl_mem flow = nullptr, aux = nullptr;          // RGBA16F, CL_MEM_READ_WRITE
    cl_mem nnFlow = nullptr, metrics = nullptr;
};

struct ClEngine::Impl {
    const ClApi* cl = nullptr;
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel kPack = nullptr, kWarp = nullptr;
    PFN_clImportMemoryARM importArm = nullptr;
    PFN_clCreateBufferWithProperties createWithProps = nullptr;
    std::string desc;        // platform/device/driver
    std::string importPath;  // đường import đã dùng ở SetupSlot gần nhất
    Slot slots[kMaxSlots];

    void ReleaseSlot(Slot& s) {
        for (cl_mem* m : {&s.img[0], &s.img[1], &s.img[2], &s.flow, &s.aux, &s.nnFlow, &s.metrics}) {
            if (*m) cl->ReleaseMemObject(*m);
            *m = nullptr;
        }
        // Image giữ tham chiếu tới buffer; release buffer sau image.
        for (cl_mem& m : s.buf) {
            if (m) cl->ReleaseMemObject(m);
            m = nullptr;
        }
        for (int& f : s.fd) {
            if (f >= 0) close(f);
            f = -1;
        }
        s = Slot{};
    }

    ~Impl() {
        if (!cl) return;
        if (q) cl->Finish(q);
        for (Slot& s : slots) ReleaseSlot(s);
        if (kPack) cl->ReleaseKernel(kPack);
        if (kWarp) cl->ReleaseKernel(kWarp);
        if (prog) cl->ReleaseProgram(prog);
        if (q) cl->ReleaseCommandQueue(q);
        if (ctx) cl->ReleaseContext(ctx);
    }

    // Import một dma-buf (fd đã dup) thành cl buffer. Trả nullptr + *err.
    cl_mem ImportFd(int& fd, size_t bytes, const char* tag, std::string* err) {
        cl_int eArm = CL_INVALID_VALUE, eKhr = CL_INVALID_VALUE;
        cl_mem m = nullptr;
        if (importArm) {
            const ImportPropsArm props[] = {CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
            m = importArm(ctx, CL_MEM_READ_WRITE, props, &fd, bytes, &eArm);
            if (m) { importPath = "clImportMemoryARM(DMA_BUF)"; return m; }
        }
        if (createWithProps) {
            const MemProps props[] = {CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, static_cast<MemProps>(fd), 0};
            m = createWithProps(ctx, props, CL_MEM_READ_WRITE, bytes, nullptr, &eKhr);
            if (m) { importPath = "clCreateBufferWithProperties(DMA_BUF_KHR)"; return m; }
        }
        *err = Fmt("ClEngine: import %s thất bại (fd %d, %zu B): clImportMemoryARM %s -> %s; "
                   "clCreateBufferWithProperties %s -> %s",
                   tag, fd, bytes, importArm ? "có" : "null", ClErrStr(eArm).c_str(),
                   createWithProps ? "có" : "null", ClErrStr(eKhr).c_str());
        return nullptr;
    }

    cl_mem MakeImage(cl_mem_flags flags, cl_channel_type type, int w, int h, cl_int* e) {
        cl_image_format fmt{CL_RGBA, type};
        cl_image_desc d{};
        d.image_type = CL_MEM_OBJECT_IMAGE2D;
        d.image_width = static_cast<size_t>(w);
        d.image_height = static_cast<size_t>(h);
        return cl->CreateImage(ctx, flags, &fmt, &d, nullptr, e);
    }
};

ClEngine::ClEngine() : d_(new Impl()) {}
ClEngine::~ClEngine() = default;

std::unique_ptr<ClEngine> ClEngine::Create(const std::string& clSource, std::string* info) {
    std::string dummy;
    if (!info) info = &dummy;
    if (clSource.empty()) { *info = "ClEngine: source .cl rỗng"; return nullptr; }
    const ClApi* cl = ClLoad();
    if (!cl) { *info = "ClEngine: không nạp được libOpenCL (" + ClStatus().error + ClStatus().missing + ")"; return nullptr; }

    std::unique_ptr<ClEngine> eng(new (std::nothrow) ClEngine());
    if (!eng || !eng->d_) { *info = "ClEngine: hết bộ nhớ"; return nullptr; }
    Impl& m = *eng->d_;
    m.cl = cl;
    cl_int e = CL_SUCCESS;

    cl_uint np = 0;
    if ((e = cl->GetPlatformIDs(0, nullptr, &np)) != CL_SUCCESS || np == 0) {
        *info = Err("clGetPlatformIDs", e);
        return nullptr;
    }
    std::vector<cl_platform_id> plats(np);
    cl->GetPlatformIDs(np, plats.data(), nullptr);
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        cl_device_id dv = nullptr;
        if (cl->GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dv, &nd) == CL_SUCCESS && nd > 0 && dv) {
            m.plat = p;
            m.dev = dv;
            break;
        }
    }
    if (!m.dev) { *info = "ClEngine: không có GPU device"; return nullptr; }

    const std::string exts = DevStr(cl, m.dev, CL_DEVICE_EXTENSIONS);
    if (exts.find("cl_khr_image2d_from_buffer") == std::string::npos) {
        *info = "ClEngine: thiếu cl_khr_image2d_from_buffer";
        return nullptr;
    }

    const cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(m.plat), 0};
    m.ctx = cl->CreateContext(props, 1, &m.dev, nullptr, nullptr, &e);
    if (!m.ctx) { *info = Err("clCreateContext", e); return nullptr; }
    m.q = cl->CreateCommandQueue(m.ctx, m.dev, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!m.q) { *info = Err("clCreateCommandQueue", e); return nullptr; }

    const char* src = clSource.c_str();
    const size_t len = clSource.size();
    m.prog = cl->CreateProgramWithSource(m.ctx, 1, &src, &len, &e);
    if (!m.prog) { *info = Err("clCreateProgramWithSource", e); return nullptr; }
    e = cl->BuildProgram(m.prog, 1, &m.dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t n = 0;
        cl->GetProgramBuildInfo(m.prog, m.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) cl->GetProgramBuildInfo(m.prog, m.dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        while (!log.empty() && log.back() == '\0') log.pop_back();
        if (log.size() > 3000) log.resize(3000);
        *info = Err("clBuildProgram", e) + "\n" + log;
        return nullptr;
    }
    m.kPack = cl->CreateKernel(m.prog, clabi::kFlowpackName, &e);
    if (!m.kPack) { *info = Err("clCreateKernel(flowpack)", e); return nullptr; }
    m.kWarp = cl->CreateKernel(m.prog, clabi::kWarpName, &e);
    if (!m.kWarp) { *info = Err("clCreateKernel(warp)", e); return nullptr; }

    if (cl->GetExtensionFunctionAddressForPlatform)
        m.importArm = reinterpret_cast<PFN_clImportMemoryARM>(
            cl->GetExtensionFunctionAddressForPlatform(m.plat, "clImportMemoryARM"));
    if (void* lib = dlopen(ClStatus().path.c_str(), RTLD_NOW | RTLD_NOLOAD)) {
        m.createWithProps = reinterpret_cast<PFN_clCreateBufferWithProperties>(dlsym(lib, "clCreateBufferWithProperties"));
        dlclose(lib);  // NOLOAD chỉ tăng refcount; loader vẫn giữ thư viện
    }
    if (!m.createWithProps && cl->GetExtensionFunctionAddressForPlatform)
        m.createWithProps = reinterpret_cast<PFN_clCreateBufferWithProperties>(
            cl->GetExtensionFunctionAddressForPlatform(m.plat, "clCreateBufferWithProperties"));
    if (!m.importArm && !m.createWithProps) {
        *info = "ClEngine: không có clImportMemoryARM lẫn clCreateBufferWithProperties";
        return nullptr;
    }

    cl_uint pitchAlign = 0;
    const cl_int ePa = cl->GetDeviceInfo(m.dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(pitchAlign), &pitchAlign, nullptr);
    m.desc = Fmt("%s / %s / driver %s / %s, pitchAlign %s px, import arm=%s khr=%s",
                 PlatStr(cl, m.plat, CL_PLATFORM_NAME).c_str(), DevStr(cl, m.dev, CL_DEVICE_NAME).c_str(),
                 DevStr(cl, m.dev, CL_DRIVER_VERSION).c_str(), DevStr(cl, m.dev, CL_DEVICE_VERSION).c_str(),
                 ePa == CL_SUCCESS ? std::to_string(pitchAlign).c_str() : "?", m.importArm ? "có" : "null",
                 m.createWithProps ? "có" : "null");
    *info = "ClEngine OK: " + m.desc;
    return eng;
}

bool ClEngine::SetupSlot(int slot, int prevFd, int curFd, int outFd, size_t bufBytes, int w, int h, int strideBytes,
                         int procW, int procH, std::string* err) {
    std::string dummy;
    if (!err) err = &dummy;
    Impl& m = *d_;
    const ClApi* cl = m.cl;
    if (slot < 0 || slot >= kMaxSlots) { *err = Fmt("ClEngine: slot %d sai", slot); return false; }
    Slot& s = m.slots[slot];
    m.ReleaseSlot(s);
    if (w <= 0 || h <= 0 || procW < 2 || procH < 2 || strideBytes < w * 4) {
        *err = Fmt("ClEngine: kích thước sai (w %d h %d stride %d B proc %dx%d)", w, h, strideBytes, procW, procH);
        return false;
    }
    if (bufBytes < static_cast<size_t>(strideBytes) * static_cast<size_t>(h)) {
        *err = Fmt("ClEngine: bufBytes %zu < stride %d * h %d", bufBytes, strideBytes, h);
        return false;
    }
    const int inFd[3] = {prevFd, curFd, outFd};
    const char* tags[3] = {"prev", "cur", "out"};
    for (int i = 0; i < 3; ++i) {
        if (inFd[i] < 0) { *err = Fmt("ClEngine: fd %s âm", tags[i]); return false; }
        s.fd[i] = fcntl(inFd[i], F_DUPFD_CLOEXEC, 0);
        if (s.fd[i] < 0) {
            *err = Fmt("ClEngine: dup fd %s (%d) thất bại: %s", tags[i], inFd[i], std::strerror(errno));
            m.ReleaseSlot(s);
            return false;
        }
    }

    cl_int e = CL_SUCCESS;
    auto fail = [&](const std::string& msg) {
        *err = msg + Fmt(" (slot %d)", slot);
        m.ReleaseSlot(s);
        return false;
    };
    const cl_mem_flags imgFlags[3] = {CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    for (int i = 0; i < 3; ++i) {
        s.buf[i] = m.ImportFd(s.fd[i], bufBytes, tags[i], err);
        if (!s.buf[i]) return fail(*err);
        cl_image_format fmt{CL_RGBA, CL_UNORM_INT8};
        cl_image_desc idesc{};
        idesc.image_type = CL_MEM_OBJECT_IMAGE2D;
        idesc.image_width = static_cast<size_t>(w);
        idesc.image_height = static_cast<size_t>(h);
        idesc.image_row_pitch = static_cast<size_t>(strideBytes);
        idesc.buffer = s.buf[i];
        s.img[i] = cl->CreateImage(m.ctx, imgFlags[i], &fmt, &idesc, nullptr, &e);
        if (!s.img[i])
            return fail(Err(Fmt("clCreateImage(image2d_from_buffer %s %dx%d pitch %d B)", tags[i], w, h, strideBytes), e));
    }

    s.flowW = procW / 2;
    s.flowH = procH / 2;
    // READ_WRITE: flowpack truyền như write_only, warp như read_only (CL 1.2 cho phép với READ_WRITE mem).
    if (!(s.flow = m.MakeImage(CL_MEM_READ_WRITE, CL_HALF_FLOAT, s.flowW, s.flowH, &e)))
        return fail(Err("clCreateImage(flow RGBA16F)", e));
    if (!(s.aux = m.MakeImage(CL_MEM_READ_WRITE, CL_HALF_FLOAT, s.flowW, s.flowH, &e)))
        return fail(Err("clCreateImage(aux RGBA16F)", e));
    s.nnFlowBytes = sizeof(float) * 5 * static_cast<size_t>(s.flowW) * static_cast<size_t>(s.flowH);
    if (!(s.nnFlow = cl->CreateBuffer(m.ctx, CL_MEM_READ_ONLY, s.nnFlowBytes, nullptr, &e)))
        return fail(Err("clCreateBuffer(nnFlow)", e));
    if (!(s.metrics = cl->CreateBuffer(m.ctx, CL_MEM_READ_ONLY, clabi::kMetricsBytes, nullptr, &e)))
        return fail(Err("clCreateBuffer(metrics)", e));
    s.w = w;
    s.h = h;
    s.ready = true;
    *err = Fmt("ClEngine slot %d: %dx%d pitch %d B, flow %dx%d, %s", slot, w, h, strideBytes, s.flowW, s.flowH,
               m.importPath.c_str());
    return true;
}

bool ClEngine::Run(int slot, const void* pc64, const float* nnFlow, size_t nnFlowBytes, const void* metrics64,
                   double* gpuMs, std::string* err) {
    std::string dummy;
    if (!err) err = &dummy;
    if (gpuMs) *gpuMs = -1.0;
    Impl& m = *d_;
    const ClApi* cl = m.cl;
    if (slot < 0 || slot >= kMaxSlots || !m.slots[slot].ready) { *err = Fmt("ClEngine: slot %d chưa setup", slot); return false; }
    Slot& s = m.slots[slot];
    if (!pc64 || !nnFlow || !metrics64) { *err = "ClEngine: con trỏ null"; return false; }
    if (nnFlowBytes == 0 || nnFlowBytes > s.nnFlowBytes) {
        *err = Fmt("ClEngine: nnFlowBytes %zu (buffer %zu)", nnFlowBytes, s.nnFlowBytes);
        return false;
    }
    clabi::FramePC pc;
    std::memcpy(&pc, pc64, sizeof(pc));

    const size_t M = sizeof(cl_mem);
    const size_t local[2] = {clabi::kLocalX, clabi::kLocalY};
    const size_t gPack[2] = {static_cast<size_t>(clabi::RoundUp8(s.flowW)), static_cast<size_t>(clabi::RoundUp8(s.flowH))};
    const size_t gWarp[2] = {static_cast<size_t>(clabi::RoundUp8(s.w)), static_cast<size_t>(clabi::RoundUp8(s.h))};

    cl_event ev[4] = {};
    int nev = 0;
    cl_int e = CL_SUCCESS;
    auto releaseEvs = [&]() {
        for (int i = 0; i < nev; ++i)
            if (ev[i]) cl->ReleaseEvent(ev[i]);
    };
    auto fail = [&](const char* what) {
        if (m.q) cl->Finish(m.q);  // không để lệnh dở dang tham chiếu host pointer của bên gọi
        releaseEvs();
        *err = Err(Fmt("Run slot %d: %s", slot, what), e);
        return false;
    };

    if ((e = cl->EnqueueWriteBuffer(m.q, s.nnFlow, CL_FALSE, 0, nnFlowBytes, nnFlow, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
        return fail("clEnqueueWriteBuffer(nnFlow)");
    if ((e = cl->EnqueueWriteBuffer(m.q, s.metrics, CL_FALSE, 0, clabi::kMetricsBytes, metrics64, 0, nullptr,
                                    &ev[nev++])) != CL_SUCCESS)
        return fail("clEnqueueWriteBuffer(metrics)");

    // flowpack(pc, nnFlow, flowImg, auxImg)
    if ((e = cl->SetKernelArg(m.kPack, 0, sizeof(pc), &pc)) != CL_SUCCESS) return fail("set flowpack.pc");
    if ((e = cl->SetKernelArg(m.kPack, 1, M, &s.nnFlow)) != CL_SUCCESS) return fail("set flowpack.nnFlow");
    if ((e = cl->SetKernelArg(m.kPack, 2, M, &s.flow)) != CL_SUCCESS) return fail("set flowpack.flowImg");
    if ((e = cl->SetKernelArg(m.kPack, 3, M, &s.aux)) != CL_SUCCESS) return fail("set flowpack.auxImg");
    if ((e = cl->EnqueueNDRangeKernel(m.q, m.kPack, 2, nullptr, gPack, local, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
        return fail("enqueue flowpack");

    // warp(pc, uPrev, uCur, uFlow, outImg, m, uAux)
    if ((e = cl->SetKernelArg(m.kWarp, 0, sizeof(pc), &pc)) != CL_SUCCESS) return fail("set warp.pc");
    if ((e = cl->SetKernelArg(m.kWarp, 1, M, &s.img[0])) != CL_SUCCESS) return fail("set warp.uPrev");
    if ((e = cl->SetKernelArg(m.kWarp, 2, M, &s.img[1])) != CL_SUCCESS) return fail("set warp.uCur");
    if ((e = cl->SetKernelArg(m.kWarp, 3, M, &s.flow)) != CL_SUCCESS) return fail("set warp.uFlow");
    if ((e = cl->SetKernelArg(m.kWarp, 4, M, &s.img[2])) != CL_SUCCESS) return fail("set warp.outImg");
    if ((e = cl->SetKernelArg(m.kWarp, 5, M, &s.metrics)) != CL_SUCCESS) return fail("set warp.m");
    if ((e = cl->SetKernelArg(m.kWarp, 6, M, &s.aux)) != CL_SUCCESS) return fail("set warp.uAux");
    if ((e = cl->EnqueueNDRangeKernel(m.q, m.kWarp, 2, nullptr, gWarp, local, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
        return fail("enqueue warp");

    if ((e = cl->Finish(m.q)) != CL_SUCCESS) return fail("clFinish");

    if (gpuMs) {
        cl_ulong t0 = 0, t1 = 0;
        if (cl->GetEventProfilingInfo(ev[0], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr) == CL_SUCCESS &&
            cl->GetEventProfilingInfo(ev[nev - 1], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr) == CL_SUCCESS &&
            t1 >= t0)
            *gpuMs = static_cast<double>(t1 - t0) * 1e-6;
    }
    releaseEvs();
    return true;
}

void ClEngine::ReleaseSlots() {
    Impl& m = *d_;
    if (m.q) m.cl->Finish(m.q);
    for (Slot& s : m.slots) m.ReleaseSlot(s);
}

std::string ClEngine::Describe() const {
    const Impl& m = *d_;
    return m.desc + ", path " + (m.importPath.empty() ? std::string("(chưa import)") : m.importPath);
}

}  // namespace ffcl
