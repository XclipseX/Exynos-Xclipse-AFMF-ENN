// Stress chuỗi khung OpenCL mô phỏng đường NN Vulkan (NnPre: preprocess; NnPost: flowpack + warp mode 1).
// Xem cl_stress.h. Không ném ngoại lệ ra ngoài cố ý; mọi lỗi trả về qua info.
#include "cl_stress.h"

#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "../bench_pattern.h"
#include "cl_kernels_abi.h"
#include "cl_loader.h"

// Hằng số không có trong cl_loader.h. Giá trị đã đối chiếu Khronos OpenCL-Headers CL/cl_ext.h (main).
#ifndef CL_IMPORT_TYPE_ARM
#define CL_IMPORT_TYPE_ARM 0x40B2
#define CL_IMPORT_TYPE_DMA_BUF_ARM 0x40B4
#endif
#ifndef CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
#define CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR 0x2067
#endif
#ifndef CL_DEVICE_IMAGE_PITCH_ALIGNMENT
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT 0x104A          // CL/cl.h (OpenCL 2.0 / cl_khr_image2d_from_buffer)
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT 0x104B   // CL/cl.h
#endif

namespace ffcl {
namespace {

typedef intptr_t ImportPropsArm;  // cl_import_properties_arm
typedef cl_mem (*PFN_clImportMemoryARM)(cl_context, cl_mem_flags, const ImportPropsArm*, void*, size_t, cl_int*);
typedef cl_ulong MemProps;        // cl_mem_properties
typedef cl_mem (*PFN_clCreateBufferWithProperties)(cl_context, const MemProps*, cl_mem_flags, size_t, void*, cl_int*);

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

std::string Fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string Fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}

std::string Err(const std::string& what, cl_int e) { return "CL stress: lỗi " + what + " " + ClErrStr(e); }

struct Stress {
    std::mutex mu;
    const ClApi* cl = nullptr;
    int mode = 0;
    int outW = 0, outH = 0, procW = 0, procH = 0, flowW = 0, flowH = 0;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel kPre = nullptr, kPack = nullptr, kWarp = nullptr;
    cl_mem img[2] = {nullptr, nullptr};     // img[0] = mẫu prev, img[1] = mẫu cur (đổi vai mỗi khung)
    cl_mem imgBuf[2] = {nullptr, nullptr};  // mode 1: buffer import từ dma-buf
    cl_mem lowPrev = nullptr, lowCur = nullptr, nnIn = nullptr, nnFlow = nullptr;
    cl_mem flow = nullptr, aux = nullptr, out = nullptr, metrics = nullptr;
    AHardwareBuffer* ahb[2] = {nullptr, nullptr};
    int fd[2] = {-1, -1};
    std::vector<uint8_t> host[2];          // mode 0: dữ liệu upload (packed outW*4)
    std::vector<float> nnFlowHost;
    uint32_t zeros[16] = {};
    clabi::FramePC pc{};
    long long frameNo = 0;

    ~Stress() {
        if (cl) {
            if (q) cl->Finish(q);
            for (cl_mem m : {img[0], img[1], imgBuf[0], imgBuf[1], lowPrev, lowCur, nnIn, nnFlow, flow, aux, out, metrics})
                if (m) cl->ReleaseMemObject(m);
            for (cl_kernel k : {kPre, kPack, kWarp}) if (k) cl->ReleaseKernel(k);
            if (prog) cl->ReleaseProgram(prog);
            if (q) cl->ReleaseCommandQueue(q);
            if (ctx) cl->ReleaseContext(ctx);
        }
        for (int i = 0; i < 2; ++i) {
            if (fd[i] >= 0) close(fd[i]);
            if (ahb[i]) AHardwareBuffer_release(ahb[i]);
        }
    }
};

cl_mem MakeImage(Stress& s, cl_mem_flags flags, cl_channel_type type, int w, int h, cl_int* e) {
    cl_image_format fmt{CL_RGBA, type};
    cl_image_desc d{};
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = static_cast<size_t>(w);
    d.image_height = static_cast<size_t>(h);
    return s.cl->CreateImage(s.ctx, flags, &fmt, &d, nullptr, e);
}

// Lấy dma-buf fd của AHB không qua API vndk: gửi native_handle qua socketpair rồi nhận SCM_RIGHTS.
// Trả fd (fd đầu tiên) hoặc -1 + *err.
int AhbDmaBufFd(const AHardwareBuffer* b, std::string* err) {
    int sv[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv) != 0) {
        *err = Fmt("socketpair errno %d (%s)", errno, std::strerror(errno));
        return -1;
    }
    const int rs = AHardwareBuffer_sendHandleToUnixSocket(b, sv[0]);
    if (rs != 0) {
        *err = Fmt("AHardwareBuffer_sendHandleToUnixSocket = %d", rs);
        close(sv[0]); close(sv[1]);
        return -1;
    }
    constexpr int kMaxFds = 32;
    std::vector<uint8_t> data(16384);
    alignas(struct cmsghdr) char ctrl[CMSG_SPACE(sizeof(int) * kMaxFds)];
    std::memset(ctrl, 0, sizeof(ctrl));
    iovec iov{data.data(), data.size()};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl;
    msg.msg_controllen = sizeof(ctrl);
    ssize_t n;
    do { n = recvmsg(sv[1], &msg, MSG_CMSG_CLOEXEC | MSG_DONTWAIT); } while (n < 0 && errno == EINTR);
    const int recvErrno = errno;
    close(sv[0]); close(sv[1]);
    std::vector<int> fds;
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            const size_t cnt = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            const int* p = reinterpret_cast<const int*>(CMSG_DATA(c));
            for (size_t i = 0; i < cnt; ++i) { int v; std::memcpy(&v, p + i, sizeof(int)); fds.push_back(v); }
        }
    }
    if (n < 0) {
        for (int f : fds) close(f);
        *err = Fmt("recvmsg errno %d (%s)", recvErrno, std::strerror(recvErrno));
        return -1;
    }
    if (msg.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) {
        for (int f : fds) close(f);
        *err = Fmt("recvmsg bị cắt (flags 0x%x, %zd byte)", msg.msg_flags, n);
        return -1;
    }
    if (fds.empty()) {
        *err = Fmt("recvmsg không có SCM_RIGHTS (%zd byte)", n);
        return -1;
    }
    for (size_t i = 1; i < fds.size(); ++i) close(fds[i]);
    if (fds.size() > 1) *err = Fmt("%zu fd, dùng fd[0]", fds.size());
    return fds[0];
}

bool BuildProgram(Stress& s, cl_device_id dev, const std::string& src, std::string* info) {
    cl_int e = CL_SUCCESS;
    const char* p = src.c_str();
    const size_t len = src.size();
    s.prog = s.cl->CreateProgramWithSource(s.ctx, 1, &p, &len, &e);
    if (!s.prog) { *info = Err("clCreateProgramWithSource", e); return false; }
    e = s.cl->BuildProgram(s.prog, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t n = 0;
        s.cl->GetProgramBuildInfo(s.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) s.cl->GetProgramBuildInfo(s.prog, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        while (!log.empty() && log.back() == '\0') log.pop_back();
        if (log.size() > 3000) log.resize(3000);
        *info = Err("clBuildProgram", e) + "\n" + log;
        return false;
    }
    s.kPre = s.cl->CreateKernel(s.prog, clabi::kPreprocessName, &e);
    if (!s.kPre) { *info = Err("clCreateKernel(preprocess)", e); return false; }
    s.kPack = s.cl->CreateKernel(s.prog, clabi::kFlowpackName, &e);
    if (!s.kPack) { *info = Err("clCreateKernel(flowpack)", e); return false; }
    s.kWarp = s.cl->CreateKernel(s.prog, clabi::kWarpName, &e);
    if (!s.kWarp) { *info = Err("clCreateKernel(warp)", e); return false; }
    return true;
}

// mode 1: 2 AHB -> dma-buf -> cl buffer -> image2d. Ghi chi tiết vào *desc / *info.
bool SetupDmaBuf(Stress& s, cl_platform_id plat, cl_device_id dev, std::string* desc, std::string* info) {
    const ClApi* cl = s.cl;
    std::string exts;
    {
        size_t n = 0;
        cl->GetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &n);
        exts.assign(n, '\0');
        if (n) cl->GetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, n, &exts[0], nullptr);
    }
    const bool hasArm = exts.find("cl_arm_import_memory_dma_buf") != std::string::npos ||
                        exts.find("cl_arm_import_memory ") != std::string::npos;
    const bool hasKhr = exts.find("cl_khr_external_memory_dma_buf") != std::string::npos;
    const bool hasI2B = exts.find("cl_khr_image2d_from_buffer") != std::string::npos;

    cl_uint pitchAlign = 0, baseAlign = 0;
    const cl_int ePa = cl->GetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(pitchAlign), &pitchAlign, nullptr);
    const cl_int eBa = cl->GetDeviceInfo(dev, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(baseAlign), &baseAlign, nullptr);

    PFN_clImportMemoryARM importArm = nullptr;
    if (cl->GetExtensionFunctionAddressForPlatform)
        importArm = reinterpret_cast<PFN_clImportMemoryARM>(cl->GetExtensionFunctionAddressForPlatform(plat, "clImportMemoryARM"));
    PFN_clCreateBufferWithProperties createWithProps = nullptr;
    if (void* lib = dlopen(ClStatus().path.c_str(), RTLD_NOW | RTLD_NOLOAD)) {
        createWithProps = reinterpret_cast<PFN_clCreateBufferWithProperties>(dlsym(lib, "clCreateBufferWithProperties"));
        dlclose(lib);  // NOLOAD chỉ tăng refcount; loader vẫn giữ thư viện
    }
    if (!createWithProps && cl->GetExtensionFunctionAddressForPlatform)
        createWithProps = reinterpret_cast<PFN_clCreateBufferWithProperties>(
            cl->GetExtensionFunctionAddressForPlatform(plat, "clCreateBufferWithProperties"));

    const std::string caps = Fmt("ext arm=%d khr_dmabuf=%d img2d_from_buf=%d, clImportMemoryARM=%s, "
                                 "clCreateBufferWithProperties=%s, pitchAlign=%u px (%s), baseAlign=%u (%s)",
                                 hasArm, hasKhr, hasI2B, importArm ? "có" : "null", createWithProps ? "có" : "null",
                                 pitchAlign, ePa == CL_SUCCESS ? "ok" : ClErrStr(ePa).c_str(), baseAlign,
                                 eBa == CL_SUCCESS ? "ok" : ClErrStr(eBa).c_str());

    std::string paths;
    for (int i = 0; i < 2; ++i) {
        const char* tag = i == 0 ? "AHB prev" : "AHB cur";
        AHardwareBuffer_Desc d{};
        d.width = static_cast<uint32_t>(s.outW);
        d.height = static_cast<uint32_t>(s.outH);
        d.layers = 1;
        d.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
        d.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        const int ra = AHardwareBuffer_allocate(&d, &s.ahb[i]);
        if (ra != 0 || !s.ahb[i]) {
            s.ahb[i] = nullptr;
            *info = Fmt("CL stress dmabuf: %s AHardwareBuffer_allocate(%dx%d RGBA8, usage 0x%llx) = %d", tag, s.outW,
                        s.outH, static_cast<unsigned long long>(d.usage), ra);
            return false;
        }
        AHardwareBuffer_Desc got{};
        AHardwareBuffer_describe(s.ahb[i], &got);
        const uint32_t stride = got.stride;  // pixel
        if (stride < static_cast<uint32_t>(s.outW)) {
            *info = Fmt("CL stress dmabuf: %s stride %u < width %d", tag, stride, s.outW);
            return false;
        }

        // Điền mẫu một lần qua CPU, theo stride.
        void* va = nullptr;
        const int rl = AHardwareBuffer_lock(s.ahb[i], AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &va);
        if (rl != 0 || !va) {
            *info = Fmt("CL stress dmabuf: %s AHardwareBuffer_lock = %d (stride %u)", tag, rl, stride);
            return false;
        }
        {
            std::vector<uint8_t> tmp(static_cast<size_t>(s.outW) * s.outH * 4);
            ff::bench::FillRgba8(tmp.data(), s.outW, s.outH, i == 1);
            const size_t rowBytes = static_cast<size_t>(s.outW) * 4;
            for (int y = 0; y < s.outH; ++y)
                std::memcpy(static_cast<uint8_t*>(va) + static_cast<size_t>(y) * stride * 4, tmp.data() + y * rowBytes, rowBytes);
        }
        AHardwareBuffer_unlock(s.ahb[i], nullptr);

        std::string fdNote;
        s.fd[i] = AhbDmaBufFd(s.ahb[i], &fdNote);
        if (s.fd[i] < 0) {
            *info = Fmt("CL stress dmabuf: %s lấy dma-buf fd thất bại: %s (stride %u)", tag, fdNote.c_str(), stride);
            return false;
        }
        const off_t fdSize = lseek(s.fd[i], 0, SEEK_END);
        lseek(s.fd[i], 0, SEEK_SET);
        const size_t bytes = static_cast<size_t>(stride) * 4 * static_cast<size_t>(s.outH);

        // 1) cl_arm_import_memory (dma_buf)
        cl_int eArm = CL_SUCCESS, eKhr = CL_SUCCESS;
        const char* used = nullptr;
        if (importArm) {
            const ImportPropsArm props[] = {CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
            s.imgBuf[i] = importArm(s.ctx, CL_MEM_READ_WRITE, props, &s.fd[i], bytes, &eArm);
            if (s.imgBuf[i]) used = "clImportMemoryARM(DMA_BUF)";
        } else {
            eArm = CL_INVALID_VALUE;
        }
        // 2) dự phòng: clCreateBufferWithProperties + CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
        if (!s.imgBuf[i] && createWithProps) {
            const MemProps props[] = {CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, static_cast<MemProps>(s.fd[i]), 0};
            s.imgBuf[i] = createWithProps(s.ctx, props, CL_MEM_READ_WRITE, bytes, nullptr, &eKhr);
            if (s.imgBuf[i]) used = "clCreateBufferWithProperties(DMA_BUF_KHR)";
        } else if (!s.imgBuf[i]) {
            eKhr = CL_INVALID_VALUE;
        }
        if (!s.imgBuf[i]) {
            *info = Fmt("CL stress dmabuf: %s import thất bại: clImportMemoryARM %s -> %s; "
                        "clCreateBufferWithProperties %s -> %s (fd %d, fdSize %lld, size %zu, stride %u). ",
                        tag, importArm ? "có" : "null", ClErrStr(eArm).c_str(), createWithProps ? "có" : "null",
                        ClErrStr(eKhr).c_str(), s.fd[i], static_cast<long long>(fdSize), bytes, stride) + caps;
            return false;
        }

        // 3) image2d từ buffer (cl_khr_image2d_from_buffer)
        cl_image_format fmt{CL_RGBA, CL_UNORM_INT8};
        cl_image_desc idesc{};
        idesc.image_type = CL_MEM_OBJECT_IMAGE2D;
        idesc.image_width = static_cast<size_t>(s.outW);
        idesc.image_height = static_cast<size_t>(s.outH);
        idesc.image_row_pitch = static_cast<size_t>(stride) * 4;
        idesc.buffer = s.imgBuf[i];
        cl_int eImg = CL_SUCCESS;
        s.img[i] = cl->CreateImage(s.ctx, CL_MEM_READ_ONLY, &fmt, &idesc, nullptr, &eImg);
        const bool pitchOk = pitchAlign == 0 || stride % pitchAlign == 0;
        if (!s.img[i]) {
            *info = Fmt("CL stress dmabuf: %s clCreateImage(image2d_from_buffer, row_pitch %zu) %s qua %s; stride %u px, "
                        "pitchAlign %u px%s. ",
                        tag, idesc.image_row_pitch, ClErrStr(eImg).c_str(), used, stride, pitchAlign,
                        pitchOk ? "" : " -> LỆCH CĂN (stride không chia hết pitchAlign)") + caps;
            return false;
        }
        paths += Fmt("%s%s: %s, stride %u px%s, fd %d (%s, %lld B)", i ? "; " : "", tag, used, stride,
                     pitchOk ? "" : " (LỆCH pitchAlign nhưng driver chấp nhận)", s.fd[i],
                     fdNote.empty() ? "1 fd" : fdNote.c_str(), static_cast<long long>(fdSize));
    }
    *desc = "dmabuf zero-copy: " + paths + ". " + caps;
    return true;
}

bool SetArg(Stress& s, cl_kernel k, cl_uint i, size_t sz, const void* p, const char* name, std::string* info) {
    const cl_int e = s.cl->SetKernelArg(k, i, sz, p);
    if (e != CL_SUCCESS) { *info = Err(std::string("clSetKernelArg ") + name, e); return false; }
    return true;
}

}  // namespace

void* clStressOpen(const std::string& src, int outW, int outH, int procW, int procH, int mode, std::string* info) {
    std::string dummy;
    if (!info) info = &dummy;
    if (outW <= 0 || outH <= 0 || procW < 2 || procH < 2) { *info = "CL stress: kích thước sai"; return nullptr; }
    if (mode != 0 && mode != 1) { *info = "CL stress: mode phải 0 (copy) hoặc 1 (dmabuf)"; return nullptr; }
    if (src.empty()) { *info = "CL stress: source .cl rỗng"; return nullptr; }
    const ClApi* cl = ClLoad();
    if (!cl) { *info = "CL stress: không nạp được libOpenCL (" + ClStatus().error + ")"; return nullptr; }

    auto* s = new (std::nothrow) Stress();
    if (!s) { *info = "CL stress: hết bộ nhớ"; return nullptr; }
    struct Guard { Stress* p; ~Guard() { delete p; } } guard{s};
    s->cl = cl;
    s->mode = mode;
    s->outW = outW; s->outH = outH; s->procW = procW; s->procH = procH;
    s->flowW = procW / 2; s->flowH = procH / 2;
    cl_int e = CL_SUCCESS;

    cl_uint np = 0;
    if ((e = cl->GetPlatformIDs(0, nullptr, &np)) != CL_SUCCESS || np == 0) { *info = Err("clGetPlatformIDs", e); return nullptr; }
    std::vector<cl_platform_id> plats(np);
    cl->GetPlatformIDs(np, plats.data(), nullptr);
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (cl->GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) == CL_SUCCESS && nd > 0) { plat = p; break; }
        dev = nullptr;
    }
    if (!dev) { *info = "CL stress: không có GPU device"; return nullptr; }
    const cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(plat), 0};
    s->ctx = cl->CreateContext(props, 1, &dev, nullptr, nullptr, &e);
    if (!s->ctx) { *info = Err("clCreateContext", e); return nullptr; }
    s->q = cl->CreateCommandQueue(s->ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!s->q) { *info = Err("clCreateCommandQueue", e); return nullptr; }
    if (!BuildProgram(*s, dev, src, info)) return nullptr;

    // Ảnh khung vào.
    std::string desc;
    if (mode == 0) {
        for (int i = 0; i < 2; ++i) {
            s->host[i].resize(static_cast<size_t>(outW) * outH * 4);
            ff::bench::FillRgba8(s->host[i].data(), outW, outH, i == 1);
            s->img[i] = MakeImage(*s, CL_MEM_READ_ONLY, CL_UNORM_INT8, outW, outH, &e);
            if (!s->img[i]) { *info = Err("clCreateImage(frame RGBA8)", e); return nullptr; }
            const size_t origin[3] = {0, 0, 0};
            const size_t reg[3] = {static_cast<size_t>(outW), static_cast<size_t>(outH), 1};
            if ((e = cl->EnqueueWriteImage(s->q, s->img[i], CL_TRUE, origin, reg, 0, 0, s->host[i].data(), 0, nullptr,
                                           nullptr)) != CL_SUCCESS) {
                *info = Err("clEnqueueWriteImage(khởi tạo)", e);
                return nullptr;
            }
        }
        desc = "copy: clEnqueueWriteImage RGBA8 mỗi khung";
    } else {
        if (!SetupDmaBuf(*s, plat, dev, &desc, info)) return nullptr;
    }

    // Tài nguyên trung gian giống AFrame Vulkan.
    const int fw = s->flowW, fh = s->flowH;
    if (!(s->lowPrev = MakeImage(*s, CL_MEM_READ_WRITE, CL_UNORM_INT8, procW, procH, &e))) { *info = Err("clCreateImage(lowPrev)", e); return nullptr; }
    if (!(s->lowCur = MakeImage(*s, CL_MEM_READ_WRITE, CL_UNORM_INT8, procW, procH, &e))) { *info = Err("clCreateImage(lowCur)", e); return nullptr; }
    if (!(s->flow = MakeImage(*s, CL_MEM_READ_WRITE, CL_HALF_FLOAT, fw, fh, &e))) { *info = Err("clCreateImage(flow RGBA16F)", e); return nullptr; }
    if (!(s->aux = MakeImage(*s, CL_MEM_READ_WRITE, CL_HALF_FLOAT, fw, fh, &e))) { *info = Err("clCreateImage(aux RGBA16F)", e); return nullptr; }
    if (!(s->out = MakeImage(*s, CL_MEM_WRITE_ONLY, CL_UNORM_INT8, outW, outH, &e))) { *info = Err("clCreateImage(out)", e); return nullptr; }
    const size_t nnInBytes = sizeof(float) * 6 * static_cast<size_t>(procW) * procH;
    const size_t nnFlowBytes = sizeof(float) * 5 * static_cast<size_t>(fw) * fh;
    if (!(s->nnIn = cl->CreateBuffer(s->ctx, CL_MEM_READ_WRITE, nnInBytes, nullptr, &e))) { *info = Err("clCreateBuffer(nnIn)", e); return nullptr; }
    if (!(s->nnFlow = cl->CreateBuffer(s->ctx, CL_MEM_READ_ONLY, nnFlowBytes, nullptr, &e))) { *info = Err("clCreateBuffer(nnFlow)", e); return nullptr; }
    if (!(s->metrics = cl->CreateBuffer(s->ctx, CL_MEM_READ_WRITE, clabi::kMetricsBytes, nullptr, &e))) { *info = Err("clCreateBuffer(metrics)", e); return nullptr; }

    // nnFlow tổng hợp: pixel proc-res, chuyển động đều (kDx,kDy) pixel khung ra; mask 0.5.
    {
        const size_t plane = static_cast<size_t>(fw) * fh;
        s->nnFlowHost.assign(5 * plane, 0.0f);
        const float fx = 0.5f * ff::bench::kDx * static_cast<float>(procW) / static_cast<float>(outW);
        const float fy = 0.5f * ff::bench::kDy * static_cast<float>(procH) / static_cast<float>(outH);
        const float v[5] = {-fx, -fy, fx, fy, 0.5f};
        for (int c = 0; c < 5; ++c) std::fill(s->nnFlowHost.begin() + c * plane, s->nnFlowHost.begin() + (c + 1) * plane, v[c]);
    }

    // Push constants như NnPre/NnPost (staticThr/occThr/pad0 = mặc định tune).
    clabi::FramePC& pc = s->pc;
    pc.srcW = outW; pc.srcH = outH;
    pc.lowW = procW; pc.lowH = procH;
    pc.flowW = fw; pc.flowH = fh;
    pc.outW = outW; pc.outH = outH;
    pc.t = 0.5f;
    pc.mode = 1;
    pc.useNN = 1;
    pc.pad0 = 0;
    pc.flowScaleX = 1.0f / static_cast<float>(procW);
    pc.flowScaleY = 1.0f / static_cast<float>(procH);
    pc.staticThr = 0.02f;
    pc.occThr = 0.14f;

    // Arg cố định (prev/cur đặt lại mỗi khung).
    const size_t M = sizeof(cl_mem);
    if (!SetArg(*s, s->kPre, 0, sizeof(pc), &pc, "preprocess.pc", info) ||
        !SetArg(*s, s->kPre, 3, M, &s->lowPrev, "preprocess.lowPrev", info) ||
        !SetArg(*s, s->kPre, 4, M, &s->lowCur, "preprocess.lowCur", info) ||
        !SetArg(*s, s->kPre, 5, M, &s->nnIn, "preprocess.nnIn", info) ||
        !SetArg(*s, s->kPre, 6, M, &s->metrics, "preprocess.m", info) ||
        !SetArg(*s, s->kPack, 0, sizeof(pc), &pc, "flowpack.pc", info) ||
        !SetArg(*s, s->kPack, 1, M, &s->nnFlow, "flowpack.nnFlow", info) ||
        !SetArg(*s, s->kPack, 2, M, &s->flow, "flowpack.flowImg", info) ||
        !SetArg(*s, s->kPack, 3, M, &s->aux, "flowpack.auxImg", info) ||
        !SetArg(*s, s->kWarp, 0, sizeof(pc), &pc, "warp.pc", info) ||
        !SetArg(*s, s->kWarp, 3, M, &s->flow, "warp.uFlow", info) ||
        !SetArg(*s, s->kWarp, 4, M, &s->out, "warp.outImg", info) ||
        !SetArg(*s, s->kWarp, 5, M, &s->metrics, "warp.m", info) ||
        !SetArg(*s, s->kWarp, 6, M, &s->aux, "warp.uAux", info))
        return nullptr;
    if ((e = cl->Finish(s->q)) != CL_SUCCESS) { *info = Err("clFinish(open)", e); return nullptr; }

    *info = Fmt("CL stress mode %d, out %dx%d, proc %dx%d, flow %dx%d. ", mode, outW, outH, procW, procH, fw, fh) + desc;
    guard.p = nullptr;
    return s;
}

bool clStressRun(void* h, int frames, double* out, std::string* info) {
    std::string dummy;
    if (!info) info = &dummy;
    if (!h || !out || frames <= 0) { *info = "CL stress: tham số run sai"; return false; }
    Stress& s = *static_cast<Stress*>(h);
    std::lock_guard<std::mutex> lk(s.mu);
    const ClApi* cl = s.cl;
    const size_t origin[3] = {0, 0, 0};
    const size_t regOut[3] = {static_cast<size_t>(s.outW), static_cast<size_t>(s.outH), 1};
    const size_t local[2] = {clabi::kLocalX, clabi::kLocalY};
    const size_t gPre[2] = {static_cast<size_t>(clabi::RoundUp8(s.procW)), static_cast<size_t>(clabi::RoundUp8(s.procH))};
    const size_t gPack[2] = {static_cast<size_t>(clabi::RoundUp8(s.flowW)), static_cast<size_t>(clabi::RoundUp8(s.flowH))};
    const size_t gWarp[2] = {static_cast<size_t>(clabi::RoundUp8(s.outW)), static_cast<size_t>(clabi::RoundUp8(s.outH))};
    const size_t M = sizeof(cl_mem);

    for (int f = 0; f < frames; ++f) {
        const int ci = static_cast<int>((s.frameNo + 1) & 1);  // ảnh đóng vai cur
        cl_mem prev = s.img[ci ^ 1], cur = s.img[ci];
        cl_event ev[6] = {};
        int nev = 0;
        cl_int e = CL_SUCCESS;
        auto releaseEvs = [&]() { for (int i = 0; i < nev; ++i) if (ev[i]) cl->ReleaseEvent(ev[i]); };
        auto fail = [&](const char* what) { releaseEvs(); *info = Err(Fmt("khung %d: %s", f, what), e); return false; };

        const auto t0 = Clock::now();
        if (s.mode == 0) {
            if ((e = cl->EnqueueWriteImage(s.q, cur, CL_FALSE, origin, regOut, 0, 0, s.host[ci].data(), 0, nullptr,
                                           &ev[nev++])) != CL_SUCCESS)
                return fail("clEnqueueWriteImage(cur)");
        }
        if ((e = cl->EnqueueWriteBuffer(s.q, s.nnFlow, CL_FALSE, 0, s.nnFlowHost.size() * sizeof(float),
                                        s.nnFlowHost.data(), 0, nullptr, &ev[nev++])) != CL_SUCCESS)
            return fail("clEnqueueWriteBuffer(nnFlow)");
        if ((e = cl->EnqueueWriteBuffer(s.q, s.metrics, CL_FALSE, 0, clabi::kMetricsBytes, s.zeros, 0, nullptr,
                                        &ev[nev++])) != CL_SUCCESS)
            return fail("clEnqueueWriteBuffer(metrics)");
        if ((e = cl->SetKernelArg(s.kPre, 1, M, &prev)) != CL_SUCCESS) return fail("set preprocess.uPrev");
        if ((e = cl->SetKernelArg(s.kPre, 2, M, &cur)) != CL_SUCCESS) return fail("set preprocess.uCur");
        if ((e = cl->EnqueueNDRangeKernel(s.q, s.kPre, 2, nullptr, gPre, local, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
            return fail("enqueue preprocess");
        if ((e = cl->EnqueueNDRangeKernel(s.q, s.kPack, 2, nullptr, gPack, local, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
            return fail("enqueue flowpack");
        if ((e = cl->SetKernelArg(s.kWarp, 1, M, &prev)) != CL_SUCCESS) return fail("set warp.uPrev");
        if ((e = cl->SetKernelArg(s.kWarp, 2, M, &cur)) != CL_SUCCESS) return fail("set warp.uCur");
        if ((e = cl->EnqueueNDRangeKernel(s.q, s.kWarp, 2, nullptr, gWarp, local, 0, nullptr, &ev[nev++])) != CL_SUCCESS)
            return fail("enqueue warp");
        if ((e = cl->Finish(s.q)) != CL_SUCCESS) return fail("clFinish");
        out[f] = MsSince(t0);

        double gpu = 0.0;
        bool ok = true;
        for (int i = 0; i < nev && ok; ++i) {
            cl_ulong a = 0, b = 0;
            ok = cl->GetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr) == CL_SUCCESS &&
                 cl->GetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr) == CL_SUCCESS;
            if (ok && b >= a) gpu += static_cast<double>(b - a) * 1e-6;
        }
        out[frames + f] = ok ? gpu : -1.0;
        releaseEvs();
        ++s.frameNo;
    }
    *info = Fmt("CL stress: %d khung OK (mode %d)", frames, s.mode);
    return true;
}

void clStressClose(void* h) {
    delete static_cast<Stress*>(h);
}

}  // namespace ffcl
