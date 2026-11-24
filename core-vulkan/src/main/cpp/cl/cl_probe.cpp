// Báo cáo khả năng OpenCL của máy: platform, GPU device, các extension interop quan trọng.
#include <sstream>
#include <string>
#include <vector>

#include "cl_loader.h"

namespace ffcl {
namespace {

std::string PlatStr(const ClApi* cl, cl_platform_id p, cl_platform_info what) {
    size_t n = 0;
    if (cl->GetPlatformInfo(p, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "?";
    std::string s(n, '\0');
    if (cl->GetPlatformInfo(p, what, n, &s[0], nullptr) != CL_SUCCESS) return "?";
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

std::string DevStr(const ClApi* cl, cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (cl->GetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "?";
    std::string s(n, '\0');
    if (cl->GetDeviceInfo(d, what, n, &s[0], nullptr) != CL_SUCCESS) return "?";
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
T DevVal(const ClApi* cl, cl_device_id d, cl_device_info what) {
    T v{};
    if (cl->GetDeviceInfo(d, what, sizeof(T), &v, nullptr) != CL_SUCCESS) return T{};
    return v;
}

bool HasExt(const std::string& exts, const char* name) {
    // So khớp nguyên token (tránh cl_khr_external_memory khớp nhầm cl_khr_external_memory_dma_buf).
    const std::string n(name);
    size_t pos = 0;
    while ((pos = exts.find(n, pos)) != std::string::npos) {
        const bool startOk = pos == 0 || exts[pos - 1] == ' ';
        const size_t end = pos + n.size();
        const bool endOk = end == exts.size() || exts[end] == ' ';
        if (startOk && endOk) return true;
        pos = end;
    }
    return false;
}

const char* kFlags[] = {
    "cl_khr_external_memory_android_hardware_buffer",
    "cl_khr_external_memory_dma_buf",
    "cl_khr_external_memory",
    "cl_khr_external_semaphore",
    "cl_arm_import_memory",
    "cl_arm_import_memory_android_hardware_buffer",
    "cl_khr_egl_image",
    "cl_khr_fp16",
    "cl_khr_subgroups",
};

}  // namespace

std::string clProbeReport() {
    std::ostringstream o;
    const ClLoadStatus& st = ClStatus();
    const ClApi* cl = ClLoad();
    if (!cl) {
        o << "OpenCL: KHÔNG nạp được\n";
        if (!st.path.empty()) o << "đã mở: " << st.path << "\n";
        o << st.error;
        return o.str();
    }
    o << "OpenCL: nạp OK từ " << st.path << "\n";
    o << "clGetExtensionFunctionAddressForPlatform: "
      << (cl->GetExtensionFunctionAddressForPlatform ? "có" : "không") << "\n";

    cl_uint np = 0;
    cl_int e = cl->GetPlatformIDs(0, nullptr, &np);
    if (e != CL_SUCCESS || np == 0) {
        o << "Không có platform: " << ClErrStr(e) << "\n";
        return o.str();
    }
    std::vector<cl_platform_id> plats(np);
    cl->GetPlatformIDs(np, plats.data(), nullptr);

    for (cl_uint i = 0; i < np; ++i) {
        cl_platform_id p = plats[i];
        const std::string pext = PlatStr(cl, p, CL_PLATFORM_EXTENSIONS);
        o << "\n== Platform " << i << ": " << PlatStr(cl, p, CL_PLATFORM_NAME) << "\n"
          << "Hãng: " << PlatStr(cl, p, CL_PLATFORM_VENDOR) << "\n"
          << "Phiên bản: " << PlatStr(cl, p, CL_PLATFORM_VERSION) << "\n"
          << "Profile: " << PlatStr(cl, p, CL_PLATFORM_PROFILE) << "\n"
          << "Ext platform: " << pext << "\n";

        cl_uint nd = 0;
        e = cl->GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
        if (e != CL_SUCCESS || nd == 0) {
            o << "Không có GPU device: " << ClErrStr(e) << "\n";
            continue;
        }
        std::vector<cl_device_id> devs(nd);
        cl->GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (cl_uint j = 0; j < nd; ++j) {
            cl_device_id d = devs[j];
            const std::string dext = DevStr(cl, d, CL_DEVICE_EXTENSIONS);
            const std::string all = dext + " " + pext;
            o << "-- GPU " << j << ": " << DevStr(cl, d, CL_DEVICE_NAME) << "\n"
              << "Hãng: " << DevStr(cl, d, CL_DEVICE_VENDOR) << "\n"
              << "Driver: " << DevStr(cl, d, CL_DRIVER_VERSION) << "\n"
              << "Phiên bản: " << DevStr(cl, d, CL_DEVICE_VERSION) << "\n"
              << "OpenCL C: " << DevStr(cl, d, CL_DEVICE_OPENCL_C_VERSION) << "\n"
              << "Compute units: " << DevVal<cl_uint>(cl, d, CL_DEVICE_MAX_COMPUTE_UNITS) << "\n"
              << "Xung max: " << DevVal<cl_uint>(cl, d, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz\n"
              << "Global mem: " << (DevVal<cl_ulong>(cl, d, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20) << " MB\n"
              << "Max alloc: " << (DevVal<cl_ulong>(cl, d, CL_DEVICE_MAX_MEM_ALLOC_SIZE) >> 20) << " MB\n"
              << "Local mem: " << (DevVal<cl_ulong>(cl, d, CL_DEVICE_LOCAL_MEM_SIZE) >> 10) << " KB\n"
              << "Max work-group: " << DevVal<size_t>(cl, d, CL_DEVICE_MAX_WORK_GROUP_SIZE) << "\n"
              << "Timer res: " << DevVal<size_t>(cl, d, CL_DEVICE_PROFILING_TIMER_RESOLUTION) << " ns\n"
              << "Image: " << (DevVal<cl_bool>(cl, d, CL_DEVICE_IMAGE_SUPPORT) ? "có" : "không")
              << ", image2d max " << DevVal<size_t>(cl, d, CL_DEVICE_IMAGE2D_MAX_WIDTH) << "x"
              << DevVal<size_t>(cl, d, CL_DEVICE_IMAGE2D_MAX_HEIGHT) << "\n";
            o << "Ext quan trọng:\n";
            for (const char* f : kFlags) o << "  [" << (HasExt(all, f) ? "x" : " ") << "] " << f << "\n";
            o << "Ext device: " << dext << "\n";
        }
    }
    return o.str();
}

}  // namespace ffcl
