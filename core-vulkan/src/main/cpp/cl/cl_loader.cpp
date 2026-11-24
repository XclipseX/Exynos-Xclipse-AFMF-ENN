#include "cl_loader.h"

#include <dlfcn.h>

#include <mutex>

namespace ffcl {
namespace {

ClApi g_api{};
ClLoadStatus g_status;
std::once_flag g_once;
void* g_lib = nullptr;

template <typename T>
bool Sym(void* lib, const char* name, T& out, bool required) {
    out = reinterpret_cast<T>(dlsym(lib, name));
    if (!out && required) {
        if (!g_status.missing.empty()) g_status.missing += ", ";
        g_status.missing += name;
    }
    return out != nullptr;
}

void DoLoad() {
    static const char* kPaths[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
    };
    for (const char* p : kPaths) {
        dlerror();
        void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            g_lib = h;
            g_status.path = p;
            break;
        }
        const char* e = dlerror();
        g_status.error += std::string(p) + ": " + (e ? e : "?") + "\n";
    }
    if (!g_lib) return;

    ClApi& a = g_api;
    Sym(g_lib, "clGetPlatformIDs", a.GetPlatformIDs, true);
    Sym(g_lib, "clGetPlatformInfo", a.GetPlatformInfo, true);
    Sym(g_lib, "clGetDeviceIDs", a.GetDeviceIDs, true);
    Sym(g_lib, "clGetDeviceInfo", a.GetDeviceInfo, true);
    Sym(g_lib, "clCreateContext", a.CreateContext, true);
    Sym(g_lib, "clCreateCommandQueue", a.CreateCommandQueue, true);
    Sym(g_lib, "clCreateProgramWithSource", a.CreateProgramWithSource, true);
    Sym(g_lib, "clBuildProgram", a.BuildProgram, true);
    Sym(g_lib, "clGetProgramBuildInfo", a.GetProgramBuildInfo, true);
    Sym(g_lib, "clCreateKernel", a.CreateKernel, true);
    Sym(g_lib, "clSetKernelArg", a.SetKernelArg, true);
    Sym(g_lib, "clCreateBuffer", a.CreateBuffer, true);
    Sym(g_lib, "clCreateImage", a.CreateImage, true);
    Sym(g_lib, "clEnqueueNDRangeKernel", a.EnqueueNDRangeKernel, true);
    Sym(g_lib, "clEnqueueWriteBuffer", a.EnqueueWriteBuffer, true);
    Sym(g_lib, "clEnqueueReadBuffer", a.EnqueueReadBuffer, true);
    Sym(g_lib, "clEnqueueWriteImage", a.EnqueueWriteImage, true);
    Sym(g_lib, "clEnqueueReadImage", a.EnqueueReadImage, true);
    Sym(g_lib, "clFinish", a.Finish, true);
    Sym(g_lib, "clWaitForEvents", a.WaitForEvents, true);
    Sym(g_lib, "clGetEventProfilingInfo", a.GetEventProfilingInfo, true);
    Sym(g_lib, "clReleaseEvent", a.ReleaseEvent, true);
    Sym(g_lib, "clReleaseMemObject", a.ReleaseMemObject, true);
    Sym(g_lib, "clReleaseKernel", a.ReleaseKernel, true);
    Sym(g_lib, "clReleaseProgram", a.ReleaseProgram, true);
    Sym(g_lib, "clReleaseCommandQueue", a.ReleaseCommandQueue, true);
    Sym(g_lib, "clReleaseContext", a.ReleaseContext, true);
    Sym(g_lib, "clGetExtensionFunctionAddressForPlatform", a.GetExtensionFunctionAddressForPlatform, false);

    g_status.ok = g_status.missing.empty();
    if (!g_status.ok) g_status.error += "thiếu symbol: " + g_status.missing + "\n";
}

}  // namespace

const ClApi* ClLoad() {
    std::call_once(g_once, DoLoad);
    return g_status.ok ? &g_api : nullptr;
}

const ClLoadStatus& ClStatus() {
    std::call_once(g_once, DoLoad);
    return g_status;
}

std::string ClErrStr(cl_int e) {
    const char* n = nullptr;
    switch (e) {
        case CL_SUCCESS: n = "CL_SUCCESS"; break;
        case CL_DEVICE_NOT_FOUND: n = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE: n = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE: n = "CL_COMPILER_NOT_AVAILABLE"; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: n = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
        case CL_OUT_OF_RESOURCES: n = "CL_OUT_OF_RESOURCES"; break;
        case CL_OUT_OF_HOST_MEMORY: n = "CL_OUT_OF_HOST_MEMORY"; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE: n = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: n = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
        case CL_BUILD_PROGRAM_FAILURE: n = "CL_BUILD_PROGRAM_FAILURE"; break;
        case CL_INVALID_VALUE: n = "CL_INVALID_VALUE"; break;
        case CL_INVALID_PLATFORM: n = "CL_INVALID_PLATFORM"; break;
        case CL_INVALID_DEVICE: n = "CL_INVALID_DEVICE"; break;
        case CL_INVALID_KERNEL_NAME: n = "CL_INVALID_KERNEL_NAME"; break;
        case CL_INVALID_ARG_INDEX: n = "CL_INVALID_ARG_INDEX"; break;
        case CL_INVALID_ARG_VALUE: n = "CL_INVALID_ARG_VALUE"; break;
        case CL_INVALID_ARG_SIZE: n = "CL_INVALID_ARG_SIZE"; break;
        case CL_INVALID_KERNEL_ARGS: n = "CL_INVALID_KERNEL_ARGS"; break;
        case CL_INVALID_WORK_GROUP_SIZE: n = "CL_INVALID_WORK_GROUP_SIZE"; break;
        case CL_INVALID_IMAGE_DESCRIPTOR: n = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
        case CL_PLATFORM_NOT_FOUND_KHR: n = "CL_PLATFORM_NOT_FOUND_KHR"; break;
        default: break;
    }
    return std::string(n ? n : "CL_ERR") + "(" + std::to_string(e) + ")";
}

}  // namespace ffcl
