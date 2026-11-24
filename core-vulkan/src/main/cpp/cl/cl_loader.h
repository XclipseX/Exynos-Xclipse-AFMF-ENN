#pragma once
// Loader OpenCL 1.2 tối giản: không có header Khronos trong NDK nên tự khai báo kiểu/hằng
// (giá trị đúng theo CL/cl.h của Khronos), dlopen libOpenCL.so của vendor lúc chạy.
#include <cstddef>
#include <cstdint>
#include <string>

#ifndef __OPENCL_CL_H
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_bitfield cl_command_queue_properties;
typedef intptr_t cl_context_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_uint cl_mem_object_type;
typedef cl_uint cl_program_build_info;
typedef cl_int cl_build_status;
typedef cl_uint cl_profiling_info;
typedef cl_uint cl_channel_order;
typedef cl_uint cl_channel_type;

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

typedef struct _cl_image_format {
    cl_channel_order image_channel_order;
    cl_channel_type image_channel_data_type;
} cl_image_format;

typedef struct _cl_image_desc {
    cl_mem_object_type image_type;
    size_t image_width;
    size_t image_height;
    size_t image_depth;
    size_t image_array_size;
    size_t image_row_pitch;
    size_t image_slice_pitch;
    cl_uint num_mip_levels;
    cl_uint num_samples;
    cl_mem buffer;  // union { buffer, mem_object } ở CL 2.0, cùng layout
} cl_image_desc;

#define CL_SUCCESS 0
#define CL_DEVICE_NOT_FOUND -1
#define CL_DEVICE_NOT_AVAILABLE -2
#define CL_COMPILER_NOT_AVAILABLE -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE -4
#define CL_OUT_OF_RESOURCES -5
#define CL_OUT_OF_HOST_MEMORY -6
#define CL_PROFILING_INFO_NOT_AVAILABLE -7
#define CL_IMAGE_FORMAT_NOT_SUPPORTED -10
#define CL_BUILD_PROGRAM_FAILURE -11
#define CL_INVALID_VALUE -30
#define CL_INVALID_PLATFORM -32
#define CL_INVALID_DEVICE -33
#define CL_INVALID_KERNEL_NAME -46
#define CL_INVALID_ARG_INDEX -49
#define CL_INVALID_ARG_VALUE -50
#define CL_INVALID_ARG_SIZE -51
#define CL_INVALID_KERNEL_ARGS -52
#define CL_INVALID_WORK_GROUP_SIZE -54
#define CL_INVALID_IMAGE_DESCRIPTOR -65
#define CL_PLATFORM_NOT_FOUND_KHR -1001

#define CL_FALSE 0
#define CL_TRUE 1

#define CL_PLATFORM_PROFILE 0x0900
#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VENDOR 0x0903
#define CL_PLATFORM_EXTENSIONS 0x0904

#define CL_DEVICE_TYPE_DEFAULT (1 << 0)
#define CL_DEVICE_TYPE_CPU (1 << 1)
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR (1 << 3)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF

#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_VENDOR_ID 0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS 0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES 0x1005
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_IMAGE2D_MAX_WIDTH 0x1011
#define CL_DEVICE_IMAGE2D_MAX_HEIGHT 0x1012
#define CL_DEVICE_IMAGE_SUPPORT 0x1016
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_PROFILING_TIMER_RESOLUTION 0x1025
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_PROFILE 0x102E
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_OPENCL_C_VERSION 0x103D

#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)

#define CL_CONTEXT_PLATFORM 0x1084

#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

#define CL_R 0x10B0
#define CL_RG 0x10B2
#define CL_RGBA 0x10B5
#define CL_BGRA 0x10B6

#define CL_UNORM_INT8 0x10D2
#define CL_UNSIGNED_INT32 0x10DC
#define CL_HALF_FLOAT 0x10DD
#define CL_FLOAT 0x10DE

#define CL_MEM_OBJECT_BUFFER 0x10F0
#define CL_MEM_OBJECT_IMAGE2D 0x10F1

#define CL_PROGRAM_BUILD_STATUS 0x1181
#define CL_PROGRAM_BUILD_OPTIONS 0x1182
#define CL_PROGRAM_BUILD_LOG 0x1183

#define CL_PROFILING_COMMAND_QUEUED 0x1280
#define CL_PROFILING_COMMAND_SUBMIT 0x1281
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#endif  // __OPENCL_CL_H

namespace ffcl {

struct ClApi {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (*GetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                           void (*)(cl_program, void*), void*);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_mem (*CreateImage)(cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                   const size_t*, cl_uint, const cl_event*, cl_event*);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*,
                                 cl_uint, const cl_event*, cl_event*);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                                cl_uint, const cl_event*, cl_event*);
    cl_int (*EnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*,
                                size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    cl_int (*EnqueueReadImage)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*,
                               size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*WaitForEvents)(cl_uint, const cl_event*);
    cl_int (*GetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void*, size_t*);
    cl_int (*ReleaseEvent)(cl_event);
    cl_int (*ReleaseMemObject)(cl_mem);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_int (*ReleaseProgram)(cl_program);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_int (*ReleaseContext)(cl_context);
    // Tùy chọn (có thể null).
    void* (*GetExtensionFunctionAddressForPlatform)(cl_platform_id, const char*);
};

struct ClLoadStatus {
    bool ok = false;
    std::string path;      // đường dẫn đã dlopen được
    std::string error;     // dlerror từng đường dẫn / symbol thiếu
    std::string missing;   // danh sách symbol bắt buộc bị thiếu
};

// Nạp một lần (thread-safe). Trả nullptr nếu không nạp được hoặc thiếu symbol bắt buộc.
const ClApi* ClLoad();
const ClLoadStatus& ClStatus();

// Tên lỗi CL dễ đọc (ví dụ "CL_BUILD_PROGRAM_FAILURE(-11)").
std::string ClErrStr(cl_int e);

}  // namespace ffcl
