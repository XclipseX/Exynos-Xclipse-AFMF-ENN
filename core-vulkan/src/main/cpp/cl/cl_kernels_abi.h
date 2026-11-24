// cl_kernels_abi.h — ABI giữa host C++ và assets/cl/frame_kernels.cl (OpenCL C 1.2).
// Không include CL/cl.h: hằng số định dạng ảnh bên dưới là bản sao giá trị trong CL/cl.h
// (host có thể static_assert(kClRGBA == CL_RGBA) ... ở file có include CL).
//
// Mọi kernel: local size 8x8x1 (reqd_work_group_size), global size = làm tròn lên bội 8 của
// số work-item Vulkan dispatch (CeilDiv(n, 8) * 8). Tham số 0 luôn là FramePC truyền theo giá trị
// (clSetKernelArg(k, 0, sizeof(FramePC), &pc)), các giá trị điền giống hệt PushConstants trong pipeline.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace clabi {

// ---------------------------------------------------------------------------------------------
// Push constants (layout std430 push_constant trong GLSL: 16 trường 4 byte, offset 0..60, tổng 64 byte)
// ---------------------------------------------------------------------------------------------
struct FramePC {
    int32_t srcW, srcH;
    int32_t lowW, lowH;
    int32_t flowW, flowH;
    int32_t outW, outH;
    float t;
    int32_t mode;    // 0 = chép frame hiện tại, 1 = nội suy (chỉ warp đọc)
    int32_t useNN;   // 1 = flow/mask từ NN, 0 = flow block-match + mask 0.5
    int32_t pad0;    // bit 0: tắt bảo vệ HUD (tuneFlags_)
    float flowScaleX, flowScaleY;  // 1/procW, 1/procH
    float staticThr;
    float occThr;
};
using PreprocessPC = FramePC;
using BlockmatchPC = FramePC;
using FlowpackPC = FramePC;
using WarpPC = FramePC;

static_assert(sizeof(FramePC) == 64, "FramePC phải 64 byte như push constant GLSL");
static_assert(alignof(FramePC) == 4, "FramePC chỉ gồm trường 4 byte");
static_assert(offsetof(FramePC, srcW) == 0, "layout");
static_assert(offsetof(FramePC, srcH) == 4, "layout");
static_assert(offsetof(FramePC, lowW) == 8, "layout");
static_assert(offsetof(FramePC, lowH) == 12, "layout");
static_assert(offsetof(FramePC, flowW) == 16, "layout");
static_assert(offsetof(FramePC, flowH) == 20, "layout");
static_assert(offsetof(FramePC, outW) == 24, "layout");
static_assert(offsetof(FramePC, outH) == 28, "layout");
static_assert(offsetof(FramePC, t) == 32, "layout");
static_assert(offsetof(FramePC, mode) == 36, "layout");
static_assert(offsetof(FramePC, useNN) == 40, "layout");
static_assert(offsetof(FramePC, pad0) == 44, "layout");
static_assert(offsetof(FramePC, flowScaleX) == 48, "layout");
static_assert(offsetof(FramePC, flowScaleY) == 52, "layout");
static_assert(offsetof(FramePC, staticThr) == 56, "layout");
static_assert(offsetof(FramePC, occThr) == 60, "layout");

// ---------------------------------------------------------------------------------------------
// Hằng số (giá trị trùng CL/cl.h)
// ---------------------------------------------------------------------------------------------
constexpr uint32_t kClNone = 0;                 // không phải ảnh
constexpr uint32_t kClRGBA = 0x10B5;            // CL_RGBA
constexpr uint32_t kClBGRA = 0x10B6;            // CL_BGRA (chấp nhận cho uPrev/uCur nếu AHB là BGRA)
constexpr uint32_t kClUnormInt8 = 0x10D2;       // CL_UNORM_INT8  <-> VK_FORMAT_R8G8B8A8_UNORM
constexpr uint32_t kClHalfFloat = 0x10DD;       // CL_HALF_FLOAT  <-> VK_FORMAT_R16G16B16A16_SFLOAT

constexpr int kLocalX = 8, kLocalY = 8;
constexpr int kBlock = 8;      // BM_BLOCK trong kernel, khớp kBlock pipeline.cpp
constexpr int kMetricsBytes = 64;  // 16 x uint32; host memset 0 mỗi khung trước preprocess
// metrics: [0] tổng |luma diff| (preprocess), [1] block chạm biên, [2] tổng block, [3] tổng SAD (blockmatch)

constexpr int RoundUp8(int n) { return (n + 7) / 8 * 8; }

enum class ArgKind : uint8_t { PushConst, ImageRead, ImageWrite, BufferRW, BufferRO };

struct KernelArg {
    int index;               // chỉ số clSetKernelArg
    const char* name;        // tên tham số trong .cl
    const char* clType;      // kiểu OpenCL C
    ArgKind kind;
    uint32_t channelOrder;   // kClRGBA... hoặc kClNone
    uint32_t channelType;    // kClUnormInt8 / kClHalfFloat hoặc kClNone
    int vkBinding;           // binding Vulkan tương ứng (-1 = push constant)
    const char* meaning;
};

// ---------------------------------------------------------------------------------------------
// preprocess — global = RoundUp8(lowW) x RoundUp8(lowH)   (Vulkan: CeilDiv(procW,8) x CeilDiv(procH,8) nhóm)
// ---------------------------------------------------------------------------------------------
constexpr const char* kPreprocessName = "preprocess";
constexpr KernelArg kPreprocessArgs[] = {
    {0, "pc", "PreprocessPC", ArgKind::PushConst, kClNone, kClNone, -1, "push constants (by value, 64 B)"},
    {1, "uPrev", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 0,
     "frame trước, kích thước srcW x srcH, lấy mẫu linear/clamp-to-edge, toạ độ chuẩn hoá"},
    {2, "uCur", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 1,
     "frame hiện tại, srcW x srcH, linear/clamp-to-edge"},
    {3, "lowPrev", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClUnormInt8, 2,
     "prev thu nhỏ lowW x lowH (rgba8)"},
    {4, "lowCur", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClUnormInt8, 3,
     "cur thu nhỏ lowW x lowH (rgba8)"},
    {5, "nnIn", "__global float*", ArgKind::BufferRW, kClNone, kClNone, 4,
     "tensor NN NCHW [6, lowH, lowW] float; chỉ ghi khi useNN != 0 (được phép NULL khi useNN == 0)"},
    {6, "m", "__global uint*", ArgKind::BufferRW, kClNone, kClNone, 9, "metrics (atomic_add m[0])"},
};

// ---------------------------------------------------------------------------------------------
// blockmatch — global = RoundUp8(ceil(lowW/8)) x RoundUp8(ceil(lowH/8))
// ---------------------------------------------------------------------------------------------
constexpr const char* kBlockmatchName = "blockmatch";
constexpr KernelArg kBlockmatchArgs[] = {
    {0, "pc", "BlockmatchPC", ArgKind::PushConst, kClNone, kClNone, -1, "push constants (by value, 64 B)"},
    {1, "lowPrev", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 2,
     "prev proc-res; đọc bằng read_imagef(int2) không sampler (= imageLoad)"},
    {2, "lowCur", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 3,
     "cur proc-res; read_imagef(int2)"},
    {3, "flowImg", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClHalfFloat, 6,
     "flow flowW x flowH: xy = F_t->0, zw = F_t->1 (UV chuẩn hoá)"},
    {4, "m", "__global uint*", ArgKind::BufferRW, kClNone, kClNone, 9, "metrics (atomic_add m[1], m[2], m[3])"},
};

// ---------------------------------------------------------------------------------------------
// flowpack — global = RoundUp8(flowW) x RoundUp8(flowH)
// ---------------------------------------------------------------------------------------------
constexpr const char* kFlowpackName = "flowpack";
constexpr KernelArg kFlowpackArgs[] = {
    {0, "pc", "FlowpackPC", ArgKind::PushConst, kClNone, kClNone, -1, "push constants (by value, 64 B)"},
    {1, "nnFlow", "__global const float*", ArgKind::BufferRO, kClNone, kClNone, 5,
     "đầu ra NN NCHW [5, flowH, flowW] (pixel proc-res)"},
    {2, "flowImg", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClHalfFloat, 6, "flow (rgba16f)"},
    {3, "auxImg", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClHalfFloat, 10,
     "mask NN ở kênh r (rgba16f)"},
};

// ---------------------------------------------------------------------------------------------
// warp — global = RoundUp8(outW) x RoundUp8(outH). Chạy 2 lần mỗi khung: mode=1 (nội suy), mode=0 (khung thật).
// ---------------------------------------------------------------------------------------------
constexpr const char* kWarpName = "warp";
constexpr KernelArg kWarpArgs[] = {
    {0, "pc", "WarpPC", ArgKind::PushConst, kClNone, kClNone, -1, "push constants (by value, 64 B)"},
    {1, "uPrev", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 0,
     "frame trước, linear/clamp-to-edge"},
    {2, "uCur", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClUnormInt8, 1,
     "frame hiện tại, linear/clamp-to-edge"},
    {3, "uFlow", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClHalfFloat, 7,
     "flow (cùng ảnh với flowImg binding 6), linear/clamp-to-edge"},
    {4, "outImg", "write_only image2d_t", ArgKind::ImageWrite, kClRGBA, kClUnormInt8, 8, "ảnh ra outW x outH"},
    {5, "m", "__global const uint*", ArgKind::BufferRO, kClNone, kClNone, 9, "metrics (đọc m[0..2])"},
    {6, "uAux", "read_only image2d_t", ArgKind::ImageRead, kClRGBA, kClHalfFloat, 11,
     "mask NN (cùng ảnh auxImg binding 10); chỉ đọc khi useNN != 0 nhưng phải luôn set"},
};

constexpr int kPreprocessArgCount = static_cast<int>(sizeof(kPreprocessArgs) / sizeof(kPreprocessArgs[0]));
constexpr int kBlockmatchArgCount = static_cast<int>(sizeof(kBlockmatchArgs) / sizeof(kBlockmatchArgs[0]));
constexpr int kFlowpackArgCount = static_cast<int>(sizeof(kFlowpackArgs) / sizeof(kFlowpackArgs[0]));
constexpr int kWarpArgCount = static_cast<int>(sizeof(kWarpArgs) / sizeof(kWarpArgs[0]));
static_assert(kPreprocessArgCount == 7, "preprocess có 7 tham số");
static_assert(kBlockmatchArgCount == 5, "blockmatch có 5 tham số");
static_assert(kFlowpackArgCount == 4, "flowpack có 4 tham số");
static_assert(kWarpArgCount == 7, "warp có 7 tham số");

}  // namespace clabi
