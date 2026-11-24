#pragma once
// ClEngine — compute OpenCL LIVE cho đường khung NPU (thay NnPost Vulkan: flowpack + warp mode=1).
// Vulkan sở hữu toàn bộ bộ nhớ khung: mỗi slot async (0..kMaxSlots-1) Vulkan xuất 3 dma-buf fd của
// VkBuffer tuyến tính RGBA8 (prev, cur, out; row pitch = strideBytes). Vulkan ghi prev/cur và chờ fence
// trên CPU trước khi gọi Run; sau khi Run trả về Vulkan chép buffer out vào ảnh của nó và present.
// Không khoá nội bộ: bên gọi tuần tự hoá mọi phương thức bằng mutex riêng.
#include <cstddef>
#include <memory>
#include <string>

#include "cl_loader.h"

namespace ffcl {

class ClEngine {
public:
    static constexpr int kMaxSlots = 3;

    // Build frame_kernels.cl (-cl-std=CL1.2, không fast-math), context + queue (profiling) trên GPU đầu tiên,
    // kernel flowpack + warp, resolve clImportMemoryARM (dự phòng clCreateBufferWithProperties DMA_BUF_KHR).
    // nullptr + *info khi lỗi (kèm build log đã cắt).
    static std::unique_ptr<ClEngine> Create(const std::string& clSource, std::string* info);
    ~ClEngine();

    ClEngine(const ClEngine&) = delete;
    ClEngine& operator=(const ClEngine&) = delete;

    // (Tái) cấu hình một slot. fd vẫn thuộc bên gọi (engine dup() riêng, đóng bản dup khi release).
    // bufBytes = kích thước mỗi dma-buf. prev/cur: image2d_from_buffer READ_ONLY, out: WRITE_ONLY,
    // CL_RGBA/CL_UNORM_INT8, w x h, row pitch strideBytes. Flow/aux RGBA16F procW/2 x procH/2,
    // nnFlow (5*flowW*flowH float), metrics 64 B — riêng từng slot. Giải phóng đối tượng cũ của slot trước.
    bool SetupSlot(int slot, int prevFd, int curFd, int outFd, size_t bufBytes, int w, int h, int strideBytes,
                   int procW, int procH, std::string* err);

    // Một khung nội suy: ghi nnFlow + metrics (64 B), flowpack rồi warp (pc64 = PushConstants Vulkan 64 B,
    // mode=1, chuyển nguyên), clFinish. gpuMs = END(warp) - START(lệnh đầu tiên) theo profiling, -1 nếu không có.
    bool Run(int slot, const void* pc64, const float* nnFlow, size_t nnFlowBytes, const void* metrics64,
             double* gpuMs, std::string* err);

    void ReleaseSlots();
    std::string Describe() const;  // platform/device/driver + đường import, một dòng

private:
    struct Slot;
    struct Impl;
    ClEngine();
    std::unique_ptr<Impl> d_;
};

}  // namespace ffcl
