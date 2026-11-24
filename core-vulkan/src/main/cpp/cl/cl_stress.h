#pragma once
// Chuỗi khung OpenCL "stress" chạy liên tục, mô phỏng đường NN Vulkan (NnPre + NnPost) để A/B nhiều phút.
//   mode 0 = "copy":   mỗi khung clEnqueueWriteImage khung hiện tại RGBA8 (CL không interop).
//   mode 1 = "dmabuf": zero-copy, 2 AHardwareBuffer -> dma-buf fd -> cl_arm_import_memory
//                      (dự phòng cl_khr_external_memory_dma_buf) -> image2d_from_buffer; mỗi khung chỉ đổi prev/cur.
// Mỗi khung: [mode0: ghi ảnh cur] + ghi nnFlow (5*flowW*flowH float) + reset metrics 64 B
//            + preprocess + flowpack + warp(mode=1) + clFinish.
#include <string>

namespace ffcl {

// Trả handle hoặc nullptr; info = mô tả ngắn cấu hình (đường import, stride) hoặc lỗi chi tiết.
void* clStressOpen(const std::string& src, int outW, int outH, int procW, int procH, int mode, std::string* info);

// Chạy `frames` khung liên tiếp. out phải có 2*frames phần tử:
//   out[i]          = wall ms khung i (bắt đầu enqueue -> clFinish xong)
//   out[frames + i] = GPU ms khung i = TỔNG (END - START) profiling của mọi lệnh trong khung
//                     (ghi buffer/ảnh + 3 kernel; không tính khoảng trống giữa các lệnh). -1 nếu không đọc được.
// Trả false + info khi lỗi.
bool clStressRun(void* h, int frames, double* out, std::string* info);

void clStressClose(void* h);

}  // namespace ffcl
