#pragma once
// Dữ liệu tổng hợp DÙNG CHUNG cho bench warp Vulkan và OpenCL: hai bên phải nhận đúng cùng ảnh/flow
// thì số ms mới so được (warp.comp có nhánh sớm theo dữ liệu: vùng tĩnh, HUD, occlusion).
#include <cstdint>
#include <cstring>

namespace ff::bench {

// Khung RGBA8: hoạ tiết có cạnh + chuyển động: cur = prev dịch (kDx, kDy) pixel.
constexpr int kDx = 4, kDy = 2;

inline void FillRgba8(uint8_t* dst, int w, int h, bool cur) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int sx = cur ? x - kDx : x, sy = cur ? y - kDy : y;
            uint8_t* p = dst + (static_cast<size_t>(y) * w + x) * 4;
            p[0] = static_cast<uint8_t>((sx * 7 + sy * 3) & 255);
            p[1] = static_cast<uint8_t>(((sx * 3) ^ (sy * 5)) & 255);
            p[2] = static_cast<uint8_t>(((sx + sy) * 11) & 255);
            p[3] = 255;
        }
    }
}

// float -> half (IEEE 754 binary16), đủ dùng cho giá trị nhỏ, không NaN.
inline uint16_t ToHalf(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return static_cast<uint16_t>(sign);
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m >> 13));
}

// Ảnh flow RGBA16F (flowW x flowH) giống đầu ra flowpack.comp: (F_t->0.xy, F_t->1.xy) theo UV chuẩn hoá.
// Chuyển động đều: t=0.5 -> F_t->0 = -0.5*d, F_t->1 = +0.5*d (d tính theo UV của khung ra outW x outH).
inline void FillFlowRgba16f(uint16_t* dst, int flowW, int flowH, int outW, int outH) {
    const float ux = static_cast<float>(kDx) / static_cast<float>(outW);
    const float uy = static_cast<float>(kDy) / static_cast<float>(outH);
    const uint16_t v[4] = {ToHalf(-0.5f * ux), ToHalf(-0.5f * uy), ToHalf(0.5f * ux), ToHalf(0.5f * uy)};
    for (size_t i = 0, n = static_cast<size_t>(flowW) * flowH; i < n; ++i) std::memcpy(dst + i * 4, v, 8);
}

// Ảnh aux RGBA16F: (mask = 0.5, 0, 0, 1) giống flowpack.comp.
inline void FillAuxRgba16f(uint16_t* dst, int flowW, int flowH) {
    const uint16_t v[4] = {ToHalf(0.5f), 0, 0, ToHalf(1.0f)};
    for (size_t i = 0, n = static_cast<size_t>(flowW) * flowH; i < n; ++i) std::memcpy(dst + i * 4, v, 8);
}

// Push constant cho bench (giống đường NN thật trong GpuAsync): procW = flowW*2.
struct WarpBenchParams { int outW, outH, procW, procH; };
// Buffer metrics: toàn 0 (không chuyển cảnh). Số phần tử uint tối thiểu: 64.
constexpr int kMetricsUints = 64;

}  // namespace ff::bench
