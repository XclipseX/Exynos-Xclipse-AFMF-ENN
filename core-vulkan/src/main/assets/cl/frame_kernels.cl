// frame_kernels.cl — bản OpenCL C 1.2 của preprocess.comp / blockmatch.comp / flowpack.comp / warp.comp.
// Giữ nguyên toán, ngưỡng, thứ tự phép tính và cách xử lý biên của bản GLSL.
// ABI (thứ tự tham số, định dạng ảnh, binding Vulkan tương ứng): cpp/cl/cl_kernels_abi.h.
//
// Quy ước:
//  - Local size 8x8 (giống layout(local_size_x = 8, local_size_y = 8)); host làm tròn global size lên bội 8.
//  - sampler2D (VK_FILTER_LINEAR, CLAMP_TO_EDGE, maxLod 0) -> read_imagef + kLinear, toạ độ chuẩn hoá.
//  - imageLoad(ivec2)  -> read_imagef(image, int2) (không sampler, CL 1.2) — toạ độ đã clamp sẵn trong code.
//  - imageStore rgba8 / rgba16f -> write_imagef.
//  - mix/smoothstep viết tay theo đúng công thức GLSL (CL mix = x + (y-x)*a khác thứ tự làm tròn).
//  - Không build với -cl-fast-relaxed-math / -cl-mad-enable nếu cần khớp số học với bản Vulkan.

// Push constant: 16 trường 4 byte = 64 byte, khớp PushConstants trong pipeline.h.
typedef struct {
    int srcW, srcH, lowW, lowH, flowW, flowH, outW, outH;
    float t; int mode; int useNN; int pad0;
    float flowScaleX, flowScaleY; float staticThr; float occThr;
} FramePC;

typedef FramePC PreprocessPC;
typedef FramePC BlockmatchPC;
typedef FramePC FlowpackPC;
typedef FramePC WarpPC;

__constant sampler_t kLinear = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// GLSL: mix(x, y, a) = x * (1 - a) + y * a
inline float3 glsl_mix3(float3 x, float3 y, float a) { return x * (1.0f - a) + y * a; }

// GLSL: smoothstep(e0, e1, x): t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2 t)
inline float glsl_smoothstep(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float maxAbs3(float3 v) { return max(fabs(v.x), max(fabs(v.y), fabs(v.z))); }

inline float3 tex3(read_only image2d_t s, float2 uv) { return read_imagef(s, kLinear, uv).xyz; }

// =============================================================================================
// preprocess — một work-item = một pixel proc-res (lowW x lowH). Global = ceil8(lowW) x ceil8(lowH).
// =============================================================================================

inline float3 box(read_only image2d_t s, float2 uv, FramePC pc) {
    float2 d = 0.25f * (float2)((float)pc.srcW / (float)pc.lowW, (float)pc.srcH / (float)pc.lowH)
                     / (float2)((float)pc.srcW, (float)pc.srcH);
    return 0.25f * (tex3(s, uv + (float2)(-d.x, -d.y)) + tex3(s, uv + (float2)(d.x, -d.y)) +
                    tex3(s, uv + (float2)(-d.x, d.y)) + tex3(s, uv + (float2)(d.x, d.y)));
}

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void preprocess(PreprocessPC pc,
                read_only image2d_t uPrev,      // binding 0
                read_only image2d_t uCur,       // binding 1
                write_only image2d_t lowPrev,   // binding 2 (rgba8)
                write_only image2d_t lowCur,    // binding 3 (rgba8)
                __global float* nnIn,           // binding 4 (chỉ dùng khi useNN != 0, có thể NULL khi useNN == 0)
                __global uint* m)               // binding 9
{
    __local uint sDiff;
    const bool lid0 = (get_local_id(0) == 0 && get_local_id(1) == 0);
    if (lid0) sDiff = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 p = (int2)((int)get_global_id(0), (int)get_global_id(1));
    uint d8 = 0u;
    if (p.x < pc.lowW && p.y < pc.lowH) {
        float2 uv = ((float2)((float)p.x, (float)p.y) + 0.5f) / (float2)((float)pc.lowW, (float)pc.lowH);
        float3 a = box(uPrev, uv, pc);
        float3 b = box(uCur, uv, pc);
        write_imagef(lowPrev, p, (float4)(a, 1.0f));
        write_imagef(lowCur, p, (float4)(b, 1.0f));

        if (pc.useNN != 0) {
            int plane = pc.lowW * pc.lowH;
            int idx = p.y * pc.lowW + p.x;
            nnIn[0 * plane + idx] = a.x;
            nnIn[1 * plane + idx] = a.y;
            nnIn[2 * plane + idx] = a.z;
            nnIn[3 * plane + idx] = b.x;
            nnIn[4 * plane + idx] = b.y;
            nnIn[5 * plane + idx] = b.z;
        }
        const float3 W = (float3)(0.299f, 0.587f, 0.114f);
        d8 = (uint)(clamp(fabs(dot(a, W) - dot(b, W)), 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    atomic_add(&sDiff, d8);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0 && sDiff != 0u) atomic_add(&m[0], sDiff);
}

// =============================================================================================
// blockmatch — một work-item = một block 8x8 proc-res.
// Global = ceil8(ceil(lowW/8)) x ceil8(ceil(lowH/8)).
// =============================================================================================

#define BM_BLOCK 8
#define BM_RANGE 24
#define LW_R 0.299f
#define LW_G 0.587f
#define LW_B 0.114f

inline float bm_luma(read_only image2d_t img, int2 p, FramePC pc) {
    p = clamp(p, (int2)(0, 0), (int2)(pc.lowW - 1, pc.lowH - 1));
    float3 c = read_imagef(img, p).xyz;
    return LW_R * c.x + LW_G * c.y + LW_B * c.z;
}

// SAD trên lưới 8x8 mẫu (bước 2) trong cửa sổ 16x16.
inline float bm_sad(read_only image2d_t lowPrev, read_only image2d_t lowCur, int2 origin, int2 mv, FramePC pc) {
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            int2 o = origin + (int2)(i, j) * 2 + (int2)(1, 1);
            s += fabs(bm_luma(lowPrev, o, pc) - bm_luma(lowCur, o + mv, pc));
        }
    }
    return s;
}

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void blockmatch(BlockmatchPC pc,
                read_only image2d_t lowPrev,    // binding 2 (rgba8, imageLoad)
                read_only image2d_t lowCur,     // binding 3 (rgba8, imageLoad)
                write_only image2d_t flowImg,   // binding 6 (rgba16f)
                __global uint* m)               // binding 9
{
    int bw = (pc.lowW + BM_BLOCK - 1) / BM_BLOCK;
    int bh = (pc.lowH + BM_BLOCK - 1) / BM_BLOCK;
    int2 b = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (b.x >= bw || b.y >= bh) return;

    int2 origin = b * BM_BLOCK - (int2)(4, 4);
    float best = bm_sad(lowPrev, lowCur, origin, (int2)(0, 0), pc) - 0.02f;
    int2 bestMv = (int2)(0, 0);

    for (int dy = -BM_RANGE; dy <= BM_RANGE; dy += 6) {
        for (int dx = -BM_RANGE; dx <= BM_RANGE; dx += 6) {
            if (dx == 0 && dy == 0) continue;
            float s = bm_sad(lowPrev, lowCur, origin, (int2)(dx, dy), pc);
            if (s < best) { best = s; bestMv = (int2)(dx, dy); }
        }
    }
    for (int step = 3; step >= 1; step -= 1) {
        int2 c = bestMv;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int2 mv = clamp(c + (int2)(dx, dy) * step, (int2)(-BM_RANGE, -BM_RANGE), (int2)(BM_RANGE, BM_RANGE));
                float s = bm_sad(lowPrev, lowCur, origin, mv, pc);
                if (s < best) { best = s; bestMv = mv; }
            }
        }
    }

    bool boundary = (int)abs(bestMv.x) >= BM_RANGE - 1 || (int)abs(bestMv.y) >= BM_RANGE - 1;
    atomic_add(&m[2], 1u);
    if (boundary) atomic_add(&m[1], 1u);
    atomic_add(&m[3], (uint)(clamp(best / 64.0f, 0.0f, 1.0f) * 255.0f + 0.5f));

    float2 mv = (float2)((float)bestMv.x, (float)bestMv.y) * (float2)(pc.flowScaleX, pc.flowScaleY);
    float4 f = (float4)(-0.5f * mv, 0.5f * mv);
    int fb = BM_BLOCK / 2;   // block 8 px proc-res = 4 texel flow
    for (int j = 0; j < fb; ++j) {
        for (int i = 0; i < fb; ++i) {
            int2 q = b * fb + (int2)(i, j);
            if (q.x < pc.flowW && q.y < pc.flowH) write_imagef(flowImg, q, f);
        }
    }
}

// =============================================================================================
// flowpack — NN NCHW [5, flowH, flowW] -> flowImg / auxImg. Global = ceil8(flowW) x ceil8(flowH).
// =============================================================================================

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void flowpack(FlowpackPC pc,
              __global const float* nnFlow,   // binding 5
              write_only image2d_t flowImg,   // binding 6 (rgba16f)
              write_only image2d_t auxImg)    // binding 10 (rgba16f)
{
    int2 p = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (p.x >= pc.flowW || p.y >= pc.flowH) return;
    int plane = pc.flowW * pc.flowH;
    int idx = p.y * pc.flowW + p.x;
    float2 sc = (float2)(pc.flowScaleX, pc.flowScaleY);
    float2 f0 = (float2)(nnFlow[0 * plane + idx], nnFlow[1 * plane + idx]) * sc;
    float2 f1 = (float2)(nnFlow[2 * plane + idx], nnFlow[3 * plane + idx]) * sc;
    float mask = clamp(nnFlow[4 * plane + idx], 0.0f, 1.0f);
    write_imagef(flowImg, p, (float4)(f0, f1));
    write_imagef(auxImg, p, (float4)(mask, 0.0f, 0.0f, 1.0f));
}

// =============================================================================================
// warp — mode 0: chép frame hiện tại; mode 1: nội suy. Global = ceil8(outW) x ceil8(outH).
// =============================================================================================

// Điểm HUD = đứng yên giữa hai khung VÀ nằm trên biên rõ.
inline bool hudEdge(read_only image2d_t uPrev, read_only image2d_t uCur, float2 q, float2 px, FramePC pc) {
    float3 c = tex3(uCur, q);
    if (maxAbs3(tex3(uPrev, q) - c) >= pc.staticThr) return false;
    float g = max(maxAbs3(tex3(uCur, q + (float2)(px.x, 0.0f)) - c), maxAbs3(tex3(uCur, q + (float2)(0.0f, px.y)) - c));
    return g > 0.06f;
}

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void warp(WarpPC pc,
          read_only image2d_t uPrev,      // binding 0
          read_only image2d_t uCur,       // binding 1
          read_only image2d_t uFlow,      // binding 7 (rgba16f, linear)
          write_only image2d_t outImg,    // binding 8 (rgba8)
          __global const uint* m,         // binding 9
          read_only image2d_t uAux)       // binding 11 (rgba16f, linear; chỉ đọc khi useNN != 0)
{
    int2 p = (int2)((int)get_global_id(0), (int)get_global_id(1));
    if (p.x >= pc.outW || p.y >= pc.outH) return;
    float2 uv = ((float2)((float)p.x, (float)p.y) + 0.5f) / (float2)((float)pc.outW, (float)pc.outH);
    float3 cur = tex3(uCur, uv);

    if (pc.mode == 0) {
        write_imagef(outImg, p, (float4)(cur, 1.0f));
        return;
    }

    // Chuyển cảnh / chuyển động quá lớn.
    {
        float meanDiff = (float)m[0] / (255.0f * (float)pc.lowW * (float)pc.lowH);
        bool cut = meanDiff > 0.28f;
        if (pc.useNN == 0 && m[2] > 0u && (float)m[1] / (float)m[2] > 0.35f) cut = true;
        if (cut) { write_imagef(outImg, p, (float4)(cur, 1.0f)); return; }
    }

    float3 prev = tex3(uPrev, uv);
    if (maxAbs3(prev - cur) < pc.staticThr) {
        write_imagef(outImg, p, (float4)(cur, 1.0f));
        return;
    }

    // Nở mặt nạ HUD ~3px.
    if ((pc.pad0 & 1) == 0) {
        float2 px = 1.0f / (float2)((float)pc.outW, (float)pc.outH);
        float2 r = px * 3.0f;
        if (hudEdge(uPrev, uCur, uv + (float2)(r.x, 0.0f), px, pc) || hudEdge(uPrev, uCur, uv - (float2)(r.x, 0.0f), px, pc) ||
            hudEdge(uPrev, uCur, uv + (float2)(0.0f, r.y), px, pc) || hudEdge(uPrev, uCur, uv - (float2)(0.0f, r.y), px, pc)) {
            write_imagef(outImg, p, (float4)(cur, 1.0f));
            return;
        }
    }

    // 5 ứng viên flow (giữa + 4 khối lân cận), chọn cặp warp khớp nhất.
    float2 off = (float2)(4.0f / (float)pc.flowW, 4.0f / (float)pc.flowH);
    float4 cand[5];
    cand[0] = read_imagef(uFlow, kLinear, uv);
    cand[1] = read_imagef(uFlow, kLinear, uv + (float2)(off.x, 0.0f));
    cand[2] = read_imagef(uFlow, kLinear, uv - (float2)(off.x, 0.0f));
    cand[3] = read_imagef(uFlow, kLinear, uv + (float2)(0.0f, off.y));
    cand[4] = read_imagef(uFlow, kLinear, uv - (float2)(0.0f, off.y));
    float4 fl = cand[0];
    float3 a = tex3(uPrev, uv + fl.xy);
    float3 b = tex3(uCur, uv + fl.zw);
    float err = maxAbs3(a - b);
    float same = 0.05f / (float)max(pc.srcW, pc.srcH);
    if (err > 0.012f) {
        for (int i = 1; i < 5; ++i) {
            float4 d = fabs(cand[i] - cand[0]);
            if (max(max(d.x, d.y), max(d.z, d.w)) < same) continue;
            float3 ai = tex3(uPrev, uv + cand[i].xy);
            float3 bi = tex3(uCur, uv + cand[i].zw);
            float ei = maxAbs3(ai - bi);
            if (ei < err - 0.01f) { err = ei; fl = cand[i]; a = ai; b = bi; }
        }
    }

    // Chống "bóng ma" HUD.
    float thr = pc.staticThr * 2.0f;
    float2 qa = uv + fl.xy, qb = uv + fl.zw;
    bool aHud = maxAbs3(tex3(uPrev, qa) - tex3(uCur, qa)) < thr && length(fl.xy) > 0.0015f;
    bool bHud = maxAbs3(tex3(uPrev, qb) - tex3(uCur, qb)) < thr && length(fl.zw) > 0.0015f;
    if ((pc.pad0 & 1) != 0) { aHud = false; bHud = false; }
    if (aHud && bHud) { write_imagef(outImg, p, (float4)(cur, 1.0f)); return; }
    float mask = (pc.useNN != 0) ? read_imagef(uAux, kLinear, uv).x : 0.5f;  // trọng số phía prev
    if (aHud) { mask = 0.0f; err = 0.0f; }
    else if (bHud) { mask = 1.0f; err = 0.0f; }

    // Che khuất: lùi mềm về khung thật.
    float3 warped = glsl_mix3(b, a, mask);
    float w = glsl_smoothstep(pc.occThr * 0.5f, pc.occThr * 1.2f, err);
    float3 outc = glsl_mix3(warped, cur, w);
    write_imagef(outImg, p, (float4)(outc, 1.0f));
}
