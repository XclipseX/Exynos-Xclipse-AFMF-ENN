---
type: architecture
tags: [pipeline, vulkan, opencl, npu]
updated: 2026-11-24
---
# Pipeline overview
- Capture: MediaProjection VirtualDisplay → ImageReader AHardwareBuffer (`core-capture/.../ScreenCaptureEngine.kt`, cỡ từ `CaptureConfig.fromContext`, upscale ×2 hạ cạnh dài 1170 qua `FluidSettings.captureLongSide()`).
- Điều phối: `app/.../FramePipeline.kt` `process()` → passthrough / `upscaleOnly` / nội suy (NN qua NnPre→NPU→NnPost, hoặc GPU block-match GpuAsync).
- Vulkan: `core-vulkan/src/main/cpp/pipeline.cpp` (shaders `core-vulkan/src/main/shaders/*.comp`), 3 slot async `af_[]`, present qua swapchain overlay.
- OpenCL (thử nghiệm, chỉ đường NN): NnPre chạy `cl_pack.comp` chép prev/cur vào VkBuffer xuất dma_buf; `cl/cl_engine.cpp` (ANGLE-CL) chạy flowpack+warp; NnPost copy kết quả vào F.out rồi present. Bật bằng `FluidSettings.gpuApi`.
- NPU: LiteRT 2.3.2 CompiledModel + `libLiteRtDispatch_Samsung.so`, model AOT E9965 (`core-enn/.../NpuEnv.kt`, `SrBackend.kt`, `LiteRtBackend.kt`).
- Model trong APK: flow_{192,640,1280}_int8, sr_x2_1170x540, sr_x1_1536x704.
- Đo: tab NPU (`MainActivity.runGpuBench/runGpuStress`, `GpuStress.kt`), HUD + CSV (`TelemetryRecorder`).
