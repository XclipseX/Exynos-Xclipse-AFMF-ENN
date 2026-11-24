---
type: decision
tags: [gpu, opencl, vulkan]
updated: 2026-11-24
---
# ADR-0001: Vulkan mặc định, OpenCL là tuỳ chọn thử nghiệm
- OpenCL trên S26+ là ANGLE-CL 3.0 trên Vulkan (driver Samsung 25.2.39), không phải driver AMD gốc.
- Stress 2 phút (2340x1080, proc 640x288): cùng ms×MHz (≈980); OpenCL nhanh hơn chỉ vì GPU bị ép 980 MHz, Vulkan chạy ~250 MHz.
- @120 fps: Vulkan p50 6.3 ms, trễ hạn 15%, điện thêm ~0 W; OpenCL dma_buf p50 2.4 ms, 0% trễ hạn, +1.0 W; mJ/khung (hết sức) 1.8 vs 5.7. OpenCL chép khung: tệ nhất (+2.5–5 W).
- VK_KHR_global_priority HIGH/REALTIME được cấp nhưng không tăng xung → vô ích.
- Quyết định: giữ Vulkan; thêm menu "Tính toán GPU: Vulkan / OpenCL (thử)" cho đường NN (commit 5d26c85), lỗi 3 lần → tự về Vulkan.
