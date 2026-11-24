---
type: index
updated: 2026-11-24
---
# Xclipse ENN-SS — Map of Content

> AI agents: đọc note này trước, rồi note mới nhất trong sessions/. Người dùng: Khôi Nguyên (tiếng Việt, chỉ ra quyết định, test mọi build trên Galaxy S26+ SM-S947B, Exynos 2600, Android 16).

## Architecture
- [[pipeline-overview]] — capture → Vulkan/OpenCL → NPU (LiteRT+ENN) → overlay present
- [[build-and-release]] — CI, AOT model, Release "nightly"

## Decisions
- [[ADR-0001-vulkan-default-opencl-experimental]]
- [[ADR-0002-flow-3-int8-tiers]]
- [[ADR-0003-upscale-separate-mode]]

## Known bugs / gotchas
- [[bug-app-zoom-when-enabled]] (OPEN)
- NPU: DEPTH_TO_SPACE không chạy NPU; dispatch Samsung chỉ nhận DMA-BUF; TensorBuffer int8 phải gọi bằng reflection
- Nhiệt SoC: dùng /sys/class/sec/sec-ap-thermistor/temperature (sysfs thermal bị SELinux chặn)

## Recent sessions
- [[2026-11-24]]
