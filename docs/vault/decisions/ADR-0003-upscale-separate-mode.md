---
type: decision
tags: [upscale, sr]
updated: 2026-11-24
---
# ADR-0003: Upscale là chế độ riêng
- ×2: chụp 1170x540, SR INT8 trên NPU (+4.62 dB so với bilinear); ×1: làm nét vùng giữa 1536x704. Không chạy chung nội suy; menu 2 tab.
- Chỉ hỗ trợ ngang (model cố định 1170x540 / 1536x704); dọc → srNote, không SR.
- Pipeline 3 tầng để giữ 60 fps (commit 6f47f70) — chưa xác nhận trên máy.
