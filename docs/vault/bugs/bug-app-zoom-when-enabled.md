---
type: bug
tags: [overlay, capture, open]
updated: 2026-11-24
---
# BUG (OPEN): bật Xclipse là cả app/màn hình bị phóng to
- Người dùng: bản trước không bị, bản 5d26c85 bị; bấm Bật trong menu → lag rồi nội dung bị zoom (cả app chính lẫn game). Lần đầu nhắc ~16:48 cùng ngày.
- Ảnh: status bar, nav bar, HUD, bong bóng vẫn đúng cỡ; chỉ nội dung bên dưới to → nghi lớp overlay present khung sai cỡ/hướng. HUD lúc đó: PASSTHROUGH 70p, FPS 0/0, "(overlay tắt)", tab Upscale đang chọn.
- Nghi phạm: upscale ×2 hạ capture còn 1170 (`captureLongSide`) / overlay giữ khung cũ; cần dòng Capture trên HUD (mức Đầy đủ) và thử: Dừng / chuyển tab Nội suy.
