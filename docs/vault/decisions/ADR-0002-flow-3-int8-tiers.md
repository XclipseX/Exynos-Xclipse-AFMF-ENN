---
type: decision
tags: [model, apk]
updated: 2026-11-24
---
# ADR-0002: Flow chỉ 3 mức INT8
- Người dùng chọn 70p/360p/720p (192/640/1280), chỉ INT8, mặc định 360p (`ProcTier.DEFAULT`, `ProcTier.AVAILABLE`).
- Bỏ model thử op, SR thử, lib x86/armv7: APK 215 MB → 44 MB (commit 609e627).
