---
type: architecture
tags: [ci, release]
updated: 2026-11-24
---
# Build & release
- Repo: github.com/XclipseX/Exynos-Xclipse-AFMF-ENN (đổi từ ShizukaDreemurr), nhánh `feat/initial-import`.
- Push → `.github/workflows/build.yml` build debug APK → Release `nightly` (link cố định `releases/download/nightly/XclipseENN-SS.apk`). Phiên này không có quyền GitHub API repo mới: theo dõi bằng `git ls-remote origin refs/tags/nightly` == HEAD.
- Sau push: `git update-ref refs/remotes/origin/feat/initial-import HEAD`.
- Đổi model: workflow `aot-model.yml` (ai-edge-litert 2.3.0 + samsung nightly, Target E9965) publish nhánh `prebuilt-models`; build.yml chỉ copy 5 model app dùng.
- Người dùng cho phép: tự build, theo dõi, gửi link (từ 2026-11-24). Commit kèm Co-Authored-By + Claude-Session.
