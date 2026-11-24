#!/usr/bin/env python3
"""Print the README skill table from skills-lock.json + first-party skills."""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lock = json.load(open(os.path.join(ROOT, "skills-lock.json")))["skills"]
OWN = {"graphify": "memory", "obsidian-memory": "memory", "cloud-session-persistence": "context",
       "craft-anything": "engineering", "remotion": "video", "video-cli-tools": "video", "find-mcp-and-connectors": "meta", "skill-vetting": "meta", "knowledge-vault": "memory", "skillpack-bootstrap": "meta"}
CAT = {"memory": "🧠 Bộ nhớ & Knowledge graph", "context": "🪶 Tiết kiệm token & ngữ cảnh", "workflow": "🧭 Quy trình code",
       "engineering": "🛠️ Kỹ thuật phần mềm", "frontend": "🎨 Frontend & UI", "security": "🔒 Bảo mật",
       "video": "🎬 Edit video", "youtube": "📺 YouTube", "meta": "🔎 Tìm & quản lý skill/MCP/connector"}
rows = {}
for d in sorted(os.listdir(os.path.join(ROOT, ".agents", "skills"))):
    f = os.path.join(ROOT, ".agents", "skills", d, "SKILL.md")
    if not os.path.exists(f): continue
    t = open(f, encoding="utf-8").read()
    heavy = "🔐" if "Requires login" in t else "🔴" if "Requires extra installation" in t else ("🟡" if re.search(r"^> (Needs|\*\*Requires small)", t, re.M) else "🟢")
    if d in lock: src, cat = f"[{lock[d]['repo']}](https://github.com/{lock[d]['repo']})", lock[d]["category"]
    else: src, cat = "Skillpack", OWN.get(d, "engineering")
    rows.setdefault(cat, []).append(f"| [{d}](.agents/skills/{d}/SKILL.md) | {heavy} | {src} |")
for c in CAT:
    if c in rows:
        print(f"\n### {CAT[c]} ({len(rows[c])})\n\n| Skill | Cài đặt | Nguồn |\n|---|---|---|"); print("\n".join(rows[c]))
