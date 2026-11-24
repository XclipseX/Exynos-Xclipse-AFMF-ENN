#!/usr/bin/env python3
"""Vendor third-party skills listed in vendor/manifest.json into .agents/skills/.

Usage: python3 scripts/sync_vendor.py [--only owner/repo]
Writes skills-lock.json (repo, path, commit, license, weight) and injects an
"ask before installing" notice into heavy skills. Never executes cloned code.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, ".agents", "skills")
NOTICE = """
> **⚠️ Requires extra installation.** Needs: {requires}.
> Before using: check whether it is installed. If not, **install it yourself** (prefer user-level installs, no sudo unless needed)
> and tell the user in one line what you installed. Stop and ask only if the install fails or needs more than ~1 GB.
> Source: https://github.com/{repo} (vendored, license: {license}).
"""
AUTH_NOTICE = """
> **🔐 Requires login / account / API key.** Needs: {requires}.
> **Ask the user before installing or signing in** – say what will be installed, which account or key is needed and what it can access.
> Never ask for passwords or keys in chat; use the platform's secret/credential settings or the tool's own login flow.
> If the user declines, continue with a lighter skill from this pack or instructions only.
> Source: https://github.com/{repo} (vendored, license: {license}).
"""

def sh(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

def inject(skill_md, text):
    s = open(skill_md, encoding="utf-8").read()
    m = re.match(r"^---\n.*?\n---\n", s, re.S)
    s = s[:m.end()] + text + s[m.end():] if m else text + s
    open(skill_md, "w", encoding="utf-8").write(s)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only")
    a = ap.parse_args()
    man = json.load(open(os.path.join(ROOT, "vendor", "manifest.json")))
    lock_p = os.path.join(ROOT, "skills-lock.json")
    lock = json.load(open(lock_p)) if os.path.exists(lock_p) else {"skills": {}}
    os.makedirs(DEST, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for src in man["sources"]:
            if a.only and src["repo"] != a.only: continue
            d = os.path.join(tmp, src["repo"].replace("/", "__"))
            env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
            subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{src['repo']}.git", d], check=True, env=env)
            commit = sh("git", "rev-parse", "--short", "HEAD", cwd=d)
            lic = next((f for f in os.listdir(d) if f.upper().startswith("LICEN")), None)
            for sk in src["skills"]:
                sp = os.path.normpath(os.path.join(d, sk["path"]))
                name = sk.get("name") or os.path.basename(sp)
                out = os.path.join(DEST, name)
                if os.path.exists(out) and lock["skills"].get(name, {}).get("repo") not in (None, src["repo"]):
                    sys.exit(f"name clash: {name}")
                shutil.rmtree(out, ignore_errors=True)
                ex = set(sk.get("exclude", [])) | {".git"}
                shutil.copytree(sp, out, ignore=lambda p, n: [x for x in n if x in ex and p == sp] + [x for x in n if x == ".git"])
                if lic and not any(f.upper().startswith("LICEN") for f in os.listdir(out)):
                    shutil.copy(os.path.join(d, lic), os.path.join(out, "LICENSE.upstream"))
                if sk.get("name"):  # keep frontmatter name in sync with folder name
                    f = os.path.join(out, "SKILL.md"); t = open(f, encoding="utf-8").read()
                    open(f, "w", encoding="utf-8").write(re.sub(r"(?m)^name:.*$", f"name: {name}", t, count=1))
                weight = sk.get("weight", "light")
                fmt = dict(requires=sk.get("requires", "see source"), repo=src["repo"], license=src["license"])
                if sk.get("auth"):
                    inject(os.path.join(out, "SKILL.md"), AUTH_NOTICE.format(**fmt))
                elif weight == "heavy":
                    inject(os.path.join(out, "SKILL.md"), NOTICE.format(**fmt))
                elif sk.get("requires"):
                    inject(os.path.join(out, "SKILL.md"), f"\n> Needs: {sk['requires']} – if missing, install it yourself (one command) and tell the user.\n")
                lock["skills"][name] = {"repo": src["repo"], "path": sk["path"], "commit": commit, "license": src["license"],
                                        "category": src["category"], "weight": weight, "auth": bool(sk.get("auth"))}
                print(f"{name:40s} {src['repo']}@{commit} [{weight}]")
    lock["skills"] = dict(sorted(lock["skills"].items()))
    json.dump(lock, open(lock_p, "w"), indent=1, ensure_ascii=False); open(lock_p, "a").write("\n")

if __name__ == "__main__":
    main()
