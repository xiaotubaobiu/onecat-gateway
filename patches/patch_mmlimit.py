p = "/home/z/onecat/bin/start-target-tuned.sh"
s = open(p).read()
if "--limit-mm-per-prompt" in s:
    print("ALREADY_PATCHED")
    raise SystemExit
old = "  --enable-prefix-caching \\\n"
new = "  --limit-mm-per-prompt '{\"image\": 8}' \\\n  --enable-prefix-caching \\\n"
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("PATCHED")
