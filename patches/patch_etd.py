import sys

# 2026-09-18: 开启 --enable-prompt-tokens-details, 引擎按请求回报
# usage.prompt_tokens_details.cached_tokens (prefix cache 观测的前提)
p = '/home/z/onecat/bin/start-target-tuned.sh'
s = open(p).read()
if '--enable-prompt-tokens-details' in s:
    print('ALREADY_PATCHED')
    sys.exit(0)
old = '  --enable-prefix-caching \\\n'
new = '  --enable-prefix-caching \\\n  --enable-prompt-tokens-details \\\n'
assert old in s, 'anchor not found'
open(p, 'w').write(s.replace(old, new, 1))
print('PATCHED')
