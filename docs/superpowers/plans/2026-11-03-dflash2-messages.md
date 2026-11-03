# DFlash2 and Anthropic Messages Implementation Plan

> Execute inline using superpowers:executing-plans and test-driven-development. User authorized automatic deployment and GitHub publication.

**Goal:** Serve the requested uncensored NVFP4 checkpoint with DFlash2 on GPUs 4–7, retain qwen3.8-27b and 262144 context, and expose Anthropic Messages through the gateway.

**Architecture:** Retain Docker host-network vLLM on localhost:8000 and the user-systemd aiohttp gateway on :2223. Native Anthropic traffic uses a byte-preserving proxy and the same semaphore as OpenAI traffic. Keep the existing Responses translation.

**Tech stack:** Existing 1Cat-vLLM 1.5.0, Docker Compose, Python/aiohttp, unittest, Hugging Face CLI.

**Spec:** User's five-task instruction in this chat.

## Global constraints
- GPUs 4,5,6,7; TP4; served model qwen3.8-27b; context 262144.
- float16, FLASH_ATTN_V100, fp8_e4m3, mamba align, prefix caching.
- DFlash2 method=dflash, draft kv_cache_dtype=auto; try 6 slots / 0.80 memory first, then 4 if necessary; GPU use below 30.5 GiB.
- Gateway user slots equal final engine slots; queue 10, timeout 120s. Disable background warm replay to avoid bypassing this bound.
- Preserve working deployment in local backups. No credentials, model weights, private request logs in Git.

## Review focus
- Anthropic system/tools/thinking payload must bypass OpenAI normalization.
- x-api-key and Bearer authentication, upstream headers/errors, redirect status must survive correctly.
- SSE first chunk must arrive before upstream finishes; disconnect must close upstream and free shared slot.
- Long context and draft must fit memory; verify actual speculative metrics, not only startup flags.
- Do not claim generalized lack of refusals from one benign content sample.

## Tasks
1. [x] Download pinned main checkpoint; validate config, indexed tensors and hashes; verify existing draft against its pinned upstream revision.
2. [x] Write failing HTTP integration tests for Messages/auth/status/SSE/concurrency/cancellation; implement native forwarding and run full unit suite.
3. [x] Back up live files; add tuned DFlash launcher and parameters; recreate engine using existing Compose management; retry documented resource settings if necessary.
4. [x] Install gateway and explicit systemd concurrency; daemon-reload/restart; test chat and Messages JSON/SSE, tools, errors, Responses regression, peak GPU memory, speculative metrics and decode throughput.
5. [x] Fresh code review, fix substantive findings, publish reviewed code and deployment documentation to supplied repository; verify remote commit.

## Progress and rulings
- Existing deployment has no installed engine systemd unit: Docker Compose owns engine; gateway uses systemd --user. Recreate engine via Compose and reload/restart the actual gateway user unit.
- Installed source supports explicit E4M3 but documents E5M2 as optimized. Honor explicit E4M3 initially and record performance.
- Main checkpoint pinned to f1e503c3b16f8baac5d08decad8a2fad7dee8c6b. Existing draft is already downloaded; verify it before deciding whether download is needed.
- Git checkout is a new local clone on upgrade/dflash2-messages-20261103, separate from running production files.

- Download ruling: /Dataset is fuseblk (NTFS), and all 24 range writer threads block in filesystem write locks. Network-only 8 MiB requests finish in ~1.2–2s. Stage weights on local /home/z ext4 with HF CLI; expose the requested sibling path via symlink if sequential transfer remains unsuitable. Docker already mounts both paths.
- Gateway native proxy: 7 tests failed before implementation; 9 HTTP regression tests now pass including gzip, queue cancellation, and shared semaphore.
- Independent review completed: fixed upstream SSE Cache-Control preservation (regression failed then passed), require positive speculative metric deltas, and reject missing/failed GPU samples. Nine HTTP tests pass; shell/Python syntax and Compose config validated.
- Main weights validated: all 20 repository files, all LFS SHA256 values, safetensors headers/data boundaries, and 1775 indexed tensors. HF CLI completed on ext4; sibling model path is a symlink. Engine first attempt 6 / 0.80 is compiling with DFlash2 (7 speculative tokens).
- Deployment complete on first 6 / 0.80 attempt. Live smoke 12/12, unit HTTP suite 9/9. Benchmark 121.40 decode tok/s with positive draft and acceptance deltas. Long input 250033 tokens: initial client timeout at 900s, retry reused 240608 tokens and returned OK/200 in 75.87s; peak 26.0625 GiB per GPU. Code published to main as e08aa58904c1621ea2898a1fc45e4711d66a57c7; detailed acceptance report accompanies final documentation update. Docker unless-stopped restored.
