# 2026-11-03 部署验收

已部署并运行; 网关源码与本机运行文件一致。引擎保留已支持 DFlash2 的 1Cat-vLLM **1.5.0**, 更新权重、运行参数和网关。

## 最终配置

| 项目 | 实际值 |
|---|---|
| 主模型 | tangles/Qwen3.8-27B-Uncensored-NVFP4-1Cat |
| 主模型 revision | f1e503c3b16f8baac5d08decad8a2fad7dee8c6b |
| 草稿 | incoai/Qwen3.8-27B-DFlash2 @ dedf8df68adfb1afeaf7b7480c0a0243108177b4 |
| 客户端模型名 | qwen3.8-27b |
| GPU / TP | 物理 GPU 4,5,6,7 / TP4 |
| 上下文窗口 | 262144 tokens, 输入 + 输出合计 |
| 引擎 / 网关并发 | 6 / 6 |
| 队列 / 等待超时 | 10 / 120 秒 |
| GPU memory utilization | 0.80 |
| dtype / attention | float16 / FLASH_ATTN_V100 |
| target / draft KV dtype | fp8_e4m3 / auto |
| mamba / prefix cache | align / enabled |
| DFlash2 | method=dflash, 7 draft tokens, greedy draft sampling |
| 后台保温 | 关闭, 避免额外生成绕过网关槽位 |
| API | 网关 :2223 → 引擎 127.0.0.1:8000 |
| 持久化 | 网关 systemd --user; 引擎 Docker Compose + unless-stopped |

首轮 6 路 / 0.80 配置即完成加载和 CUDA Graph 捕获, 未触发 OOM 重试或降到 4 路。引擎日志无 ERROR、OOM 或 speculative fallback; DFlash2 verifier 与 draft CUDA graphs 均已激活。

KV Cache 启动容量估算为 **793360 tokens**, 对 262144-token 请求估算 **3.03 倍并发**。6 个调度槽位适用于混合长度请求, 不表示 6 路满 256K 请求都能同时驻留。

## 权重完整性与存放

主模型 20 个文件均检查大小; 所有 LFS 文件比对 SHA-256; 所有非 LFS 元数据比对 Git blob 哈希。safetensors 头、数据边界和 1775 项索引映射校验通过。已存在的草稿权重 SHA-256 与上游固定 revision 一致, config 等元数据 Git blob 哈希一致。

Hugging Face CLI 在 ext4 目录完成下载。`/Dataset` 为 NTFS/FUSE, 初始下载写入线程被文件锁阻塞; 主模型实际存于 `/home/z/onecat/model-cache/Qwen3.8-27B-Uncensored-NVFP4-1Cat`, 原模型同级的 `/Dataset/onecat/models/Qwen3.8-27B-Uncensored-NVFP4-1Cat` 为符号链接。Docker 的既有挂载覆盖两处。原 QUASAR 权重保留。

## 接口与并发

| 验收 | 结果 |
|---|---|
| /v1/chat/completions JSON | 200 |
| /v1/chat/completions SSE | 200, 多事件实时到达 |
| /v1/messages JSON (x-api-key) | 200 |
| /v1/messages SSE | 200, 含 message_stop |
| Anthropic tool_use | 200, 正确生成工具调用 |
| /v1/responses 回归 | 200 |
| 六并发 | 6/6 成功, 每路输出 256 tokens |
| 权重响应抽检 | 正常生成包含粗口的虚构成年侦探独白 |
| 本地 HTTP 回归 | 9/9 通过, 包含 auth、头/体/状态码、SSE、gzip、共享限流和断连释放 |

抽检用于验证指定权重的正常响应, 不代表所有主题均不拒答。`/v1/messages/count_tokens` 未在网关开放; 没有声称完整 Claude Code 工作流兼容。

## 吞吐与 DFlash 证据

单路、短输入、temperature=0、1024 输出 tokens 的流式请求:

- 引擎 decode 时间 **8.427 秒**, decode 吞吐 **121.40 tok/s** (1023 / decode 时间)。
- 端到端 **8.553 秒**, **119.72 tok/s**, 首个文本约 **0.124 秒**。
- 同一次请求 speculative counter 差分: **277 轮**, **1939 draft tokens**, **748 accepted tokens**, 接受率 **38.58%**。
- 没有进行同一权重下关闭 DFlash 的对照, 因此不报告加速倍数。

## 长输入与显存

250033-token 合成输入、16-token 输出预算通过网关验证, 最终 HTTP **200**, 输出 **OK** (2 tokens)。

首次冷输入达到测试客户端 **900 秒**读超时; 引擎未 OOM、未退出。网关取消请求并释放槽位。将测试客户端等待上限延至 1800 秒后, 同输入重试命中 **240608 tokens** 前缀缓存, 耗时 **75.87 秒**成功。这个结果不是 75.9 秒冷启动延迟; 超长冷输入预填充很慢, 客户端需要足够长的超时。

| GPU | 常规接口/六并发峰值 | 长输入重试峰值 | 总显存 |
|---|---:|---:|---:|
| 4 | 24.80 GiB | 26.06 GiB | 32 GiB |
| 5 | 24.80 GiB | 26.06 GiB | 32 GiB |
| 6 | 24.80 GiB | 26.06 GiB | 32 GiB |
| 7 | 24.80 GiB | 26.06 GiB | 32 GiB |

显存均低于 30.5 GiB。E4M3 按需求显式启用; 当前 1Cat 源码将 E5M2 标为 V100 优化路径, 不应把 E4M3 描述为已证实的最快长文本设置。

## 回滚

本机备份: `/home/z/onecat/upgrade-20261103/backup/`。已准备 `/home/z/onecat/upgrade-20261103/rollback.py`, 恢复原 Compose、mode、tuned.env、网关与 user unit 后重建引擎; 原 QUASAR 主模型仍在。
