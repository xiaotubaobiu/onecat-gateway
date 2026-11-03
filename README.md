# onecat-gateway

v100 服务器(192.168.20.145/.146)上 Qwen3.8-27B 服务的 API 网关。单文件 aiohttp 实现,无第三方依赖(仅 aiohttp),systemd user unit 持久化。

## 架构

```
用户应用 (Pisper/Codex 等)
   │  OpenAI / Anthropic Messages API
   ▼
10.23.80.140:23303            办公网端口转发
   ▼
192.168.20.146:2223           onecat-gateway v4.4 (本仓库 gateway.py, systemd: onecat-gateway.service)
   │  翻译为 chat completions
   ▼
127.0.0.1:8000                1Cat-vLLM 1.5.0 容器 onecat-vllm (GPUs 4-7, TP=4)
                              Qwen3.8-27B-Uncensored-NVFP4-1Cat + DFlash2 (262144 上下文)
```

## 网关职责(v4.4)

- **Anthropic Messages**: `/v1/messages` 原生转发到引擎; 保留 system/tools/thinking、Anthropic 请求头、查询参数、上游状态码及响应头; JSON 与 SSE 逐块透传, 不走 OpenAI 规范化。支持 `x-api-key` 和 Bearer; 断连取消上游并释放槽位。
- **协议翻译**: `/v1/responses` → `/v1/chat/completions` 完整适配(引擎原生 responses 对工具调用长对话不可靠),含流式回译
- **图片透传**: `input_image` 部件三种形态(消息 content 内 / 顶层散装 item / function_call_output 列表)→ chat `image_url` 多部件;system 内图片丢弃,tool 内图片转挂为紧随其后的 user 消息(Qwen 模板限制);**全请求**上限 8 张(引擎按整条请求校验,历史图随轮次累积):超出丢最老保最新,`X-Images-Dropped` 响应头+日志报数;单图 data URI 30MB
- **上下文超窗自愈**: 引擎报"输入+输出超窗"时弹掉 `max_tokens` 重发一次,引擎原生按 `max_model_len - prompt_tokens` 钳制;prompt 本身超窗则透传 400 并附中文指引。注意:1Cat 引擎错误文本里的 prompt token 数是合成值 `ctx - max_tokens + 1`,不可解析
- **角色规范化**: instructions/developer/system 合并为开头唯一 system(解决 "Unexpected message role" / "System message must be at the beginning")
- **thinking 档位映射**: low/medium/high(max)→模板合法值 low/medium/xhigh,收进 chat_template_kwargs
- **并发控制**: OpenAI 与 Anthropic 共用信号量; 部署 unit 显式设置 6 路, 与引擎 `ONCAT_MAX_SEQS=6` 对齐。队列 10、超时 120s → 429; 未配置环境变量时程序兼容默认值仍为 3。
- **推理 token 回报**: fork 的 usage 缺 `completion_tokens_details`(上游代码未随 fork 发布),网关翻译时把思考文本 POST 引擎 `/tokenize` 取**精确**计数回填 `output_tokens_details.reasoning_tokens`(不可用时字符估算兜底),Pisper"推理 Tokens"面板得以显示
- **prefix cache 观测**: 每请求 cached_tokens 统计、`GET /cache/stats` 实时指标、非流式响应带 `X-Cached-Tokens` 头; 本次部署关闭后台保温重放 (`ONECAT_WARM_ENABLED=0`), 避免后台生成绕过用户槽位; 引擎 prefix caching 保持开启
- **认证**: `Authorization: Bearer <VLLM_API_KEY>`,密钥只在服务器 env 文件(600 权限),不入库不入日志

## 引擎侧关键参数(deploy/start-dflash2-tuned.sh)

| 参数 | 值 | 说明 |
|---|---|---|
| --max-model-len | 262144 | tuned.env ONCAT_MAX_MODEL_LEN |
| --limit-mm-per-prompt | {"image": 8} | 默认 1 张/请求,Pisper agent 流程必超 |
| --enable-prompt-tokens-details | - | 按请求回报 cached_tokens |
| --kv-cache-dtype | fp8_e4m3 | 显式使用 E4M3; 当前 fork 将 E5M2 标为优化路径, 两者不要混淆 |
| --attention-backend | FLASH_ATTN_V100 | 1Cat fork 的 V100 路径 |
| --mamba-cache-mode | align | |
| --dtype | float16 | V100 |
| --max-num-seqs | 6 | 网关必须同步设置 |
| --gpu-memory-utilization | 0.80 | 启动与并发测试校验显存余量 |
| --speculative-config | method=dflash, draft kv_cache_dtype=auto | 已有 DFlash2 草稿; 默认 greedy 草稿采样 |

## 部署 / 更新

```bash
# 网关(秒级,不断引擎)
scp gateway.py v100:/home/z/onecat/gateway/gateway.py
ssh v100 'systemctl --user restart onecat-gateway.service'

# 引擎参数(重启容器, 停机 2-3 分钟)
ssh v100 'python3 patches/patch_mmlimit.py && docker restart onecat-vllm'
```

回滚:`/home/z/onecat/gateway/` 与 `/home/z/onecat/bin/` 下所有变更均有 `*.bak-YYYYMMDD-*` 时间戳备份。

## 验证

```bash
# 上下文超窗六项测试 (T1-T6)
ssh v100 'set -a; . /home/z/onecat/config/onecat.env; set +a; python3 /tmp/ctx_test.py'

# 视觉四路径: responses消息内图片 / chat直通 / 工具输出附图 / 顶层散装图
# 见 2026-09-18 会话记录, 红图四路径均正确回答"红色"
```

## 2026-09-18 事故修复记录

1. **超窗 400("差 1 个 token")**:应用带 195k 输入 + 128000 输出预算,旧网关按 15% 比例盲缩 4 次(128000→66816)仍差 1 个 token。→ 改为弹掉 max_tokens 让引擎原生精确钳制。
2. **"读错图"**:网关 responses 翻译层只提取文本,`input_image` 被静默丢弃——模型从未看见任何图片,全靠工具输出里的文字(如 PIL 查到的尺寸)编造答案。→ v4.3 图片透传。
3. **"At most 1 image(s)"**:引擎多模态默认 1 张/请求,agent 流程(附件+工具读图)必超。→ `--limit-mm-per-prompt {"image": 8}`。
4. **多图随历史累积撞限**:引擎按整条请求校验图片数,agent 会话历史里的图每轮重发,累计超过 8 张会硬 400。→ 网关改为全请求统一计数、丢最老保最新,永不触发引擎限制。


## DFlash2 / Messages 部署

主模型固定 revision: `tangles/Qwen3.8-27B-Uncensored-NVFP4-1Cat@f1e503c3b16f8baac5d08decad8a2fad7dee8c6b`。
草稿使用本机已校验的 `incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4`。
新主模型仅 MLP 使用 NVFP4, attention/GDN 等权重保持 BF16 存储并以 float16 加载; 下载约 31 GB。
本机 `/Dataset` 是 NTFS/FUSE, 写入锁显著限制下载吞吐。权重实际存放于 `/home/z/onecat/model-cache/Qwen3.8-27B-Uncensored-NVFP4-1Cat` (ext4), `/Dataset/onecat/models/Qwen3.8-27B-Uncensored-NVFP4-1Cat` 作为符号链接入口; Compose 已同时挂载两个目录。
客户端模型名仍为 `qwen3.8-27b`。当前安装的 1Cat-vLLM 1.5.0 已支持所需功能, 保留该版本。

1. 下载并校验两个模型到 `/Dataset/onecat/models/`。
2. 安装 `deploy/start-dflash2-tuned.sh` 到 `/home/z/onecat/bin/`, 使用可执行权限。
3. 将 `deploy/tuned.env.example` 内容写入 `/home/z/onecat/config/tuned.env`, 合并 `deploy/docker-compose.yml` 的 `dflash-tuned` 模式分支; 模式文件写 `dflash-tuned`。
4. `docker compose -f /home/z/onecat/compose/docker-compose.yml up -d --force-recreate` 以载入新环境; 单纯 `docker restart` 不会更新 Compose 的 env_file 值。
5. 引擎健康后安装 `gateway.py` 与 `deploy/onecat-gateway.service`; 执行 `systemctl --user daemon-reload && systemctl --user restart onecat-gateway.service`。
6. 若需降为 4 路, 同时修改引擎 `ONCAT_MAX_SEQS` 与网关 `ONECAT_MAX_CONCURRENT`; 如果真实 env 文件也设置该值, 必须一并同步 (EnvironmentFile 优先于 Environment)。

该主机当前引擎由 Docker Compose 管理, 没有已安装的引擎 systemd 单元; 网关由用户级 systemd 管理。
上线前保留模型路径、Compose、mode、tuned.env、网关代码和 user unit 备份, 回退时恢复并重建容器。

### 自动验证

```bash
# 无 GPU、无密钥、无外网的真实 HTTP 回归测试 (依赖 aiohttp)
python -m unittest discover -s tests -p 'test_*.py' -v

# 显式调用的全链路测试与吞吐测量; 从既有受保护环境加载 VLLM_API_KEY
python tests/live_smoke.py --concurrency 6 --output /tmp/onecat-smoke.json
```

全链路测试包含 chat/Messages 普通和 SSE、Anthropic 工具调用、Responses 回归、并发压测、DFlash 计数差分及 GPU 峰值。
单次粗口文学样例用于检查该权重的正常生成, 不能据此保证所有主题都不拒答。
`/v1/messages/count_tokens` 尚未在网关开放; 本次验收针对 `/v1/messages`。
