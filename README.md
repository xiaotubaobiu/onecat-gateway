# onecat-gateway

v100 服务器(192.168.20.145/.146)上 Qwen3.8-27B 服务的 API 网关。单文件 aiohttp 实现,无第三方依赖(仅 aiohttp),systemd user unit 持久化。

## 架构

```
用户应用 (Pisper/Codex 等)
   │  OpenAI Responses API
   ▼
10.23.80.140:23303            办公网端口转发
   ▼
192.168.20.146:2223           onecat-gateway v4.3 (本仓库 gateway.py, systemd: onecat-gateway.service)
   │  翻译为 chat completions
   ▼
127.0.0.1:8000                1Cat-vLLM 1.5.0 容器 onecat-vllm (GPUs 4-7, TP=4)
                              Qwen3.8-27B-QUASAR-NVFP4 (Qwen3.5-VL 多模态, 262144 上下文)
```

## 网关职责(v4.3)

- **协议翻译**: `/v1/responses` → `/v1/chat/completions` 完整适配(引擎原生 responses 对工具调用长对话不可靠),含流式回译
- **图片透传**: `input_image` 部件三种形态(消息 content 内 / 顶层散装 item / function_call_output 列表)→ chat `image_url` 多部件;system 内图片丢弃,tool 内图片转挂为紧随其后的 user 消息(Qwen 模板限制);**全请求**上限 8 张(引擎按整条请求校验,历史图随轮次累积):超出丢最老保最新,`X-Images-Dropped` 响应头+日志报数;单图 data URI 30MB
- **上下文超窗自愈**: 引擎报"输入+输出超窗"时弹掉 `max_tokens` 重发一次,引擎原生按 `max_model_len - prompt_tokens` 钳制;prompt 本身超窗则透传 400 并附中文指引。注意:1Cat 引擎错误文本里的 prompt token 数是合成值 `ctx - max_tokens + 1`,不可解析
- **角色规范化**: instructions/developer/system 合并为开头唯一 system(解决 "Unexpected message role" / "System message must be at the beginning")
- **thinking 档位映射**: low/medium/high(max)→模板合法值 low/medium/xhigh,收进 chat_template_kwargs
- **并发控制**: 信号量(默认3)+ 队列(10)+ 排队超时 120s → 429
- **prefix cache 观测**: 每请求 cached_tokens 统计、`GET /cache/stats` 实时指标、非流式响应带 `X-Cached-Tokens` 头、system prompt 每 30s 保温重放防 LRU 逐出
- **认证**: `Authorization: Bearer <VLLM_API_KEY>`,密钥只在服务器 env 文件(600 权限),不入库不入日志

## 引擎侧关键参数(deploy/start-target-tuned.sh)

| 参数 | 值 | 说明 |
|---|---|---|
| --max-model-len | 262144 | tuned.env ONCAT_MAX_MODEL_LEN |
| --limit-mm-per-prompt | {"image": 8} | 默认 1 张/请求,Pisper agent 流程必超 |
| --enable-prompt-tokens-details | - | 按请求回报 cached_tokens |
| --kv-cache-dtype | fp8_e5m2 | V100 显存换窗口 |
| --attention-backend | FLASH_ATTN_V100 | 1Cat fork 的 V100 路径 |
| --mamba-cache-mode | align | |

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
