# LiteLLM Proxy vs 自研 onecat-gateway 对比评估

日期:2026-09-18 | 评估对象:[BerriAI/litellm](https://github.com/BerriAI/litellm) vs `gateway.py`(v4.3, ~1000 行)

## 结论

**暂不替换。** 当前单引擎、单租户场景下,自研网关已把 1Cat fork 的全部怪癖适配完毕且经过实战验证;LiteLLM 的核心价值(多供应商路由、密钥/预算管理、记账)在当前场景用不上,而替换后我们仍需在它的回调体系里重写全部定制逻辑,还引入 Postgres 依赖。**触发重新评估的条件见文末。**

## 逐项对比

| 维度 | LiteLLM Proxy | onecat-gateway v4.3 |
|---|---|---|
| /v1/responses 协议 | ✅ 1.63.8+ 原生支持,含 responses→chat bridge(vLLM 需 `use_chat_completions_api: true`) | ✅ 专为引擎实现,含流式回译 |
| responses 上的 guardrails | ⚠️ 仅非流式(官方文档) | 不涉及 |
| 多供应商/多引擎路由 | ✅ 核心强项,负载均衡/故障转移 | ❌ 单上游(可起第二实例) |
| 虚拟密钥/团队/预算/记账 | ✅ 强,但需 Postgres(+Redis) | ❌ 仅单一 Bearer key |
| 可观测 | ✅ Prometheus/OTel/Langfuse 等生态 | ✅ 定制:cached_tokens 统计、/cache/stats、结构化日志 |
| 上下文超窗处理 | ⚠️ 预检报错/context window fallbacks(截断或换模型) | ✅ 弹掉 max_tokens 重发,引擎原生 `max_model_len - prompt_tokens` 精确钳制 |
| 1Cat fork 模板怪癖 | ❌ 需自写回调:developer/system 合并、reasoning_effort high→xhigh 映射、chat_template_kwargs 注入 | ✅ 全部内置并实战验证 |
| 工具输出图片转挂 user 消息(Qwen 模板限制) | ❌ 需自写回调 | ✅ 内置 |
| prefix cache 保温(防 LRU 逐出) | ❌ 无此概念(它的 caching 指 Redis 语义缓存) | ✅ system prompt 30s 保温重放 |
| 运维依赖 | Python 重依赖包,完整功能要 Postgres | 单文件 + aiohttp,systemd user unit,零外部服务 |

## 迁移成本分析(若替换)

1. **必须重写的定制逻辑**(LiteLLM callback/hook 形式):
   - responses→chat 翻译里的角色合并与档位映射(LiteLLM 的 bridge 不会做 Qwen 模板的 `Unexpected message role` 适配——这正是我们第一个线上事故的修复)
   - 超窗弹预算重试(LiteLLM 的 fallback 语义是"换模型/截断",不是"让引擎钳制")
   - 工具输出图片转挂、图片数量/大小上限、prefix cache 保温
2. **新增基础设施**:Postgres(虚拟密钥/预算/记账的前提),可选 Redis
3. **风险**:LiteLLM 对"非受支持的 OpenAI 兼容 fork"的参数透传行为需要逐项回归(我们已踩过引擎模板硬校验的坑:reasoning_effort 传 high 直接 400)

## LiteLLM 真正能带来的增量

- 多租户:给不同同事/应用发独立虚拟 key,限额、记账、停用,带管理 UI
- 多引擎路由:GPU 0-3 后续若部署 Qwen3.8-Flash-Next(端口 8001),LiteLLM 可在两个引擎间做负载均衡/故障转移
- 生态:Prometheus 指标、Langfuse 追踪、guardrails 开箱即用

## 触发重新评估的条件

1. 出现**第二个引擎**(如 GPUs 0-3 部署新模型)且需要统一入口/故障转移;
2. 出现**多用户/多应用**接入,需要独立密钥、配额和用量报表;
3. 自研网关的协议维护成本明显上升(如 Responses API 规范继续演进、新增更多客户端方言)。

届时更务实的路径可能是**分层**:LiteLLM 做外层(密钥/路由/记账),自研网关降级为 1Cat 引擎专属的"方言适配器"挂在 LiteLLM 后面——两层各干各的,而不是整体替换。
