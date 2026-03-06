# Mini-OpenClaw 开发需求文档 (PRD)

## 一、项目介绍

### 1.1 功能与目标定位

Mini-OpenClaw 是一个基于 Python 重构的、轻量级且高度透明的 AI Agent 系统，旨在复刻并优化 OpenClaw（原名 Moltbot/Clawdbot）的核心体验。

本项目不追求构建庞大的 SaaS 平台，而是致力于打造一个运行在本地的、拥有"真实记忆"的数字副手。其核心差异化定位在于：

- **文件即记忆 (File-first Memory)**：摒弃不透明的向量数据库，回归最原始、最通用的 Markdown/JSON 文件系统。用户的每一次对话、Agent 的每一次反思，都以人类可读的文件形式存在。
- **技能即插件 (Skills as Plugins)**：遵循 Anthropic 的 Agent Skills 范式，通过文件夹结构管理能力，实现"拖入即用"的技能扩展。
- **透明可控**：所有的 System Prompt 拼接逻辑、工具调用过程、记忆读写操作对开发者完全透明，拒绝"黑盒"Agent。

### 1.2 项目核心技术架构

本项目要求完全采用**前后端分离**架构，后端作为纯 API 服务运行。

- **后端语言**：Python 3.10+（强制使用 Type Hinting）
- **Web 框架**：FastAPI（提供 RESTful 接口，支持异步处理）
- **Agent 编排引擎**：LangChain 1.x (Stable Release)
  - **核心 API**：必须使用 `create_agent` API（`from langchain.agents import create_agent`），基于 Graph 运行时的 Agent
  - 严禁使用旧版的 `AgentExecutor` 或早期的 `create_react_agent`（旧链式结构）
- **RAG 检索引擎**：LlamaIndex (LlamaIndex Core)，用于非结构化文档的混合检索（Hybrid Search）
- **模型接口**：兼容 OpenAI API 格式（支持 OpenRouter、DeepSeek、Claude 等模型直连）
- **数据存储**：本地文件系统 (Local File System) 为主，不引入 MySQL/Redis 等重型依赖
- **大模型**：默认使用通义千问 qwen-plus，支持用户自行配置
- **自我进化**：通过"心跳（heartbeat）"机制，让 Agent 在空闲时主动执行蒸馏与自省，从历史对话中提取经验、优化技能、更新长期记忆，实现"主动进化"
- **Channel（外部通道）**：统一的外部渠道抽象层，支持 IM、邮件、Webhook 等多种通道接入
- **Gateway**：通过心跳与任务文件实现"异步任务完成后唤醒 Agent"，支持定时任务调度
- **记忆**：项目级记忆与人格级记忆，通过工作流在每次 session 结束时自动回顾对话，将关键信息更新到 `MEMORY.md`、`USER.md` 等文件

---

## 二、内置工具

Mini-OpenClaw 在启动时，除了加载用户自定义的 Skills 外，必须内置以下 **6 个核心基础工具（Core Tools）**。根据"优先使用 LangChain 原生工具"的原则，技术选型如下：

### 2.1 命令行操作工具 (Command Line Interface)

- **功能描述**：允许 Agent 在受限的安全环境下执行 Shell 命令
- **实现逻辑**：直接使用 LangChain 内置工具 `langchain_community.tools.ShellTool`
- **配置要求**：初始化时需配置 `root_dir` 限制操作范围（沙箱化），需预置黑名单拦截高危指令
- **工具名称**：`terminal`

### 2.2 Python 代码解释器 (Python REPL)

- **功能描述**：赋予 Agent 逻辑计算、数据处理和脚本执行的能力
- **实现逻辑**：直接使用 LangChain 内置工具 `langchain_experimental.tools.PythonREPLTool`
- **工具名称**：`python_repl`

### 2.3 Fetch 网络信息获取

- **功能描述**：用于获取指定 URL 的网页内容，Agent 联网的核心
- **实现逻辑**：基于 `langchain_community.tools.RequestsGetTool` 封装，使用 BeautifulSoup 或 html2text 清洗 HTML，仅返回 Markdown 或纯文本内容
- **工具名称**：`fetch_url`

### 2.4 文件读取工具 (File Reader)

- **功能描述**：用于精准读取本地指定文件的内容，是 Agent Skills 机制的核心依赖
- **实现逻辑**：基于 `langchain_community.tools.file_management.ReadFileTool`，必须设置 `root_dir` 为项目根目录
- **工具名称**：`read_file`

### 2.5 RAG 检索工具 (Hybrid Retrieval)

- **功能描述**：当用户询问具体的知识库内容时，Agent 可调用此工具进行深度检索
- **技术选型**：LlamaIndex，实现 Hybrid Search（BM25 + Vector Search），索引持久化存储在 `storage/`
- **工具名称**：`search_knowledge_base`

### 2.6 文件写入工具 (File Writer)

- **功能描述**：赋予 Agent 直接创建和写入本地文件的能力，是记忆更新、代码生成等功能的基础
- **实现逻辑**：自定义工具类 `SafeWriteFileTool`，继承 LangChain `BaseTool`。支持创建新文件和覆盖已有文件，自动创建不存在的父目录
- **安全约束**：路径沙箱化（禁止 `../` 穿越）、禁写名单（`.env`、`.git`、`.gitignore`）、单次写入不超过 1MB
- **工具名称**：`write_file`

---

## 三、Agent Skills 系统

### 3.1 基础功能介绍

Mini-OpenClaw 的 Agent Skills 遵循 **"Instruction-following"（指令遵循）范式**，而非传统的 "Function-calling" 范式。Skills 本质上是教会 Agent 如何使用基础工具去完成任务的说明书，而不是预先写好的 Python 函数。

Agent Skills 以文件夹形式存在于 `backend/skills/` 目录下。

### 3.2 载入与执行流程

#### 读取流程 (Bootstrap)

在 Agent 启动或会话开始时，系统扫描 `skills` 文件夹，读取每个 `SKILL.md` 的元数据（Frontmatter），并将其汇总生成 `SKILLS_SNAPSHOT.md`。

**SKILLS_SNAPSHOT.md 示例：**

```xml
<available_skills>  
  <skill>  
    <name>get_weather</name>  
    <description>获取指定城市的实时天气信息</description>  
    <location>./backend/skills/get_weather/SKILL.md</location> 
  </skill>
</available_skills>
```

#### 调用流程 (Execution)

1. **感知**：Agent 在 System Prompt 中看到 `available_skills` 列表
2. **决策**：当用户请求"查询北京天气"时，Agent 发现 `get_weather` 技能匹配
3. **行动**：Agent 调用 `read_file(path="./backend/skills/get_weather/SKILL.md")`
4. **学习与执行**：Agent 读取 Markdown 内容，理解操作步骤，然后动态调用 Core Tools 来完成任务

### 3.3 Skills 依赖检查

技能加载时自动检查所需的环境依赖（通过 `SKILL.md` Frontmatter 声明）：

- **bins**：所需的系统命令行工具（如 `curl`、`jq`、`ffmpeg`）
- **env**：所需的环境变量（如 `WEATHER_API_KEY`）

依赖满足时正常加载；依赖缺失时标记 `enabled=false` 并附加警告信息。

- **实现位置**：`backend/skills/skill_manager.py`（`SkillManager` 类）

---

## 四、对话记忆管理系统

### 4.1 本地优先原则

所有记忆文件（Markdown/JSON）均存储在本地文件系统，确保完全的数据主权和可解释性。

### 4.2 系统提示词 (System Prompt) 构成

System Prompt 由以下 **7 部分**动态拼接而成（按顺序）：

1. `SKILLS_SNAPSHOT.md`（能力列表）
2. `SOUL.md`（核心设定）
3. `IDENTITY.md`（自我认知）
4. `USER.md`（用户画像）
5. `AGENTS.md`（行为准则 & 记忆操作指南）
6. `TOOLS.md`（工具使用规范）
7. `MEMORY.md`（长期记忆）

**截断策略**：单文件超 20k 字符时截断并添加 `...[truncated]` 标识；完整 Prompt 超过 `max_system_prompt_chars` 限制时整体截断。

- **实现位置**：`backend/graph/prompt_builder.py`（`PromptBuilder` 类）

### 4.3 AGENTS.md 的默认配置

必须在初始化时生成包含明确指令的 `AGENTS.md`，告知 Agent 它是通过"阅读文件"来学习技能的。

必须包含的元指令：技能调用协议（先 `read_file` 读取 `SKILL.md`，再按指示执行）和记忆协议。

### 4.4 会话存储 (Sessions)

- **路径**：`backend/sessions/{session_name}.json`
- **格式**：标准 JSON 数组，包含 `user`、`assistant`、`tool` 类型的完整消息记录
- **实现位置**：`backend/sessions/session_manager.py`（`SessionManager` 类）

---

## 五、后端 API 接口规范 (FastAPI)

后端服务作为独立进程运行，负责 Agent 逻辑、文件读写和状态管理。

- **服务端口**：`8002`
- **基础 URL**：`http://localhost:8002`
- **流式输出**：核心对话接口 `POST /api/chat` 支持 SSE (Server-Sent Events)，实时推送 Agent 的思考过程和最终回复

> 完整接口列表见[第九章 API 接口汇总](#九后端-api-接口汇总)。

---

## 六、前端开发要求

### 6.1 设计理念与布局架构

前端采用 **IDE（集成开发环境）风格**，三栏式布局：

- **左侧 (Sidebar)**：导航（Chat / Memory / Skills / Gateway）+ 会话列表。四个 Tab 使用 grid 四列布局，分别对应对话、记忆文件、技能列表和 Gateway 任务管理
- **中间 (Stage)**：对话流 + 思考链可视化（Collapsible Thoughts）。当 Gateway tab 激活时，Stage 区域展示 `GatewayPanel` 组件，提供定时任务的可视化管理界面
- **右侧 (Inspector)**：Monaco Editor，用于实时查看/编辑正在使用的 `SKILL.md` 或 `MEMORY.md`

### 6.2 技术栈

- **框架**：Next.js 14+（App Router）、TypeScript
- **UI**：Shadcn/UI、Tailwind CSS、Lucide Icons
- **Editor**：Monaco Editor（配置为 Light Theme）

### 6.3 UI/UX 风格规范

- **色调**：浅色 Apple 风格（Frosty Glass）
- **背景**：纯白/极浅灰（`#fafafa`），高透毛玻璃效果
- **强调色**：克莱因蓝（Klein Blue）或活力橙
- **导航栏**：顶部固定，半透明，左中显示 "mini OpenClaw"

---

## 七、项目目录结构

```
mini-openclaw/
├── backend/                        # FastAPI + LangChain/LangGraph
│   ├── app.py                      # 入口文件 (Port 8002)
│   ├── config.py                   # 全局配置文件
│   ├── memory/                     # 记忆存储
│   │   ├── logs/                   # Daily logs（日志归档）
│   │   ├── reflector.py            # 记忆回顾引擎
│   │   ├── MEMORY.md               # 长期记忆
│   │   └── HEARTBEAT.md            # 待办任务清单
│   ├── sessions/                   # 会话管理
│   │   ├── session_manager.py      # 会话管理器（含压缩/清理）
│   │   └── *.json                  # JSON 会话记录
│   ├── skills/                     # Agent Skills 文件夹
│   │   ├── skill_manager.py        # 技能管理器（含依赖检查）
│   │   └── get_weather/
│   │       └── SKILL.md
│   ├── workspace/                  # System Prompts
│   │   ├── SOUL.md                 # 核心设定
│   │   ├── IDENTITY.md             # 自我认知
│   │   ├── AGENTS.md               # 行为准则
│   │   ├── TOOLS.md                # 工具使用规范
│   │   └── USER.md                 # 用户画像
│   ├── tools/                      # Core Tools 实现（6 个）
│   │   ├── terminal_tool.py        # 命令行工具
│   │   ├── python_repl_tool.py     # Python 解释器
│   │   ├── fetch_url_tool.py       # 网络信息获取
│   │   ├── read_file_tool.py       # 文件读取
│   │   ├── write_file_tool.py      # 文件写入
│   │   └── search_knowledge_tool.py # RAG 检索
│   ├── graph/                      # Agent 引擎
│   │   ├── agent.py                # Agent 核心逻辑
│   │   └── prompt_builder.py       # System Prompt 拼接
│   ├── heartbeat/                  # 心跳自我进化
│   │   └── engine.py               # HeartbeatEngine
│   ├── channels/                   # 外部通道
│   │   ├── base.py                 # BaseChannel 抽象基类 + 数据模型
│   │   ├── webhook.py              # Webhook 通道实现
│   │   ├── wechat.py              # 企业微信通道实现
│   │   └── channel_manager.py      # 通道管理器
│   ├── gateway/                    # 异步任务网关
│   │   ├── job_scheduler.py        # JobScheduler 任务调度器
│   │   └── JOB.JSON                # 定时任务配置
│   ├── storage/                    # 持久化存储
│   │   └── job_runs/               # 任务运行记录
│   └── requirements.txt
│
├── frontend/                       # Next.js 14+
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx            # 主页（含斜杠命令处理）
│   │   ├── components/
│   │   │   ├── chat/               # 聊天组件（含 Sidebar）
│   │   │   ├── editor/             # Monaco Wrapper
│   │   │   ├── gateway/            # Gateway 管理组件
│   │   │   │   └── GatewayPanel.tsx # 定时任务管理面板
│   │   │   └── Navbar.tsx          # 顶部导航栏
│   │   └── lib/
│   │       └── api.ts              # API 封装（含会话管理方法）
│   └── package.json
│
└── README.md
```

---

## 八、扩展功能规格说明

### 8.1 Session Compact（会话压缩）

- **功能描述**：当对话过长时，自动或手动将历史消息压缩为摘要，防止 Token 溢出
- **触发方式**：手动（`/compact` 命令）或自动（消息数超过阈值时）
- **压缩逻辑**：调用 LLM 生成结构化摘要，保留最近 N 条消息不被压缩，摘要以 `system` 消息形式插入会话头部
- **配置项**：`compact_threshold_messages` (40)、`compact_keep_recent_messages` (6)、`compact_summary_max_chars` (2000)
- **实现位置**：`backend/sessions/session_manager.py`、`backend/graph/agent.py`

### 8.2 Chat Commands（斜杠命令系统）

- **支持的命令**：`/new`（新建会话）、`/reset`（重置会话）、`/compact`（手动压缩）、`/status`（查看状态）、`/help`（帮助）
- **前端处理**：命令在前端拦截，不发送到后端，结果以系统消息形式展示
- **实现位置**：`frontend/src/app/page.tsx`

### 8.3 Model Failover（模型故障转移）

- **功能描述**：支持配置多个备选模型，主模型失败时自动降级
- **降级策略**：按 `fallback_models` 列表顺序依次尝试，每个模型最多重试 `model_max_retries` 次，间隔 `model_retry_delay` 秒
- **配置项**：`fallback_models` (list)、`model_max_retries` (2)、`model_retry_delay` (1.0)
- **实现位置**：`backend/graph/agent.py`

### 8.4 Session Pruning（会话自动清理）

- **清理策略**：阶段一按时间清理（超过 `session_max_age_days` 天），阶段二按数量清理（超过 `session_max_count` 个）
- **配置项**：`session_max_age_days` (30)、`session_max_count` (500)
- **实现位置**：`backend/sessions/session_manager.py`

### 8.5 TOOLS.md 工具使用规范

- **功能描述**：在 System Prompt 中新增 `TOOLS.md`，描述所有内置工具的使用规范和安全约束
- **存储位置**：`backend/workspace/TOOLS.md`

### 8.6 Health Check API 增强

- **返回字段**：`status`、`version`、`model`、`skills_count`、`sessions_count`、`memory_file_exists`、`uptime_info`、`heartbeat_enabled`、`last_heartbeat`、`gateway`（含 `total`/`enabled`/`disabled`/`schedule_types`/`last_run_status`/`jobs_with_errors` 统计）
- **实现位置**：`backend/app.py`（`GET /api/health`）

### 8.7 Heartbeat 心跳自我进化机制

- **功能描述**：Agent 在空闲时主动执行蒸馏与自省，从历史对话中提取经验、优化技能、更新长期记忆
- **心跳执行流程**：日志收集 → LLM 自省 → 记忆更新（`MEMORY.md` + `USER.md`）→ 待办检查（`HEARTBEAT.md`）
- **触发方式**：定时触发（默认每 24 小时）或手动触发（`POST /api/heartbeat/trigger`）
- **配置项**：`heartbeat_enabled` (true)、`heartbeat_interval_hours` (24)、`heartbeat_log_review_days` (7)
- **实现位置**：`backend/heartbeat/engine.py`（`HeartbeatEngine` 类）

### 8.8 Channel 外部通道抽象层

- **架构设计**：
  - 抽象基类 `BaseChannel`（`backend/channels/base.py`），定义 `receive()`、`send()`、`authenticate()` 三个抽象方法
  - 数据模型：`IncomingMessage`（入站消息）和 `OutgoingMessage`（出站消息）
  - 通道管理器 `ChannelManager`（`backend/channels/channel_manager.py`），统一管理通道注册与消息处理流程
- **内置实现**：
  - `WebhookChannel`（`backend/channels/webhook.py`）：通用 Webhook 通道，支持请求头或请求体两种 Secret 验证方式
  - `WeChatChannel`（`backend/channels/wechat.py`）：企业微信通道，支持 AES-CBC-256 消息加解密、URL 验证、多消息类型解析、access_token 自动缓存刷新、按需注册
- **扩展规范**：新渠道只需继承 `BaseChannel` 并实现三个抽象方法，在 `ChannelManager` 中注册即可
- **配置项**：`webhook_secret`、`webhook_callback_url`、`wechat_corpid`、`wechat_corpsecret`、`wechat_agent_id`、`wechat_token`、`wechat_encoding_aes_key`
- **依赖项**：`pycryptodome>=3.20.0`（企业微信 AES 加解密）

### 8.9 Gateway 异步任务唤醒机制

- **功能描述**：通过心跳与任务文件实现异步任务调度，支持多种调度类型、完整 CRUD 管理、执行状态跟踪和运行日志
- **核心文件**：`HEARTBEAT.md`（待办任务清单）、`JOB.JSON`（定时任务配置）
- **调度类型**（`schedule_type`）：
  - `cron`：标准 5 段 cron 表达式，使用 `croniter` 库解析
  - `every`：间隔重复执行，支持 `s`/`m`/`h`/`d` 单位（如 `"30s"`、`"5m"`、`"2h"`）
  - `at`：一次性定时执行，ISO 8601 时间字符串，执行后自动禁用
- **执行状态跟踪**：`last_run_status`（ok/error）、`last_error`、`last_duration_ms`、`consecutive_errors`
- **重试机制**：每个任务可独立配置 `max_retries` 和 `retry_delay_seconds`
- **并发控制**：`gateway_max_concurrent_runs` 限制同时运行的任务数（默认 3）
- **运行日志**：保存到 `storage/job_runs/{job_id}/{timestamp}.json`，每个任务保留最近 20 条，同时写入 Daily Log
- **Webhook 触发**：`POST /api/gateway/webhook/trigger` 支持触发一次性任务
- **前端管理**：Sidebar Gateway tab + `GatewayPanel` 组件，支持任务的可视化 CRUD 操作
- **配置项**：`gateway_max_concurrent_runs` (3)、`gateway_job_max_retries` (2)、`gateway_job_retry_delay` (1.0)
- **依赖项**：`croniter`
- **实现位置**：`backend/gateway/job_scheduler.py`（`JobScheduler` 类）

### 8.10 会话结束自动记忆回顾

- **功能描述**：会话消息数达到阈值时，自动回顾对话内容，提取关键信息更新到长期记忆文件
- **回顾流程**：LLM 分析对话 → 项目级记忆追加到 `MEMORY.md` → 用户级记忆追加到 `USER.md` → 摘要归档到 `memory/logs/{date}.md`
- **执行方式**：异步后台任务（`asyncio.create_task`），不阻塞用户
- **配置项**：`auto_reflect_enabled` (true)、`auto_reflect_min_messages` (4)
- **实现位置**：`backend/memory/reflector.py`（`MemoryReflector` 类）

---

## 九、后端 API 接口汇总

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 系统健康检查（含详细状态、Gateway 统计） |
| `/api/chat` | POST | 核心对话接口（SSE 流式输出） |
| `/api/file` | GET | 读取指定文件内容 |
| `/api/file` | PUT | 保存文件修改 |
| `/api/files/tree` | GET | 列出指定目录下的文件树 |
| `/api/sessions` | GET | 获取所有历史会话列表 |
| `/api/sessions/{id}` | GET | 获取指定会话详情 |
| `/api/sessions/{id}` | DELETE | 删除指定会话 |
| `/api/sessions/{id}/compact` | POST | 压缩指定会话 |
| `/api/sessions/{id}/reset` | POST | 重置指定会话 |
| `/api/sessions/{id}/status` | GET | 查看会话状态 |
| `/api/sessions/prune` | POST | 自动清理过期会话 |
| `/api/skills` | GET | 获取所有可用技能列表 |
| `/api/skills/refresh` | POST | 刷新技能列表 |
| `/api/heartbeat/trigger` | POST | 手动触发心跳自省 |
| `/api/heartbeat/status` | GET | 查看心跳状态 |
| `/api/gateway/jobs` | GET | 获取所有定时任务列表及状态 |
| `/api/gateway/jobs` | POST | 添加新的定时任务 |
| `/api/gateway/jobs/{id}` | GET | 获取指定任务详情 |
| `/api/gateway/jobs/{id}` | PUT | 更新指定任务配置 |
| `/api/gateway/jobs/{id}` | DELETE | 删除指定任务 |
| `/api/gateway/jobs/{id}/enable` | POST | 启用指定任务 |
| `/api/gateway/jobs/{id}/disable` | POST | 禁用指定任务 |
| `/api/gateway/jobs/{id}/execute` | POST | 手动触发执行指定任务 |
| `/api/gateway/webhook/trigger` | POST | 通过 Webhook 触发一次性任务 |
| `/api/channels` | GET | 列出所有已注册的通道类型 |
| `/api/channels/webhook` | POST | 接收外部 Webhook 回调 |
| `/api/channels/wechat` | GET | 企业微信回调 URL 验证 |
| `/api/channels/wechat` | POST | 接收企业微信推送消息 |

---

## 十、设计原则

### 10.1 SOLID 原则

- **单一职责原则（SRP）**：每个模块职责清晰——`skill_manager.py` 只负责技能扫描与元数据管理、`session_manager.py` 只负责会话 CRUD 与压缩清理、`prompt_builder.py` 只负责 System Prompt 拼接、`reflector.py` 只负责记忆反思、`job_scheduler.py` 只负责任务调度与执行、`channel_manager.py` 只负责通道注册与消息路由，互不耦合
- **开闭原则（OCP）**：新增 Skill 无需修改框架代码，只需在 `backend/skills/` 下新建文件夹并放入 `SKILL.md` 即可被 `SkillManager` 自动发现和加载；新增外部通道只需继承 `BaseChannel` 并在 `ChannelManager` 中注册，无需修改已有通道代码；新增调度类型只需在 `JobScheduler._compute_next_run()` 中扩展分支
- **里氏替换原则（LSP）**：所有外部通道（`WebhookChannel`、`WeChatChannel`）均可替换 `BaseChannel` 使用，`ChannelManager` 对具体通道实现无感知，统一通过 `receive()`、`send()`、`authenticate()` 三个抽象方法交互
- **接口隔离原则（ISP）**：`BaseChannel` 抽象基类仅定义三个必要方法，不强制实现无关接口；每个 Core Tool 通过独立的 `create_xxx_tool()` 工厂函数创建，不依赖统一的工具基类
- **依赖倒置原则（DIP）**：`ChannelManager` 依赖 `BaseChannel` 抽象而非具体通道实现；`AgentEngine` 通过 `PromptBuilder` 接口获取 System Prompt，不直接读取文件；`app.py` 通过全局单例与各模块交互，不直接依赖内部实现细节

### 10.2 设计模式

- **单例模式（Singleton）**：核心模块均通过模块级全局实例实现单例（`agent_engine`、`session_manager`、`skill_manager`、`job_scheduler`、`channel_manager`、`heartbeat_engine`、`reflector`、`prompt_builder`），确保全局状态一致
- **工厂模式（Factory）**：每个 Core Tool 通过 `create_xxx_tool()` 工厂函数创建，屏蔽工具初始化和安全配置的细节，`AgentEngine._create_tools()` 统一调用
- **策略模式（Strategy）**：`JobScheduler` 通过 `schedule_type` 动态选择不同的调度策略；`AgentEngine` 通过 `fallback_models` 实现模型故障转移策略链
- **构建器模式（Builder）**：`PromptBuilder` 将 System Prompt 构建分解为 7 个独立步骤，最终通过 `build_system_prompt()` 组装，支持灵活扩展和截断控制
- **观察者模式（Observer）**：`_auto_reflect()` 在会话消息达到阈值时异步触发记忆回顾，`_heartbeat_background_task()` 定期触发自省，均为事件驱动的异步回调机制
- **模板方法模式（Template Method）**：`BaseChannel` 定义通道骨架接口，`ChannelManager.process_incoming()` 编排完整的消息处理流程（验证 → 接收 → Agent 回复 → 发送），具体实现由各通道子类填充

### 10.3 架构原则

- **声明式优先（Declarative First）**：Skill 定义通过 `SKILL.md`（YAML front matter + Markdown）声明，无需编写 Python 类，降低扩展门槛
- **约定优于配置（Convention over Configuration）**：Skill 文件夹、System Prompt 文件、会话文件均遵循固定命名和目录约定，无需额外配置
- **关注点分离（Separation of Concerns）**：Agent 编排、工具实现、技能管理、记忆系统、会话管理、外部通道、任务调度、心跳引擎各自独立为子模块
- **文件即数据库（File as Database）**：所有持久化数据均以本地文件形式存储（JSON/Markdown），不引入重型数据库依赖
- **异步优先（Async First）**：所有 I/O 密集型操作使用 `async/await` 异步处理；记忆回顾通过 `asyncio.create_task()` 在后台执行，不阻塞用户

### 10.4 工程实践原则

- **优雅降级（Graceful Degradation）**：主模型失败时自动尝试备选模型；技能依赖缺失时标记降级而非阻止加载；JSON 解析失败时返回空结果而非崩溃
- **安全沙箱（Security Sandbox）**：所有文件操作工具通过 `root_dir` 限制范围，禁止路径穿越，维护禁写名单和高危指令黑名单
- **配置集中化（Centralized Configuration）**：所有配置项集中在 `backend/config.py` 的 `Settings` 类中，基于 `pydantic-settings` 实现，支持 `.env` 文件和环境变量加载
- **可观测性（Observability）**：所有模块使用 `logging` 记录关键操作；Gateway 运行记录持久化；健康检查接口返回详细系统状态
- **渐进式增强（Progressive Enhancement）**：企业微信通道按需注册；心跳引擎可配置开关；自动记忆回顾可关闭；所有扩展功能均可独立启用或禁用
