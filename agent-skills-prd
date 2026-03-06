
# Agent-Skills 产品需求文档

## 项目概述

Agent-Skills 是一个基于 **LangChain v1 + LangGraph v1 + 通义千问** 构建的智能助理 Agent 框架，参考 **Anthropic Claude Skills 的 Meta-Tool 架构**，实现声明式 Skill 定义、运行时动态加载和 Skill 自动生成。

## 核心功能

### 1. Meta-Tool 动态 Skills
- Agent 初始只暴露 `skill_selector` 元工具，LLM 根据用户意图自主决定激活哪个 Skill
- 激活后才能使用该 Skill 的专业工具，实现按需加载
- 通过 `before_model` / `after_model` / `wrap_tool_call` 三个中间件钩子协同工作

### 2. Skill 自动生成
- 当用户需求没有匹配的 Skill 时，系统自动评估是否为常见通用场景
- 评估通过后，调用 LLM 生成完整 Skill 文件夹（SKILL.md + scripts/tools.py）
- 自动安装所需第三方依赖（pip install）
- 热注册到 Registry，当轮对话即可使用
- 持久化存储到文件系统，重启后自动发现并直接可用
- 支持纯标准库场景（编码解码、哈希计算等）和常见第三方库场景（PPT 生成、图片处理等）

### 3. 声明式 Skill 定义
- 遵循 Anthropic Claude Skills 范式，每个 Skill 是独立文件夹
- 通过 `SKILL.md`（YAML front matter + Markdown 正文）声明流程规则
- `scripts/` 存放工具函数，`references/` 存放经验沉淀，`assets/` 存放其他材料
- 无需编写 Python 类，纯声明式定义

### 4. 内置 Skill
- **智能数据分析**：CSV 数据加载、统计分析、图表可视化（matplotlib）、条件查询
- **智能对话**：时间查询、文本摘要、数学计算

### 5. 多会话管理
- 基于 LangGraph MemorySaver 的会话隔离和记忆持久化
- 支持创建、切换、列出多个独立会话

### 6. 实时工具调用日志
- 流式输出中区分 Skill 激活事件和工具调用事件
- 明确显示 LLM 实际调用的工具名称

## 技术架构

### 技术栈
- **Python 3.10+**
- **LangChain v1**（>=1.0.0）：`create_agent` API、`AgentMiddleware` 钩子机制、Tool 定义
- **LangGraph v1**（>=1.0.0）：StateGraph、ToolNode、MemorySaver
- **通义千问（Qwen）**：通过 DashScope OpenAI 兼容端点接入
- **langchain-openai**：OpenAI 协议兼容模型集成
- **PyYAML**：SKILL.md 的 YAML front matter 解析
- **pandas + matplotlib**：数据分析与可视化

### 项目结构
```
agent-skills/
├── main.py                    # CLI 交互入口（流式输出 + 命令处理）
├── requirements.txt           # 依赖管理
├── .env.example               # 环境变量模板
├── README.md                  # 项目说明文档
├── _export_graph.py           # LangGraph 流程图导出脚本
├── agent/
│   ├── __init__.py            # 包初始化
│   ├── config.py              # 全局配置（AgentConfig）
│   ├── llm.py                 # LLM 工厂（通义千问 via DashScope）
│   ├── graph.py               # Agent 构建（create_agent + AgentMiddleware）
│   ├── memory/
│   │   ├── __init__.py
│   │   └── memory_manager.py  # 记忆管理器（会话隔离 + MemorySaver）
│   └── skills/
│       ├── __init__.py
│       ├── base.py            # Skill 基类（BaseSkill + FileBasedSkill）
│       ├── loader.py          # Skill 文件夹加载器（YAML front matter 解析）
│       ├── registry.py        # Skill 注册中心（自动发现 + 热注册）
│       ├── middleware.py      # AgentMiddleware（Meta-Tool + 状态管理 + wrap_tool_call）
│       ├── generator.py       # Skill 自动生成器（评估 + 代码生成 + 依赖安装）
│       │
│       ├── conversation/      # 📦 智能对话 Skill（内置，手动创建）
│       │   ├── SKILL.md       #    Skill 声明文件（YAML front matter + 流程规则）
│       │   ├── __init__.py
│       │   ├── scripts/
│       │   │   ├── __init__.py
│       │   │   └── tools.py   #    工具函数（时间查询、文本摘要、数学计算）
│       │   ├── references/    #    经验沉淀（.gitkeep 占位）
│       │   └── assets/        #    其他材料（.gitkeep 占位）
│       │
│       ├── data_analysis/     # 📦 数据分析 Skill（内置，手动创建）
│       │   ├── SKILL.md       #    Skill 声明文件（YAML front matter + 流程规则）
│       │   ├── __init__.py
│       │   ├── scripts/
│       │   │   ├── __init__.py
│       │   │   └── tools.py   #    工具函数（CSV 加载、统计分析、图表可视化、条件查询）
│       │   ├── references/    #    经验沉淀（.gitkeep 占位）
│       │   └── assets/        #    其他材料（.gitkeep 占位）
│       │
│       └── ppt_maker/         # 📦 PPT 制作 Skill（自动生成示例）
│           ├── SKILL.md       #    Skill 声明文件（由 SkillGenerator 自动生成）
│           ├── __init__.py
│           ├── scripts/
│           │   ├── __init__.py
│           │   └── tools.py   #    工具函数（由 LLM 自动生成，使用 python-pptx）
│           ├── references/    #    经验沉淀（空）
│           └── assets/        #    其他材料（空）
└── info/
    ├── prd.md                 # 本文档
    └── langgraph_flow.png     # LangGraph 流程图
```

### 核心机制

#### Meta-Tool 架构
```
用户输入 → before_model（注入约束/指令）→ LLM 决策
  ├─ 调用 skill_selector → 激活 Skill → after_model 更新 State
  ├─ 调用专业工具 → wrap_tool_call 拦截执行 → 返回结果
  └─ 直接回答 → 输出文本
```

#### Skill 自动生成流程
```
skill_selector 找不到 Skill
  → SkillGenerator.assess()：LLM 评估是否可自动生成
  → SkillGenerator.generate()：LLM 生成 SKILL.md + tools.py
  → _install_packages()：自动 pip install 依赖
  → _write_skill_folder()：写入文件系统（持久化）
  → SkillLoader.load_from_folder()：加载为 FileBasedSkill
  → Registry.hot_register()：热注册
  → wrap_tool_call：拦截并执行动态工具
```

#### wrap_tool_call 动态工具执行
- LangChain v1 `create_agent` 的 ToolNode 在初始化时固定工具注册表
- 运行时新增的工具（自动生成）不在 ToolNode 注册表中
- 通过 `AgentMiddleware.wrap_tool_call` 钩子拦截未注册工具调用
- 从 Registry 中查找动态注册的工具实例，通过 `request.override(tool=...)` 传入执行

## 扩展 Skill 的三种方式

### 方式一：Skill 文件夹（推荐）
创建 `agent/skills/{skill_name}/` 文件夹，包含 `SKILL.md` 和 `scripts/tools.py`，重启后自动发现。

### 方式二：Python 类
继承 `BaseSkill`，实现 `manifest()`、`get_tools()`、`get_system_prompt_fragment()` 方法。

### 方式三：自动生成（零代码）
直接向 Agent 提出需求，系统自动判断并生成对应 Skill，持久化存储供未来复用。

## 设计原则

### SOLID 原则

- **单一职责原则（SRP）**：每个模块职责清晰——`loader.py` 只负责加载、`registry.py` 只负责注册与发现、`generator.py` 只负责生成、`middleware.py` 只负责中间件钩子逻辑，互不耦合
- **开闭原则（OCP）**：新增 Skill 无需修改框架代码，只需在 `agent/skills/` 下新建文件夹即可自动发现和加载；自动生成的 Skill 同样遵循此范式
- **里氏替换原则（LSP）**：所有 Skill（手动创建的 `FileBasedSkill`、自动生成的 Skill）均可替换 `BaseSkill` 使用，Registry 和 Middleware 对具体实现无感知
- **接口隔离原则（ISP）**：`BaseSkill` 抽象基类仅定义三个必要方法（`manifest()`、`get_tools()`、`get_system_prompt_fragment()`），不强制实现无关接口
- **依赖倒置原则（DIP）**：`Registry`、`Middleware` 依赖 `BaseSkill` 抽象而非具体实现；`graph.py` 通过 `AgentMiddleware` 接口与 Skill 系统交互，不直接依赖具体 Skill 类

### 设计模式

- **注册表模式（Registry）**：`SkillRegistry` 作为全局注册中心，统一管理 Skill 的注册、发现和查找，支持启动时自动发现和运行时热注册
- **工厂模式（Factory）**：`SkillLoader` 将文件夹结构统一转换为 `FileBasedSkill` 实例，屏蔽加载细节
- **中间件模式（Middleware）**：通过 `before_model` / `after_model` / `wrap_tool_call` 三个钩子实现 AOP 式的横切关注点，无侵入地增强 Agent 行为
- **策略模式（Strategy）**：不同 Skill 提供不同的工具集和系统提示，Agent 在运行时根据用户意图动态切换策略
- **模板方法模式（Template Method）**：`BaseSkill` 定义 Skill 的骨架接口，具体实现由子类（`FileBasedSkill`）或自动生成的 Skill 填充

### 架构原则

- **声明式优先（Declarative First）**：Skill 定义通过 `SKILL.md`（YAML front matter + Markdown）声明，无需编写 Python 类，降低扩展门槛
- **约定优于配置（Convention over Configuration）**：Skill 文件夹遵循固定目录结构约定（`SKILL.md` + `scripts/tools.py` + `references/` + `assets/`），无需额外配置文件
- **按需加载（Lazy Loading）**：工具不预加载到 LLM 上下文，通过 Meta-Tool 按需激活，减少 Token 消耗和上下文污染
- **关注点分离（Separation of Concerns）**：Skill 定义（SKILL.md）、工具实现（tools.py）、经验沉淀（references/）、静态资源（assets/）各自独立，职责清晰
- **模块化设计（Modular Design）**：每个文件职责单一，模块间通过明确的接口通信，便于独立开发、测试和维护

### 工程实践原则

- **持久化生成（Persistent Generation）**：自动生成的 Skill 写入文件系统而非仅存于内存，重启后自动可用，避免重复生成
- **热插拔（Hot-Pluggable）**：运行时通过 `hot_register()` 动态注册新 Skill，无需重启 Agent 即可使用
- **优雅降级（Graceful Degradation）**：当 `wrap_tool_call` 拦截到未注册工具时，返回明确的错误提示而非崩溃；Skill 自动生成评估失败时，友好告知用户而非静默失败
- **幂等性（Idempotency）**：Skill 注册操作幂等，重复注册同名 Skill 不会产生副作用
- **自描述性（Self-Describing）**：每个 Skill 通过 `SKILL.md` 自描述其名称、描述、触发关键词和使用流程，Registry 自动解析，无需外部元数据
- **最小权限原则（Least Privilege）**：Agent 初始仅暴露 `skill_selector` 一个元工具，只有在 LLM 明确激活后才注入对应 Skill 的专业工具，避免工具滥用
