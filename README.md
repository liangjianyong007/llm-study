# llm-study
人工智能技术点
  人工智能技术点	内容	资料整理
数学基础	"线性代数
微积分
概率论与梳理统计
最优化理论
信息论"	
python体系	"1、python基础
2、框架：tensorFlow、PyTorch、numpy、pandas、
Scipy、Matplotlib、Scikit-Learn等
3、CUDA GPU加速原理
4、pytorch CUDA实际应用案例
5、web框架：Flask、Django"	
NLP	自然语言理解、自然语言解释、Word2Vec、分词等	
机器学习	监督学习、无监督学习、半监督学习、自监督学习	
深度学习	"卷积神经网络（CNN）、循环神经网络（RNN）、
LSTM（长短期记忆）、生成对抗网络（GAN等等"	
强化学习	Q-Learning、深度Q网络（DQN、蒙特卡洛树搜索（MCTS等等	
"大模型原理、开发与
训练实战"	"1、大模型原理、理论&关键论文
2、大模型架构、模型的发展
3、模型开源代码解析或手写一个简版模型等
4、大模型开发、预训练、测试等
5、推理（COT/T0T），TensorRT、Triton、tf-serving"	
transformer框架	BERT、GPT3	
bert	基于 Transformer 的预训练语言模型，双向训练、预训练+微调	
prompt	角色扮演型、问答型、总结型等等	
springAI	整体架构、组件等，如NL2SQL	
RAG	向量数据库、分词、检索召回等	
Agent（智能体）	AutoGen、规划、行动、记忆，react、plan_excuter等	
A2A与MCP	通信协议标准、安全与权限控制等	
functioncall	机制和原理	
langchain/langchain4J	整体架构、组件等	
langgraph/langgraph4J	整体架构、组件等	
llamaindex	专注数据接入与 RAG 优化案例	
本地部署	Ollama、LM Studio、硬件要求、企业级部署等	
微调	"需要包含pipeline以及LORA，QLORA，VLLM
DPO等核心的技术手、SFT微调、RL微调
Unsloth：高效微调框架
vLLM：模型调度框架，用于验证微调后模型效果
EvalScope：模型评测框架，用于对比微调前后模型性能
wandb：模型训练数据在线记录工具，用于保存模型训练
过程中损失值的变化情况"	
蒸馏	标准知识蒸馏、特征蒸馏	
AI coding	copilot，clause code，qoder等	<img width="717" height="688" alt="image" src="https://github.com/user-attachments/assets/7e7c3f87-65dc-467e-9144-f24cef3855a7" />

　
快速读懂开源
1. 快速读懂开源代码deepwiki：可以将 GitHub 链接中的“github.com”替换为“deepwiki.com
2. 基础算法：GitHub地址 github.com/algorithm-visualizer/algorithm-visualizer

前置基础
1. 人工智能数学基础:
● 代码： https://github.com/bob329/aimath  
● 书籍：https://github.com/datawhalechina/math-for-ai?tab=readme-ov-file
2. python：
● 基础：https://github.com/walter201230/Python
● 基础算法python实现：https://github.com/TheAlgorithms/Python
3. pytorch：https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
● pytorch examples：https://github.com/pytorch/examples
    ○ pytorch Tutorial：https://github.com/yunjey/pytorch-tutorial
4. numpy：https://github.com/numpy/numpy-tutorials
5. pandas：https://github.com/tdpetrou/Learn-Pandas
6. matplotlib：https://github.com/rougier/matplotlib-tutorial
7. scikit-learn：https://github.com/jakevdp/sklearn_tutorial
零、机器学习&深度学习
1. 幂次：https://mici.jiqishidai.com/site/vod?course_id=5&fir_floor=2&sec_floor=0&course_tp=necessary&r=7147.428788022649
2. 李沐体系课程：https://github.com/mli
3. 李宏毅：https://github.com/datawhalechina/leedl-tutorial
4. 幂次学院：https://mici.jiqishidai.com/site/my_course_list
5. 阿里云机器学习：https://www.aliyun.com/resources?userCode=okjhlpr5
6. transformer：https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md
7. transformer（pytouch）：https://github.com/hyunwoongko/transformer
8. transformers：
●   GitHub：https://github.com/huggingface/transformers
●   文档：https://huggingface.co/docs/transformers

一、大模型学习
  词表库、位置向量库、分词、词向量、
1. 大模型从硬件到软件学习：https://github.com/Infrasys-AI/AIInfra/tree/main?tab=readme-ov-file
2. openai-cookbook：https://cookbook.openai.com/topic/agents
3. 动手学大模型-大模型全流程开发（课程）：https://github.com/Lordog/dive-into-llms/tree/main
4. 动手学习大模型（课程）：https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN?tab=readme-ov-file
5. 大语言模型基础：https://github.com/ZJU-LLMs/Foundations-of-LLMs/blob/main/readme.md
6. 0到1手写大模型：https://waylandzhang.github.io/en/introduction.html   https://github.com/bbruceyuan/LLMs-Zero-to-Hero  https://github.com/REXWindW/my_llm/blob/main/readme.md
7. 大模型推理：https://github.com/Infrasys-AI/AIInfra/tree/main
8. 大模型预训练与微调：https://github.com/datawhalechina/self-llm。医疗大模型型：https://github.com/shibing624/MedicalGPT/blob/main/README.md
9. 预训练、微调、强化学习:https://i232t6gteo.feishu.cn/docx/NUEDdyv12op2LKxQKjMcHJUbnNd
10. 端到端训练大模型实战经验，涵盖预训练、后训练及基础设施搭建：https://huggingface.co/spaces/HuggingFaceTB/smol-playbook-toc
11. Smol培训手册，构建世界级LLM的秘诀-Hugging Face：https://uone.alibaba-inc.com/uknow/reports/130980?spm=25c880c9.37285d8b.0.0.498db97baVkUOc&utm_source=aidate
12. 0到1写大模型：https://www.youtube.com/watch?v=F53Tt_vNLdg   https://github.com/rasbt/LLMs-from-scratch
13. 
二、propmt-Engineering
1. Prompt-Engineering-Guide：https://github.com/dair-ai/Prompt-Engineering-Guide
2. prompt指导中文版本：https://www.promptingguide.ai/zh
3. propmt：https://www.promptingguide.ai/introduction
4. propmt lear：https://www.learnprompting.org
5. prompt 市场：https://promptbase.com
6. 论文研究：https://arxiv.org/abs/2005.14165
7. 社区：https://www.reddit.com/r/PromptEngineering
8. 实战课程：https://www.coursera.org/learn/chatgpt-prompt-engineering
9. 开发框架与模板：https://langchain.readthedocs.io
10. prompt优化GitHub地址：https://github.com/linshenkx/prompt-optimizer
11. prompt-eng ：https://github.com/anthropics/prompt-eng-interactive-tutorial


三、大模型应用开发框架
1. spring-ai-alibaba：https://java2ai.com/docs/1.0.0.2/spring-ai-sourcecode-explained/chapter-1-chat-first-experience/?spm=5176.29160081.0.0.2856aa5c4M1mJa#chat%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B
2. https://github.com/alibaba/spring-ai-alibaba/blob/main/README-zh.md
3. spring-ai学习资料：https://www.yuque.com/tulingzhouyu/db22bv
4. langchain4j：https://docs.langchain4j.dev/category/tutorials
5. spring-ai：https://docs.spring.io/spring-ai/reference/
6. langgrap：https://www.langchain.com/langgraph；https://langchain-ai.github.io/langgraph/?ajs_aid=07e657cd-56dc-42ea-a4c1-26d59cb5ec39  ；https://www.aidoczh.com/langgraph/
7. 

四、agent：
1. agent模式：https://www.anthropic.com/research/building-effective-agents
2. plan—and-excute：https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
3. ReAct Agent：https://zhuanlan.zhihu.com/p/1931154686532105460
4. agents：https://github.com/anthropics/claude-cookbooks/tree/main/patterns/agents
5. agent设计模式：https://github.com/Lordog/dive-into-llms/tree/main
6. 智能体的六种设计模式：https://www.bilibili.com/video/BV1Yewge7Eiy?spm_id_from=333.788.videopod.sections&vd_source=e551fb2a5e99d67f6279c11ce67d51c5
7. Agent自我优化微软开源agent-lightning：https://github.com/microsoft/agent-lightning
8. agent论文：https://lilianweng.github.io/posts/2023-06-23-agent/#agent-system-overview
9. ai-agents-for-beginners：https://github.com/microsoft/ai-agents-for-beginners
10. agent国外相关资料：https://github.com/VanGongwanxiaowan/Agent

五、rag：

1.  理论学习：https://arxiv.org/abs/2005.11401
2. langchain 官方 rag 示例：https://langchain.readthedocs.io/
3. 向量数据库 / 存储（RAG 基础设施）Milvus（向量 DB）https://milvus.io/
4. 知识图谱：https://github.com/whyhow-ai/knowledge-table，https://github.com/whyhow-ai/knowledge-graph-studio
5. 知识图谱（GraphRAG）：https://github.com/microsoft/graphrag 
6. RAG参考库：https://github.com/HKUDS/RAG-Anything
7. 开源（RAGflow）：https://github.com/infiniflow/ragflow

六、mcp：

1. mcp官网：https://modelcontextprotocol.io/docs/getting-started/intro ；
2. aws：https://github.com/awslabs/mcp/tree/main
3. BrowserMCP：https://github.com/BrowserMCP/mcp
4. microsoft：https://github.com/microsoft/mcp
5. 案例库：https://github.com/MarkTechStation/VideoCode
6. mcp中文官网：https://mcpcn.com/
7. mcp实例：https://modelcontextprotocol.io/docs/develop/build-server
8. mcp-python-sdk：https://github.com/modelcontextprotocol/python-sdk
9. mcp市场：https://mcp.so
10. mcp-for-beginners：https://github.com/microsoft/mcp-for-beginners/blob/main/translations/zh/README.md
11. 

 开发调试工具：cherry stutio

七、AI coding
1. copilot：https://qoder.com/blog  ；https://www.coze.cn/studio?utm_campaign=6353598&utm_content=home&utm_id=0&utm_medium=sem&utm_source=baidu_pz&utm_source_platform=pc&utm_term=coze_baidu_pz_pc
2. AI编码：https://docs.qoder.com/troubleshooting/mcp-common-issue
3. agent-rules：https://github.com/Lordog/dive-into-llms/tree/main
4. SDD框架（Spec-Kit ）：https://github.com/github/spec-kit/blob/main/spec-driven.md
    a. 开发实战：https://www.bilibili.com/video/BV1tNx5ziExd/?spm_id_from=333.337.search-card.all.click&vd_source=e551fb2a5e99d67f6279c11ce67d51c5
    b. tools：https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/
5. SDD（Spec-Driven Development）：
● https://zhuanlan.zhihu.com/p/1961118042852401872  https://www.bilibili.com/video/BV1aEHCz1EgZ/?spm_id_from=333.337.search-card.all.click&vd_source=e551fb2a5e99d67f6279c11ce67d51c5
●     https://github.com/github/spec-kit/blob/main/templates/commands/plan.md
● https://www.bilibili.com/video/BV1aEHCz1EgZ/?spm_id_from=333.337.search-card.all.click&vd_source=e551fb2a5e99d67f6279c11ce67d51c5


八、A2A
1. A2A：https://www.a2aprotocol.net/  ；
2. google a2a：https://github.com/a2aproject/A2A
3. 

九、微调



1. 大模型微调：https://github.com/Lordog/dive-into-llms/tree/main
2. 大模型微调（中文）：https://github.com/datawhalechina/self-llm
3. llmafactory：https://github.com/hiyouga/LLaMA-Factory
4. AI 大模型资料 （知识库+面试）：https://www.yuque.com/aaron-wecc3/dhluml?#  密码：ghkq
5. 



十、workflow
1. n8n：https://github.com/Zie619/n8n-workflows
2. 

十一、大模型综合应用
1. 智能助手开源项目（好，数据分析）：https://github.com/apconw/sanic-web?tab=readme-ov-file  
2. 大模型案例库：https://github.com/Lordog/dive-into-llms/tree/main
3. 大模型应用案例库：https://github.com/Shubhamsaboo/awesome-llm-apps?tab=readme-ov-file
4. generative-ai-for-beginners（大模型综合应用案例）：https://github.com/microsoft/generative-ai-for-beginners
5. Scrapegraph-AI：https://github.com/ScrapeGraphAI/Scrapegraph-ai/blob/main/docs/chinese.md
6. 组建AI虚拟团队的开源工具：https://github.com/crewAIInc/crewAI
7. 智能助手合集：https://github.com/Shubhamsaboo/awesome-llm-apps
8. 500-AI-Agents-Projects：https://github.com/ashishpatel26/500-AI-Agents-Projects
9. 

十二、配套工具
1. 天气app-key：https://www.weatherapi.com/docs/  ;api-key:800af0bd600a4634b31133711251111
2. 搜索引擎代理：https://serpapi.com/manage-api-key ，apikey：35d6362071abfaa8f427ddbfcce78b24fd869dbff8f9903d471464e8f806976f
3. dashscope:api-key: sk-207b9e183ee3415796cb7671db57a72b
4. Zread AI能一键将GitHub英文项目转为中文版：https://zread.ai/
5. 如何高效看英文网站：
● 工具：https://translate.google.com/translate?sl=auto&tl=zh-CN&u=目标地址
    ○ 示例：https://translate.google.com/translate?sl=auto&tl=zh-CN&u=https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
6. 需求的软件开发agent：https://github.com/FoundationAgents/MetaGPT/blob/main/docs/README_CN.md
7. google -colab（浏览器中编写和执行 Python 代码）：https://colab.research.google.com/
8. 阿里云DSW（类似：google -colab）：https://dsw-cn-hangzhou.data.aliyun.com/
9. 阿里云魔塔（免费）：https://modelscope.cn/my/mynotebook/preset
附录
大模型体系


取数架构

工程公共架构







1、不同环节可以用不同的模型，比如向量化、排序等环节各自用不同模型。
2、AI性能监控
3、agent：评估优化器模式（代码生成是用比较好，一个生成，一个评估）、路由模式、编排工作者模式、链接工作流、并行化工作模式。
4、需要明确研发过程所有角色和职责，通过编排器生成技术方案，通过评估优化器模式二次检查，在通过评估优化器模式生成代码。
5、spring-ai，langchain4j。
