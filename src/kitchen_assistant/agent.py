"""厨房助手智能体的核心构建模块。

本模块把原先散落在 cheif_assistant.py 中的逻辑封装为可复用的函数，
供 FastAPI 接口直接调用，避免每次请求都重新初始化模型和工具。
"""

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

# 加载项目根目录下的 .env 文件
load_dotenv()

# 数据库文件路径：无论从哪个目录启动，都定位到项目内的 resources 目录
RESOURCES_DIR = Path(__file__).parent / "resources"
DB_PATH = RESOURCES_DIR / "personal_chief.db"

# 系统提示词：定义私人厨师的工作流程
SYSTEM_PROMPT = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份"当前可用食材清单"。
2.智能食谱检索：只有当用户食材较复杂、或用户明确要求更专业的菜谱时，才调用 web_search 工具联网查找。若食材简单常见（如鸡蛋、番茄、土豆等家常食材），直接用你的知识快速回答即可，不必联网搜索，以免用户等待过久。
3.多维度评估与排序：从营养价值和制作难度两个维度对候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由，帮助用户快速做出决策。

请保持回答简洁、直接、实用，不要过度冗长。
"""


@lru_cache(maxsize=1)
def get_agent():
    """构建并缓存智能体（进程内单例）。

    使用 lru_cache 缓存，保证整个服务生命周期内只初始化一次模型、
    搜索工具和记忆数据库，避免每次请求重复创建带来的性能开销。

    返回：
        CompiledStateGraph：一个可调用 invoke/stream 的 LangGraph 智能体。
    """
    # 1. 搜索工具：调用 Tavily 做菜谱联网检索
    # max_results 降到 3，减少返回体积和抓取耗时（原来 5 条会拉取大量网页正文）
    web_search = TavilySearch(
        api_key=os.getenv("Tavily_api_key"),
        max_results=3,
        topic="general",
    )

    # 2. 记忆管理：用 SQLite 持久化对话状态，实现多轮上下文记忆
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    # 3. 大语言模型：通过 OpenAI 兼容接口调用通义千问
    model = init_chat_model(
        model=os.getenv("model", "qwen3.8-flash"),
        model_provider="openai",
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url"),
    )

    # 4. 组装智能体
    agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
