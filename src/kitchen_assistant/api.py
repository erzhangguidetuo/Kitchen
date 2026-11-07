"""厨房助手的 FastAPI 接口层。

对外提供两类能力：
1. 健康检查：GET /api/health
2. 对话流式接口：POST /api/chat
   - 支持纯文字、以及文字 + 图片（图片以 base64 传入）
   - 使用 Server-Sent Events (SSE) 逐字推送 AI 回复
"""

import base64
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from .agent import get_agent

# 静态文件目录：存放前端页面
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="厨房助手 API", version="1.0.0")

# 允许跨域：方便前端页面（可能在不同端口）调用接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录，供前端页面访问
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    """聊天请求体。"""

    text: str  # 用户输入的文字
    # 图片内容，可选。base64 编码字符串，可包含 "data:image/...;base64," 前缀
    image_base64: Optional[str] = None
    # 对话线程 ID，用于多轮记忆。同一用户的连续对话应传相同值
    thread_id: str = "default"


class ChatMessage(BaseModel):
    """单条聊天消息。"""

    text: str
    image_base64: Optional[str] = None
    thread_id: str = "default"


@app.get("/api/health")
def health():
    """健康检查接口。"""
    return {"status": "ok"}


@app.get("/")
def index():
    """返回前端首页。"""
    return FileResponse(STATIC_DIR / "index.html")


def _build_human_message(text: str, image_base64: Optional[str]) -> HumanMessage:
    """把请求内容构造成 LangChain 的 HumanMessage。

    纯文字时返回普通文本消息；带图片时返回多模态消息。
    """
    if image_base64:
        # 去掉可能的 data URI 前缀，只保留纯 base64 数据
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ]
        )
    return HumanMessage(content=text)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """对话接口，以 SSE 流式返回 AI 回复。"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    agent = get_agent()
    config = {"configurable": {"thread_id": request.thread_id}}
    human_message = _build_human_message(request.text, request.image_base64)

    async def event_stream():
        """逐 token 产出 SSE 数据。"""
        try:
            # stream_mode="messages" 让智能体逐条消息流式产出
            for chunk in agent.stream(
                {"messages": [human_message]},
                config=config,
                stream_mode="messages",
            ):
                # chunk 形如 (message, metadata)
                message, _metadata = chunk
                msg_type = getattr(message, "type", None)

                # 工具调用阶段：向用户推送进度提示，避免长时间无反馈
                if msg_type == "tool":
                    # tool_call_chunks 表示 AI 正在决定调用某个工具
                    tool_name = None
                    tool_call_chunks = getattr(message, "tool_call_chunks", None)
                    if tool_call_chunks:
                        tool_name = getattr(tool_call_chunks[0], "name", None) if tool_call_chunks else None
                    if tool_name == "tavily_search":
                        yield f"data: {json.dumps({'status': '正在联网搜索菜谱…'}, ensure_ascii=False)}\n\n"
                    continue

                # 只推送 AI 产出的文本增量
                if hasattr(message, "content") and isinstance(message.content, str):
                    if message.content.strip():
                        yield f"data: {json.dumps({'text': message.content}, ensure_ascii=False)}\n\n"
            # 结束标记
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
