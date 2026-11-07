"""kitchen_assistant 包入口。

提供 `main` 函数作为命令行入口（`kitchen-assistant` 命令），
启动 uvicorn 开发服务器。
"""

import threading
import webbrowser

import uvicorn


def _open_browser():
    """延迟打开浏览器，等待服务启动完成。"""
    webbrowser.open("http://127.0.0.1:8000")


def main():
    """启动厨房助手 Web 服务。"""
    # 延迟 1.5 秒后自动打开浏览器（等待 uvicorn 就绪）
    threading.Timer(1.5, _open_browser).start()
    uvicorn.run(
        "kitchen_assistant.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
