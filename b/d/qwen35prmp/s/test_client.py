#!/usr/bin/env python3
"""通过 OpenAI 兼容接口测试 SGLang 提供的 Qwen3.5-4B 服务.

先用 serve_qwen35_sglang.sh 启动服务, 然后运行:
    /mnt/r/VENV/dj/bin/python test_client.py
    /mnt/r/VENV/dj/bin/python test_client.py --no-thinking
    /mnt/r/VENV/dj/bin/python test_client.py --prompt "用一句话介绍你自己"

采样参数遵循 Qwen3.5-4B/README.md 的 Best Practices 建议.
"""

from __future__ import annotations

import argparse
import os

from openai import OpenAI


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B SGLang 服务测试客户端")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="服务地址(默认 http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API Key(SGLang 本地服务通常为 EMPTY)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="模型名(需与服务端 --served-model-name 一致)",
    )
    parser.add_argument(
        "--prompt",
        default='Type "I love Qwen3.5" backwards',
        help="用户输入的提示词",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="最大输出 token 数",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="关闭思考模式(Instruct 模式)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    messages = [{"role": "user", "content": args.prompt}]

    # README Best Practices: 思考/非思考模式采用不同采样参数
    if args.no_thinking:
        sampling = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5)
        extra_body = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    else:
        sampling = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
        extra_body = {"top_k": 20}

    print(f"[请求] base_url={args.base_url} model={args.model} "
          f"thinking={'off' if args.no_thinking else 'on'}")
    print(f"[提示] {args.prompt}\n")

    if args.stream:
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            stream=True,
            extra_body=extra_body,
            **sampling,
        )
        print("[回复] ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                print(content, end="", flush=True)
        print()
    else:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
            **sampling,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            print("[思考]\n" + reasoning + "\n")
        print("[回复]\n" + (msg.content or ""))
        print("\n[用量]", resp.usage)


if __name__ == "__main__":
    main()
