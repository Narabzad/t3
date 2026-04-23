#!/usr/bin/env python3
"""
Minimal OpenAI-compatible proxy for Google Gemini via the google-genai SDK.

Usage:
    GOOGLE_GENAI_USE_VERTEXAI=true GEMINI_API_KEY=<key> python gemini_proxy_server.py [--port 8081]

Then point lm-eval at:
    base_url=http://localhost:8081/v1/chat/completions
"""

import argparse
import asyncio
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_REST_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    model = body.get("model", "gemini-2.5-flash")
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens", 65536)
    temperature = body.get("temperature", 0.6)
    n = body.get("n", 1)
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]

    # Separate system instruction from conversation
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    generation_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if stop:
        generation_config["stopSequences"] = stop[:4]

    payload = {"contents": contents, "generationConfig": generation_config}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = GEMINI_REST_URL.format(model=model)
    params = {"key": GEMINI_API_KEY}

    async def generate_one():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, params=params, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[proxy] error: {e}")
            return ""

    texts = await asyncio.gather(*[generate_one() for _ in range(n)])

    choices = [
        {
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }
        for i, text in enumerate(texts)
    ]

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(f"Starting Gemini proxy on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
