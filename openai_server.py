"""OpenAI-compatible REST server for OmniSVG (8B).

Reuses inference.py's model loading / generation pipeline and exposes a
`/v1/chat/completions` endpoint shaped like the OpenAI Chat Completions API,
so existing OpenAI-SDK / vLLM-style clients can call OmniSVG without changes.
"""

import argparse
import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

import inference

MODEL_ID = "omnisvg-8b"
MODEL_SIZE = "8B"

inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[lifespan] Loading OmniSVG {MODEL_SIZE}...")
    inference.load_models(model_size=MODEL_SIZE)
    print(f"[lifespan] Ready. Devices: {inference.get_model_devices_info()}")
    yield


app = FastAPI(title="OmniSVG OpenAI-compatible API", lifespan=lifespan)


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Union[ContentPartText, ContentPartImage]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False


class ChatChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


class ModelEntry(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "omnisvg"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelEntry]


def _parse_image_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Malformed data URL")
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in image_url: {e}")
        try:
            return Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")
    raise HTTPException(
        status_code=400,
        detail="Only base64 data URLs are supported (data:image/...;base64,...)",
    )


def _extract_task(messages: List[ChatMessage]):
    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found")
    last = user_msgs[-1]
    content = last.content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty user message")
        return "text-to-svg", text

    image: Optional[Image.Image] = None
    text: Optional[str] = None
    for part in content:
        if part.type == "image_url":
            image = _parse_image_url(part.image_url.url)
        elif part.type == "text":
            text = part.text

    if image is not None:
        return "image-to-svg", image
    if text and text.strip():
        return "text-to-svg", text.strip()
    raise HTTPException(
        status_code=400,
        detail="User message must contain non-empty text or an image_url",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_size": inference.current_model_size,
        "devices": inference.get_model_devices_info(),
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelEntry(id=MODEL_ID)])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if req.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not found. Available: '{MODEL_ID}'",
        )
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported (SVG post-processing requires the full token sequence).",
        )

    n = max(1, min(req.n or 1, inference.MAX_NUM_CANDIDATES))
    task_type, content = _extract_task(req.messages)

    if task_type == "text-to-svg":
        subtype = inference.detect_text_subtype(content)
        task_key = f"text-to-svg-{subtype}"
    else:
        subtype = "image"
        task_key = "image-to-svg"

    cfg = inference.TASK_CONFIGS[task_key]
    temperature = req.temperature if req.temperature is not None else cfg["default_temperature"]
    top_p = req.top_p if req.top_p is not None else cfg["default_top_p"]
    top_k = cfg["default_top_k"]
    rep_penalty = cfg["default_repetition_penalty"]
    max_length = req.max_tokens or 1024

    async with inference_lock:
        if task_type == "image-to-svg":
            img_processed, _ = inference.preprocess_image_for_svg(
                content, replace_background=True
            )
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img_processed.save(tmp.name, format="PNG")
                    tmp_path = tmp.name
                inputs = inference.prepare_inputs("image-to-svg", tmp_path)
                candidates = await asyncio.to_thread(
                    inference.generate_candidates,
                    inputs, "image-to-svg", subtype,
                    temperature, top_p, top_k, rep_penalty,
                    max_length, n, False,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            inputs = inference.prepare_inputs("text-to-svg", content)
            candidates = await asyncio.to_thread(
                inference.generate_candidates,
                inputs, "text-to-svg", subtype,
                temperature, top_p, top_k, rep_penalty,
                max_length, n, False,
            )

    if not candidates:
        raise HTTPException(status_code=500, detail="No valid SVG candidate generated")

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(sum(max(1, len(c["svg"]) // 4) for c in candidates))
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=MODEL_ID,
        choices=[
            ChatChoice(
                index=i,
                message={"role": "assistant", "content": c["svg"]},
                finish_reason="stop",
            )
            for i, c in enumerate(candidates[:n])
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.exception_handler(HTTPException)
async def openai_error_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                "code": exc.status_code,
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description="OmniSVG OpenAI-compatible server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
