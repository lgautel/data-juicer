"""Stage 2: Structured reasoning VLM evaluation via Cursor Agent."""
from __future__ import annotations

import asyncio

from loguru import logger

from .frame_utils import extract_frames_base64, frames_to_sdk_images
from .parse_utils import parse_vlm_json, validate_dimensions
from .prompts import SYSTEM_PROMPT, build_user_prompt


async def evaluate_segment(
    client,
    video_path: str,
    segment: dict,
    instruction: str,
    model_id: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
) -> dict:
    """Evaluate a single segment's instruction consistency via VLM.

    Returns parsed 4-dimension analysis dict with verdict + confidence.
    """
    from cursor_sdk import (
        AgentOptions, LocalAgentOptions, UserMessage,
    )

    frames_b64 = extract_frames_base64(
        video_path, segment["start_frame"],
        segment["end_frame"], num_frames
    )
    if not frames_b64:
        return {"verdict": "inconsistent", "confidence": 0.0,
                "error": "no_frames", "segment_id": segment["segment_id"]}

    images = frames_to_sdk_images(frames_b64)
    user_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_user_prompt(instruction, len(frames_b64))}"
    )

    for attempt in range(try_num):
        try:
            async with await client.agents.create(
                AgentOptions(
                    api_key=api_key,
                    model=model_id,
                    local=LocalAgentOptions(cwd=cwd),
                )
            ) as agent:
                run = await agent.send(
                    UserMessage(text=user_text, images=images)
                )
                result = await run.wait()
                parsed = parse_vlm_json(result.result)
                parsed["segment_id"] = segment["segment_id"]
                parsed["model"] = model_id
                return parsed
        except Exception as e:
            logger.warning(
                f"Stage 2 attempt {attempt+1}/{try_num} failed: {e}"
            )
            if attempt < try_num - 1:
                await asyncio.sleep(2 ** attempt)

    return {"verdict": "inconsistent", "confidence": 0.0,
            "error": "all_retries_failed",
            "segment_id": segment["segment_id"]}


async def run_stage2(
    client,
    video_path: str,
    segments: list[dict],
    instruction: str,
    model_id: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
) -> dict:
    """Run Stage 2 on all segments and aggregate results.

    Returns {segments: [...], overall_verdict, overall_confidence, model}.
    """
    results = []
    for seg in segments:
        r = await evaluate_segment(
            client, video_path, seg, instruction,
            model_id, num_frames, api_key, cwd, try_num
        )
        results.append(r)

    # overall: inconsistent if any segment is inconsistent
    verdicts = [r.get("verdict", "inconsistent") for r in results]
    confidences = [r.get("confidence", 0.0) for r in results]

    if "inconsistent" in verdicts:
        overall = "inconsistent"
        overall_conf = min(confidences) if confidences else 0.0
    else:
        overall = "consistent"
        overall_conf = min(confidences) if confidences else 0.0

    return {
        "segments": results,
        "overall_verdict": overall,
        "overall_confidence": round(overall_conf, 4),
        "model": model_id,
    }
