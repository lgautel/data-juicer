"""Stage 3: Multi-expert cross-model adjudication via parallel Cursor Agents."""
from __future__ import annotations

import asyncio

from loguru import logger

from .frame_utils import extract_frames_base64, frames_to_sdk_images
from .parse_utils import parse_vlm_json
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .vote import aggregate_votes


def needs_adjudication(
    stage2_result: dict,
    confidence_threshold: float,
    expert_models: list[str],
) -> bool:
    """Determine if Stage 3 multi-expert adjudication is needed."""
    if not expert_models:
        return False

    verdict = stage2_result.get("overall_verdict", "inconsistent")
    confidence = stage2_result.get("overall_confidence", 0.0)

    if verdict == "consistent" and confidence >= confidence_threshold:
        return False

    return True


async def _ask_expert(
    client,
    model_id: str,
    prompt: str,
    images,
    api_key: str,
    cwd: str,
    try_num: int,
) -> dict | None:
    """Query a single expert VLM. Returns parsed result or None on failure."""
    from cursor_sdk import (
        AgentOptions, LocalAgentOptions, UserMessage,
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
                    UserMessage(text=prompt, images=images)
                )
                result = await run.wait()
                parsed = parse_vlm_json(result.result)
                parsed["model"] = model_id
                return parsed
        except Exception as e:
            logger.warning(
                f"Expert {model_id} attempt {attempt+1}/{try_num}: {e}"
            )
            if attempt < try_num - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"Expert {model_id} failed after {try_num} attempts")
    return None


async def run_stage3(
    client,
    video_path: str,
    stage2_result: dict,
    instruction: str,
    expert_models: list[str],
    voting_strategy: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
) -> dict:
    """Run Stage 3 multi-expert adjudication.

    Parallel-queries each expert VLM and aggregates votes.
    """
    # Find the worst segment from Stage 2 for re-evaluation
    segments = stage2_result.get("segments", [])
    worst = min(segments,
                key=lambda s: s.get("confidence", 0.0),
                default=None)
    if worst is None:
        return {"final_verdict": "inconsistent", "final_score": 0.0,
                "adjudication_needed": True, "error": "no_segments"}

    frames_b64 = extract_frames_base64(
        video_path, worst.get("start_frame", 0),
        worst.get("end_frame", 0), num_frames
    )
    images = frames_to_sdk_images(frames_b64)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_user_prompt(instruction, len(frames_b64))}"
    )

    # Parallel evaluation by all experts
    votes = await asyncio.gather(
        *[_ask_expert(client, m, prompt, images,
                      api_key, cwd, try_num)
          for m in expert_models]
    )
    valid_votes = [v for v in votes if v is not None]

    if len(valid_votes) < 2:
        logger.warning(
            f"Only {len(valid_votes)} experts responded "
            f"(need >= 2). Marking as ambiguous."
        )
        return {
            "final_verdict": "ambiguous",
            "final_score": 0.0,
            "adjudication_needed": True,
            "expert_votes": valid_votes,
            "num_experts_responded": len(valid_votes),
        }

    result = aggregate_votes(valid_votes, voting_strategy)
    result["adjudication_needed"] = True
    return result
