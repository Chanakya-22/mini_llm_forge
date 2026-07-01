"""Chat API endpoint for ChatML-based conversational generation.

This module exposes a strict FastAPI endpoint that consumes
GenerationRequest instances, formats conversation history using the Hugging Face
chat template, runs generation with hardware-aware stopping criteria, and
returns structured GenerationResponse payloads.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from transformers import StoppingCriteria, StoppingCriteriaList

from src.app.schemas.protocol import GenerationRequest, GenerationResponse
from src.engine.generator import engine

router = APIRouter()


class TokenStopCriteria(StoppingCriteria):
    """Halts generation as soon as a configured stop token is emitted.

    This criterion inspects the latest generated token ID during decoding and
    returns ``True`` immediately when the model emits a configured stop token.
    It is designed for use with Hugging Face ``model.generate`` calls and helps
    enforce strict conversational termination semantics.
    """

    def __init__(self, stop_token_ids: Optional[List[int]] = None) -> None:
        """Initialize the criteria with explicit token IDs to stop on.

        Args:
            stop_token_ids: A list of token IDs that should terminate generation.
        """
        super().__init__()
        self.stop_token_ids = set(stop_token_ids or [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: object) -> bool:
        """Check whether generation should stop at the current token step.

        Args:
            input_ids: The full token sequence produced so far.
            scores: The logits for the current token step.

        Returns:
            ``True`` if the latest generated token is a stop token, otherwise
            ``False``.
        """
        if not self.stop_token_ids:
            return False

        last_token_id = int(input_ids[0, -1].item())
        return last_token_id in self.stop_token_ids


@router.post("/completions", response_model=GenerationResponse)
async def chat_completions(request: GenerationRequest) -> GenerationResponse:
    """Generate a conversational assistant response from ChatML messages.

    The endpoint formats the incoming chat history with the tokenizer's native
    chat template, appends the generation prompt, runs inference with strict
    sampling controls, and returns a structured telemetry payload.

    Args:
        request: A validated generation request containing ChatML messages and
            inference controls.

    Returns:
        A ``GenerationResponse`` with the generated assistant text and runtime
        telemetry.
    """
    try:
        if not engine.model or not engine.tokenizer:
            raise RuntimeError("The inference engine is not ready. Model loading failed.")

        tokenizer = engine.tokenizer
        model = engine.model

        prompt_text = tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device if hasattr(model, "device") else torch.device("cpu"))

        stop_token_ids = set()
        for stop_sequence in request.stop_sequences:
            if not stop_sequence:
                continue
            token_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
            if token_ids:
                stop_token_ids.update(token_ids)

        if getattr(tokenizer, "eos_token_id", None) is not None:
            stop_token_ids.add(int(tokenizer.eos_token_id))

        stopping_criteria = StoppingCriteriaList(
            [TokenStopCriteria(list(stop_token_ids))]
        )

        start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        prompt_length = int(input_ids.shape[1])
        generated_token_ids = output_ids[0, prompt_length:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        generated_text = generated_text.replace("<|im_end|>", "").strip()
        for stop_sequence in request.stop_sequences:
            if generated_text.endswith(stop_sequence):
                generated_text = generated_text[: -len(stop_sequence)].rstrip()
                break

        tokens_generated = int(generated_token_ids.shape[0])
        finish_reason = "stop" if tokens_generated < request.max_new_tokens else "length"

        return GenerationResponse(
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            finish_reason=finish_reason,
            execution_time_ms=round(elapsed_ms, 3),
            model_name=getattr(model, "name_or_path", getattr(model, "config", type("Config", (), {"_name_or_path": "unknown"})())._name_or_path),
        )
    except torch.OutOfMemoryError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Generation failed due to GPU memory exhaustion.",
                "error": str(exc),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Generation failed due to an internal server error.",
                "error": str(exc),
            },
        )