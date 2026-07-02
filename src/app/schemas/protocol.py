"""Enterprise-grade Pydantic schemas for a ChatML-compatible generation API.

This module defines strict request and response models for conversational AI
workloads. The schemas are designed for use with Pydantic V2 and enforce a
well-defined contract for chat history, inference controls, and generation
telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Represents a single message in a ChatML-formatted conversation.

    The schema enforces a rigid role contract and ensures message content is
    normalized, non-empty, and bounded to prevent abuse and resource-heavy
    payloads.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Role = Field(
        ...,
        description="The semantic role of the message in the ChatML conversation.",
    )
    content: str = Field(
        ...,
        description="The textual payload for the chat message.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> str:
        """Normalize and validate message content for safety and consistency.

        Args:
            value: The raw content supplied by the caller.

        Returns:
            A trimmed string that is non-empty and within the allowed length.

        Raises:
            ValueError: If the content is not a string, empty, or exceeds the maximum length.
        """
        if not isinstance(value, str):
            raise ValueError("Message content must be provided as a string.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("Message content must not be empty.")

        if len(normalized) > 100_000:
            raise ValueError(
                "Message content exceeds the maximum allowed length of 100,000 characters."
            )

        return normalized


class GenerationRequest(BaseModel):
    """Represents a strict generation request for a conversational AI gateway.

    This schema replaces free-form text completion inputs with a validated
    ChatML conversation history and a complete set of inference controls.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="An ordered list of ChatML messages representing the conversation history.",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of new tokens to generate for the assistant response.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling randomness during generation.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold for token selection.",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling limit for candidate token selection.",
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty applied to repeated tokens to discourage loops and repetition.",
    )
    do_sample: bool = Field(
        default=True,
        description="Whether to use stochastic sampling during generation.",
    )
    stop_sequences: List[str] = Field(
        default_factory=lambda: ["<|im_end|>"],
        max_length=20,
        description="A list of stop sequences that terminate generation when encountered.",
    )

    @model_validator(mode="after")
    def validate_conversation_structure(self) -> "GenerationRequest":
        """Ensure the request follows a coherent ChatML conversation pattern.

        Returns:
            The validated request instance.

        Raises:
            ValueError: If the conversation structure is not well-formed.
        """
        if not self.messages:
            raise ValueError("At least one chat message is required.")

        if self.messages[0].role == "assistant":
            raise ValueError("The first message in a request cannot be an assistant turn.")

        if self.messages[-1].role != "user":
            raise ValueError("The final message must be from the user to request generation.")

        return self

    @model_validator(mode="after")
    def validate_sampling_parameters(self) -> "GenerationRequest":
        """Ensure temperature and top_p are valid when sampling is enabled.

        Returns:
            The validated request instance.

        Raises:
            ValueError: If sampling parameters are invalid for the chosen mode.
        """
        if self.do_sample:
            if self.temperature <= 0.0:
                raise ValueError("Temperature must be strictly positive when do_sample is True.")
            if self.top_p <= 0.0:
                raise ValueError("top_p must be strictly positive when do_sample is True.")

        return self


class GenerationResponse(BaseModel):
    """Represents the structured output returned by the generation service.

    The schema captures both the generated assistant text and operational
    telemetry needed for observability, debugging, and client-side UX.
    """

    model_config = ConfigDict(extra="forbid")

    generated_text: str = Field(
        ...,
        description="The assistant response text emitted by the model.",
    )
    tokens_generated: int = Field(
        ...,
        ge=0,
        description="The number of tokens emitted during generation.",
    )
    finish_reason: Literal["stop", "length", "error"] = Field(
        ...,
        description="The reason generation completed.",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="The elapsed latency of generation measured in milliseconds.",
    )
    model_name: str = Field(
        ...,
        description="The exact base model and adapter combination used to generate the response.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional provider-specific metadata associated with the generation result.",
    )