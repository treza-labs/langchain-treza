"""Treza tools for LangChain agents and graphs.

Each render tool asks Treza for a price, stops when the price is over
``max_cost_usd``, starts the render once and waits for the finished file.
``max_cost_usd``, the API key and the wait are set when the tool is built, not
by the model, so an agent cannot raise a spending cap you set.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_treza.client import (
    TOP_UP_URL,
    Data,
    TrezaClient,
    TrezaError,
    compact,
    finished_or_raise,
    max_cost,
    price_limit,
    run_item,
    to_number,
)

ImageShape = Literal["1:1", "16:9", "9:16", "4:5", "3:4", "4:3", "2:3", "3:2"]
VideoShape = Literal["16:9", "9:16", "1:1"]


class _TrezaTool(BaseTool):
    """What every Treza tool shares: the key, the cap, the wait and how a render is reported."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    treza_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "TREZA_API_KEY",
            error_message="Pass api_key or set TREZA_API_KEY. Make a key at https://www.trezalabs.com/platform/api-keys",
        ),
    )
    """A Treza API key with the pipelines read, write and run permissions."""

    max_cost_usd: Optional[float] = None
    """The most one render may cost. Treza prices it first and nothing is spent when it is higher. None means no cap."""

    wait_seconds: float = 300
    """How long to wait for a render. One still going comes back with status "running" and its IDs."""

    handle_tool_error: bool = True
    """A refusal (over the cap, low balance, a bad key) goes back to the model as text instead of raising."""

    client: Optional[Any] = Field(default=None, exclude=True)
    """A ready TrezaClient, for tests."""

    def _treza(self) -> TrezaClient:
        return self.client or TrezaClient(self.treza_api_key.get_secret_value())

    def _render(self, started: Data, output_key: Optional[str], extra: Optional[Data] = None) -> str:
        treza = self._treza()
        run = treza.wait_for_run(started["pipelineId"], started["runId"], self.wait_seconds)
        run = {"runId": started["runId"], "pipelineId": started["pipelineId"], **run}
        finished_or_raise(run)
        quoted = started.get("quoted") or {}
        model = quoted.get("model")
        item = run_item(
            run,
            output_key,
            {"price_usd": started.get("priceUsd"), "model": model.get("id") if isinstance(model, dict) else model, **(extra or {})},
        )
        return json.dumps(item)

    def _guard(self, fn: Any) -> str:
        try:
            return fn()
        except TrezaError as err:
            raise ToolException(str(err)) from err


class GenerateImageInput(BaseModel):
    prompt: str = Field(description="What the picture shows.")
    aspect_ratio: ImageShape = Field(default="1:1", description='The picture\'s shape, for example "16:9" wide or "9:16" tall.')
    reference_image_urls: Optional[list[str]] = Field(
        default=None, description="Links to pictures already in the Treza library to restyle or keep a character from."
    )
    model: Optional[str] = Field(default=None, description="A Treza model ID. Leave out for Treza's default.")


class TrezaGenerateImage(_TrezaTool):
    """Make a picture from a description with Treza."""

    name: str = "treza_generate_image"
    description: str = (
        "Make a picture from a text description with Treza. Returns JSON with the finished picture's url, "
        "the price in US dollars and the model. Uses the account's prepaid Treza credits."
    )
    args_schema: type[BaseModel] = GenerateImageInput

    def _run(self, prompt: str, aspect_ratio: str = "1:1", reference_image_urls: Optional[list[str]] = None, model: Optional[str] = None, **_: Any) -> str:
        def go() -> str:
            treza = self._treza()
            cap = max_cost(self.max_cost_usd)
            args = compact(
                {"prompt": prompt, "aspectRatio": aspect_ratio, "model": model, "referenceImageUrls": reference_image_urls, "maxPriceUsd": price_limit(cap)}
            )
            return self._render(treza.start_quoted(cap, lambda: treza.request("make_image", args)), "image")

        return self._guard(go)


class GenerateVideoInput(BaseModel):
    prompt: str = Field(description="What happens in the clip.")
    aspect_ratio: VideoShape = Field(default="16:9", description='"16:9" wide, "9:16" vertical (Shorts, Reels, TikTok) or "1:1" square.')
    duration_sec: Optional[float] = Field(default=None, ge=1, le=60, description="Clip length in seconds, usually 4 to 10. Defaults to 4.")
    with_audio: Optional[bool] = Field(default=None, description="Generate sound with the video where the model can. Defaults to true.")
    first_frame_url: Optional[str] = Field(default=None, description="Link to a picture in the Treza library that the clip opens on.")
    last_frame_url: Optional[str] = Field(default=None, description="Link to a picture in the Treza library that the clip ends on.")
    model: Optional[str] = Field(default=None, description="A Treza model ID. Leave out for Treza's default.")


class TrezaGenerateVideo(_TrezaTool):
    """Make a video clip from a description with Treza."""

    name: str = "treza_generate_video"
    description: str = (
        "Make a short video clip from a text description with Treza. Takes one to five minutes. Returns JSON with "
        "the finished clip's url, the price in US dollars and the model, or status \"running\" with a run_id and "
        "pipeline_id to pass to treza_get_run when it is not done in time. Uses the account's prepaid Treza credits."
    )
    args_schema: type[BaseModel] = GenerateVideoInput

    def _run(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        duration_sec: Optional[float] = None,
        with_audio: Optional[bool] = None,
        first_frame_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        model: Optional[str] = None,
        **_: Any,
    ) -> str:
        def go() -> str:
            treza = self._treza()
            cap = max_cost(self.max_cost_usd)
            args = compact(
                {
                    "prompt": prompt,
                    "aspectRatio": aspect_ratio,
                    "durationSec": to_number(duration_sec),
                    "withAudio": with_audio,
                    "firstFrameUrl": first_frame_url,
                    "lastFrameUrl": last_frame_url,
                    "model": model,
                    "maxPriceUsd": price_limit(cap),
                }
            )
            return self._render(treza.start_quoted(cap, lambda: treza.request("make_video", args)), "video")

        return self._guard(go)


class RunPipelineInput(BaseModel):
    pipeline_id: str = Field(description="The pipeline's ID. It is in the pipeline's address in Treza and starts with pipeline_.")
    inputs: Optional[dict[str, Any]] = Field(
        default=None, description='One entry per input step of the pipeline, for example {"prompt_1": "a lighthouse at dawn"}. Leave out to run it as saved.'
    )


class TrezaRunPipeline(_TrezaTool):
    """Run a pipeline saved in the Treza account."""

    name: str = "treza_run_pipeline"
    description: str = (
        "Run one of the account's saved Treza pipelines, with new inputs if given. Returns JSON with the url of "
        "what it made, all its outputs and the price, or status \"running\" with IDs for treza_get_run. "
        "Uses the account's prepaid Treza credits."
    )
    args_schema: type[BaseModel] = RunPipelineInput
    wait_seconds: float = 600

    def _run(self, pipeline_id: str, inputs: Optional[dict[str, Any]] = None, **_: Any) -> str:
        def go() -> str:
            treza = self._treza()
            cap = max_cost(self.max_cost_usd)
            pid = pipeline_id.strip()
            if not pid:
                raise TrezaError("Give the pipeline_id. It is in the pipeline's address in Treza and starts with pipeline_.")
            return self._render(treza.start_quoted(cap, lambda: treza.request("estimate_run_cost", {"pipelineId": pid}), inputs or None), None)

        return self._guard(go)


class GetRunInput(BaseModel):
    pipeline_id: str = Field(description="The pipeline_id a Treza tool returned.")
    run_id: str = Field(description="The run_id a Treza tool returned.")


class TrezaGetRun(_TrezaTool):
    """Look up a render that was still going when its tool stopped waiting."""

    name: str = "treza_get_run"
    description: str = (
        "Check a Treza render by its pipeline_id and run_id. Returns JSON with its status and, once finished, "
        "the url of the file. Free."
    )
    args_schema: type[BaseModel] = GetRunInput

    def _run(self, pipeline_id: str, run_id: str, **_: Any) -> str:
        def go() -> str:
            run = self._treza().call("get_run", {"pipelineId": pipeline_id, "runId": run_id})
            run = {"runId": run_id, "pipelineId": pipeline_id, **run}
            finished_or_raise(run)
            return json.dumps(run_item(run))

        return self._guard(go)


class BalanceInput(BaseModel):
    pass


class TrezaBalance(_TrezaTool):
    """The credits left on the Treza account."""

    name: str = "treza_balance"
    description: str = "Get the prepaid credit balance of the Treza account, in US dollars. Free."
    args_schema: type[BaseModel] = BalanceInput

    def _run(self, **_: Any) -> str:
        def go() -> str:
            balance = self._treza().call("get_credit_balance")
            return json.dumps(
                compact(
                    {
                        "balance_usd": to_number(balance.get("balanceUsd")),
                        "approx_videos_remaining": balance.get("approxVideosRemaining"),
                        "top_up_url": balance.get("topUpUrl") or TOP_UP_URL,
                    }
                )
            )

        return self._guard(go)
