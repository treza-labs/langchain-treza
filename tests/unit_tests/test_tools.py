"""The tools against a scripted Treza: quote, cap, start once, wait, report."""

import json

import pytest
from langchain_core.tools import ToolException

from langchain_treza import TrezaBalance, TrezaGenerateImage, TrezaGenerateVideo, TrezaGetRun, TrezaRunPipeline, TrezaToolkit
from langchain_treza import client as c

from .fakes import FakeTreza

QUOTE = {"pipelineId": "pipeline_1", "quoteId": "q1", "estimatedChargeUsd": 0.28, "sufficient": True, "model": {"id": "google/veo-3.1-lite"}}
DONE = {"status": "success", "outputs": {"video": "https://media.trezalabs.com/v.mp4"}, "durationMs": 56000}


def build(tool, fake, **settings):
    return tool(api_key="treza_live_test", client=c.TrezaClient("treza_live_test", session=fake), **settings)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda s: None)


def test_video_is_quoted_started_once_and_reported():
    fake = FakeTreza(make_video=[(200, QUOTE)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, DONE)])
    out = json.loads(build(TrezaGenerateVideo, fake).invoke({"prompt": "a fox runs", "aspect_ratio": "9:16", "duration_sec": 4}))
    assert out["url"] == "https://media.trezalabs.com/v.mp4"
    assert (out["status"], out["finished"], out["price_usd"], out["model"]) == ("success", True, 0.28, "google/veo-3.1-lite")
    assert (out["run_id"], out["pipeline_id"]) == ("r1", "pipeline_1")
    # No cap set, so none is sent.
    assert fake.asked("make_video") == [{"prompt": "a fox runs", "aspectRatio": "9:16", "durationSec": 4.0}]
    assert fake.asked("run_pipeline") == [{"pipelineId": "pipeline_1", "quoteId": "q1"}]
    assert fake.calls[0][2]["User-Agent"].startswith("langchain-treza/")


def test_over_the_cap_goes_back_to_the_model_and_starts_nothing():
    fake = FakeTreza(make_video=[(200, QUOTE)])
    said = build(TrezaGenerateVideo, fake, max_cost_usd=0.10).invoke({"prompt": "a fox runs"})
    assert "over the Max cost of $0.10" in said and "Nothing was spent" in said
    assert fake.asked("run_pipeline") == []
    assert fake.asked("make_video")[0]["maxPriceUsd"] == 0.10


def test_a_refusal_raises_when_error_handling_is_off():
    fake = FakeTreza(make_video=[(200, QUOTE)])
    with pytest.raises(ToolException):
        build(TrezaGenerateVideo, fake, max_cost_usd=0.10, handle_tool_error=False).invoke({"prompt": "a fox runs"})


def test_the_model_cannot_set_the_cap_or_the_key():
    for tool in (TrezaGenerateImage, TrezaGenerateVideo, TrezaRunPipeline):
        fields = set(tool(api_key="k").args)
        assert not fields & {"max_cost_usd", "api_key", "treza_api_key", "wait_seconds"}, fields


def test_image_passes_references_and_model():
    quote = {**QUOTE, "estimatedChargeUsd": 0, "model": "google/gemini-2.5-flash-image"}
    done = {"status": "success", "outputs": {"image": "https://media.trezalabs.com/i.png"}}
    fake = FakeTreza(make_image=[(200, quote)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, done)])
    out = json.loads(
        build(TrezaGenerateImage, fake).invoke(
            {"prompt": "a red kite", "aspect_ratio": "4:5", "reference_image_urls": ["https://media.trezalabs.com/ref.png"]}
        )
    )
    assert out["url"].endswith("i.png") and out["price_usd"] == 0 and out["model"] == "google/gemini-2.5-flash-image"
    assert fake.asked("make_image") == [
        {"prompt": "a red kite", "aspectRatio": "4:5", "referenceImageUrls": ["https://media.trezalabs.com/ref.png"]}
    ]


def test_a_slow_render_comes_back_running_and_get_run_finishes_it():
    fake = FakeTreza(make_video=[(200, QUOTE)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, {"status": "running"}), (200, DONE)])
    out = json.loads(build(TrezaGenerateVideo, fake, wait_seconds=0).invoke({"prompt": "a fox runs"}))
    assert out["status"] == "running" and out["finished"] is False and "url" not in out
    later = json.loads(build(TrezaGetRun, fake).invoke({"pipeline_id": out["pipeline_id"], "run_id": out["run_id"]}))
    assert later["url"] == "https://media.trezalabs.com/v.mp4"


def test_pipeline_passes_inputs():
    quote = {"pipelineId": "pipeline_9", "quoteId": "q9", "estimatedChargeUsd": 1.5, "sufficient": True}
    done = {"status": "success", "outputs": {"final": "https://media.trezalabs.com/f.mp4", "note": "ok"}}
    fake = FakeTreza(estimate_run_cost=[(200, quote)], run_pipeline=[(200, {"runId": "r9"})], get_run=[(200, done)])
    out = json.loads(build(TrezaRunPipeline, fake, max_cost_usd=2).invoke({"pipeline_id": " pipeline_9 ", "inputs": {"prompt_1": "a lighthouse"}}))
    assert out["url"] == "https://media.trezalabs.com/f.mp4" and out["outputs"] == done["outputs"] and out["price_usd"] == 1.5
    assert fake.asked("run_pipeline") == [{"pipelineId": "pipeline_9", "inputs": {"prompt_1": "a lighthouse"}, "quoteId": "q9"}]


def test_a_broken_render_says_why():
    fake = FakeTreza(make_video=[(200, QUOTE)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, {"status": "error", "statusNote": "The model refused the prompt."})])
    assert "The model refused the prompt." in build(TrezaGenerateVideo, fake).invoke({"prompt": "x"})


def test_balance():
    fake = FakeTreza(get_credit_balance=[(200, {"balanceUsd": 20.76})])
    out = json.loads(build(TrezaBalance, fake).invoke({}))
    assert out["balance_usd"] == 20.76 and out["top_up_url"].startswith("https://www.trezalabs.com/")


def test_a_bad_key_says_what_to_do():
    fake = FakeTreza(get_credit_balance=[(401, {"error": "Invalid API key."})])
    assert "Check your Treza API key" in build(TrezaBalance, fake).invoke({})


def test_key_comes_from_the_environment_and_is_not_printed(monkeypatch):
    monkeypatch.setenv("TREZA_API_KEY", "treza_live_env")
    tool = TrezaGenerateImage()
    assert tool.treza_api_key.get_secret_value() == "treza_live_env"
    assert "treza_live_env" not in repr(tool)
    monkeypatch.delenv("TREZA_API_KEY")
    with pytest.raises(ValueError, match="TREZA_API_KEY"):
        TrezaGenerateImage()


def test_toolkit_shares_the_key_and_the_cap():
    tools = TrezaToolkit(api_key="treza_live_test", max_cost_usd=1.0).get_tools()
    assert [t.name for t in tools] == ["treza_generate_image", "treza_generate_video", "treza_run_pipeline", "treza_get_run", "treza_balance"]
    assert all(t.max_cost_usd == 1.0 and t.treza_api_key.get_secret_value() == "treza_live_test" for t in tools)
    names = [t.name for t in TrezaToolkit(api_key="k", include_pipelines=False).get_tools()]
    assert "treza_run_pipeline" not in names
