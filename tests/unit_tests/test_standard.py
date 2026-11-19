"""LangChain's standard unit tests for tools (langchain-tests)."""

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_treza import TrezaBalance, TrezaGenerateImage, TrezaGenerateVideo, TrezaGetRun, TrezaRunPipeline


class _Standard(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "treza_live_test"}

    @property
    def init_from_env_params(self):
        return ({"TREZA_API_KEY": "treza_live_test"}, {}, {"treza_api_key": "treza_live_test"})


class TestGenerateImageStandard(_Standard):
    @property
    def tool_constructor(self):
        return TrezaGenerateImage

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"prompt": "a red kite", "aspect_ratio": "4:5"}


class TestGenerateVideoStandard(_Standard):
    @property
    def tool_constructor(self):
        return TrezaGenerateVideo

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"prompt": "a fox runs", "aspect_ratio": "9:16", "duration_sec": 4}


class TestRunPipelineStandard(_Standard):
    @property
    def tool_constructor(self):
        return TrezaRunPipeline

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"pipeline_id": "pipeline_1", "inputs": {"prompt_1": "a lighthouse"}}


class TestGetRunStandard(_Standard):
    @property
    def tool_constructor(self):
        return TrezaGetRun

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"pipeline_id": "pipeline_1", "run_id": "r1"}


class TestBalanceStandard(_Standard):
    @property
    def tool_constructor(self):
        return TrezaBalance

    @property
    def tool_invoke_params_example(self) -> dict:
        return {}
