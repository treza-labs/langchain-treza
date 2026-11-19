"""All of Treza's tools with one key and one spending cap."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr

from langchain_treza.tools import TrezaBalance, TrezaGenerateImage, TrezaGenerateVideo, TrezaGetRun, TrezaRunPipeline


class TrezaToolkit(BaseToolkit):
    """Treza's tools for an agent: pictures, video clips, saved pipelines, run lookup and balance.

    Example:
        .. code-block:: python

            from langchain_treza import TrezaToolkit

            tools = TrezaToolkit(max_cost_usd=1.0).get_tools()
    """

    treza_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "TREZA_API_KEY",
            error_message="Pass api_key or set TREZA_API_KEY. Make a key at https://www.trezalabs.com/platform/api-keys",
        ),
    )
    max_cost_usd: Optional[float] = None
    """The most any one render may cost. None means no cap, so set one for an agent."""
    include_pipelines: bool = True
    """Leave out treza_run_pipeline when the agent should only generate."""

    def get_tools(self) -> list[BaseTool]:
        shared = {"api_key": self.treza_api_key, "max_cost_usd": self.max_cost_usd}
        tools: list[BaseTool] = [TrezaGenerateImage(**shared), TrezaGenerateVideo(**shared)]
        if self.include_pipelines:
            tools.append(TrezaRunPipeline(**shared))
        return [*tools, TrezaGetRun(**shared), TrezaBalance(**shared)]
