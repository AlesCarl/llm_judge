from dotenv import load_dotenv
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from agent.llm.model_factory import load_model
from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.summarizer import summarize_messages, needs_summarization

load_dotenv()

OVERALL_DIAGNOSIS_PROMPT = """\
    You are a network troubleshooting expert.
    Focus on (1) detecting if there is an anomaly, (2) localizing the faulty devices, and (3) identifying the root cause.
    
    Basic requirements:
    - Use the provided tools to gather necessary information.
    - Do not provide mitigation unless explicitly required.
"""

# Summarize when conversation exceeds this many characters ( 30k ≈ 7-8k tokens)
_SUMMARIZE_CHAR_THRESHOLD = 30_000    # 20k

# Keep this many recent messages intact after summarization
_KEEP_LAST_MESSAGES = 6


class DiagnosisAgent:
    """An agent that performs network diagnosis using the ReAct framework.

    Includes a pre_model_hook that summarizes the conversation history
    when it exceeds _SUMMARIZE_CHAR_THRESHOLD characters, preventing
    Ollama context-window overflows during long runs.
    """

    def __init__(self, llm_backend: str = "openai", model: str = "gpt-5-mini"):
        mcp_server_config = MCPServerConfig().load_config(if_submit=False)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools = None
        self.llm = load_model(llm_backend=llm_backend, model=model)

    async def load_tools(self):
        self.tools: list[StructuredTool] = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def _make_pre_model_hook(self):
        """Build the pre_model_hook closure capturing the LLM."""
        llm = self.llm

        def pre_model_hook(state: dict) -> dict:
            messages = state.get("messages", [])
            if needs_summarization(messages, _SUMMARIZE_CHAR_THRESHOLD):
                compressed = summarize_messages(
                    messages,
                    llm=llm,
                    keep_last=_KEEP_LAST_MESSAGES,
                    char_threshold=_SUMMARIZE_CHAR_THRESHOLD,
                )
                return {"messages": compressed}
            return {}

        return pre_model_hook

    def get_agent(self):
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=OVERALL_DIAGNOSIS_PROMPT,
            pre_model_hook=self._make_pre_model_hook(),
        )
        return agent