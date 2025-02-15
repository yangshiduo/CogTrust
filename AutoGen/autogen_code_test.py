import asyncio
import logging
from dataclasses import dataclass
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import CodeExecutorAgent, CodingAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import MaxMessageTermination, RoundRobinGroupChat, StopMessageTermination
from autogen_ext.code_executor.docker_executor import DockerCommandLineCodeExecutor
from autogen_ext.models import OpenAIChatCompletionClient
from typing import Sequence, List
from autogen_core.base import CancellationToken
from autogen_agentchat.base import BaseChatAgent
from autogen_agentchat.messages import (
    ChatMessage,
    StopMessage,
    TextMessage,
)
from autogen_core.components.models import (
    ChatCompletionClient,
    SystemMessage,
    LLMMessage,
    AssistantMessage,
    UserMessage,
)

@dataclass
class MyEvent:
    timestamp: str
    message: str

class MyHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, MyEvent):
                print(f"Timestamp: {record.msg.timestamp}, Message: {record.msg.message}")
        except Exception:
            self.handleError(record)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key="sk-proj-GEh9CJeWGCN24NBl3wV5T3BlbkFJI7cTgcvatBlSJLwkjwyj",
    # Optional if you have an OPENAI_API_KEY env variable set.
)

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.INFO)
my_handler = MyHandler()
logger.handlers = [my_handler]

class trust_manager(BaseChatAgent):
    def __init__(
            self,
            name: str,
            model_client: ChatCompletionClient,
            *,
            description: str = "A trust manager agent.",
            system_message: str = "You are the Trust Manager. Your role is to ensure the safety and appropriateness of all communications between the user and the assistant.",
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._system_messages = [SystemMessage(content=system_message)]
        self._model_context: List[LLMMessage] = []

    async def on_messages(self, messages: Sequence[ChatMessage],
                          cancellation_token: CancellationToken) -> ChatMessage:
        # Add messages to the model context.
        for msg in messages:
            # TODO: add special handling for handoff messages
            self._model_context.append(UserMessage(content=msg.content, source=msg.source))

        # Generate an inference result based on the current model context.
        llm_messages = self._system_messages + self._model_context
        result = await self._model_client.create(llm_messages)
        #print(result)
        # Add the response to the model context.
        self._model_context.append(AssistantMessage(content=result.content, source=self.name))
        # Detect stop request.
        request_stop = "terminate" in result.content.strip().lower()
        if request_stop:
            return StopMessage(content=result.content, source=self.name)

        return TextMessage(content=result.content, source=self.name)
async def main() -> None:
    # Create an OpenAI model client.
    writing_assistant_agent = CodingAssistantAgent(
        name="writing_assistant_agent",
        system_message="""You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.""",
        model_client=model_client,
    )

    #print("------log:------", logger.handlers)
    trust_manager_agent = trust_manager(
        name="trust_manager_agent",
        model_client=model_client,
        description="A trust manager agent.",
        system_message = "You are the Trust Manager. Your role is to ensure the safety and appropriateness of all communications between the user and the assistant.",
    )

    async with DockerCommandLineCodeExecutor(work_dir="coding") as code_executor:
        code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
        group_chat = RoundRobinGroupChat([writing_assistant_agent, code_executor_agent])
        result = await group_chat.run(
            task="Develop a program to track and manage dental office cleaning and sanitation.",
            termination_condition=MaxMessageTermination(max_messages=10), #StopMessageTermination(),
        )
    #print(result)

asyncio.run(main())