import httpx
from httpx import Timeout
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import os
from rich import console, print
from rich.console import Console

from tools import getGutenbergBooksTool, search_gutenberg_books
import json
from pyfiglet import figlet_format
from termcolor import colored

from enum import Enum

class OpenAIResponseFinishReason(Enum):
    TOOL_CALLS = "tool_calls"
    STOP = "stop"

class Agent:
    def __init__(self, model: str = "google/gemini-3-flash-preview", system: str = "", tools: list = None) -> None:
        self.model = model
        self.system = system
        self.messages: list = []
        self.console = Console()
        self.tools = tools if tools is not None else []

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        if self.system:
            self.messages.append({"role": "system", "content": system})

        text = figlet_format("ZeroAgent", font="slant")
        print(f"[red]{text}[/red]")

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})

        final_assistant_content = self.execute()

        if final_assistant_content:
            self.messages.append({"role": "assistant", "content": final_assistant_content})

        return final_assistant_content

    def execute(self) -> str:
        # Keep looping until the model provides a final text response (not tool calls)
        while True:
            with self.console.status("Running Agent ..") as status:
                response: ChatCompletion = self.client.chat.completions.create(
                    model = self.model,
                    tools = self.tools,
                    messages = self.messages
                )

                for choice in response.choices:
                    if choice.finish_reason == OpenAIResponseFinishReason.TOOL_CALLS.value and choice.message.tool_calls:
                        for tool_call in choice.message.tool_calls:
                            self.console.log(f"⛏ Initiating tool call: `{tool_call.function.name}`")
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            if function_name in globals() and callable(globals()[function_name]):
                                function_to_call = globals()[function_name]
                                executed_output = function_to_call(**function_args)
                                self.messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(executed_output),
                                })

                    elif choice.finish_reason == OpenAIResponseFinishReason.STOP.value and choice.message.content:
                        self.console.log(f"Agent processing finished")
                        return choice.message.content

if __name__ == "__main__":
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    agent = Agent(
        model = "google/gemini-3-flash-preview",
        system="You're a helpful librarian who fetches books from the remote Gutendex library",
        tools=[getGutenbergBooksTool]
    )
    agent("Hello how are you?")
    response = agent("What are the titles of some James Joyce books?")
    print(response)