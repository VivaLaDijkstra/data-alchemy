from typing import Literal, Optional

import rich
from rich.console import Console
from rich.markdown import Markdown

from data_alchemy.datasets_.openai_ import OAIMessage, OAISample
from data_alchemy.models.generative import GenerativeModel
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.markdown import render_think_prompt

console = Console()


class Chatter:
    """Chat with LLMs"""

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str],
        save_path: Optional[Path],
        # hyper params
        frequency_penalty: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
        stop: Optional[str | list[str]],
        stream: Optional[Literal[False]],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> None:
        self.system_prompt = system_prompt
        self.save_path = save_path

        chat_params = dict(
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )
        self.llm: GenerativeModel = get_model_from_name(model, **chat_params)
        self.end_input_flag: str = "\x1b[15~"

    def get_multiline_input(self, prompt: str = "") -> str:
        lines = []
        console.print(prompt, end="")
        while True:
            line = console.input()
            lines.append(line)
            if lines[-1].endswith(self.end_input_flag):
                lines[-1] = lines[-1].replace(self.end_input_flag, "")
                break
        return "\n".join(lines)

    def run(self) -> None:
        try:
            if not self.system_prompt:
                system_prompt = self.get_multiline_input(
                    "[bold green]system prompt: [/bold green] (press Enter to omit)"
                )
            else:
                system_prompt = self.system_prompt
        except KeyboardInterrupt:
            console.print("Exiting...")
            return

        history: list[dict[str, str]] = [OAIMessage(role="system", content=system_prompt)]
        while True:
            try:
                user_prompt = self.get_multiline_input("[bold cyan]user:[/bold cyan]\n")
                # console.print("[bold orange]User[/bold orange]", Markdown(user_prompt))  # TODO: need flush last input
                history.append(OAIMessage(role="user", content=user_prompt))

                response = self.llm.chat(messages=history)
                assistant_prompt: str = response["choices"][0]["message"]["content"]
                console.print(
                    "[bold yellow]assistant:[/bold yellow]",
                    Markdown(render_think_prompt(assistant_prompt)),
                )
                history.append(OAIMessage(role="assistant", content=assistant_prompt))
            except KeyboardInterrupt:
                console.print("Exiting...")
                break
            except Exception as e:
                rich.print(f"[red]Error: {e}[/red]")
                raise e
            finally:
                if self.save_path:
                    dump_samples([OAISample(history)], self.save_path)
