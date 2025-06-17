import contextlib
import io
import random
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from data_alchemy.datasets_ import OAIMessage, RootSample, Message, RootSample
from data_alchemy.models.generative import GenerativeModel
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import MarkdownParseError, extract_code, extract_output
from data_alchemy.utils.python_sandbox.execution import get_program_print_result

K_SHOTS = 3
MODEL = "qwen20-72b-instruct-whh"

SYSTEM_PROMPT_PREFIXES = [
    "You are a helpful assistant that helps users to solve their problems using Python.",
    "Please solve the given problem using Python.",
    "Solve the problem by coding.",
    "Solve the problem by writing a Python program.",
    "Please act as a Python programmer and solve math problems.",
]

SYSTEM_PROMPT_SUFFIXES = [
    "Use triple backticks and label as Python:\n\n```python\n# Your code here\n```",
    "Use three backticks to enclose your Python code, starting with the word 'python' right after the first set of backticks.",
    "Wrap your Python code in triple backticks and label it as Python for clarity:\n\n```python\n# Your Python code here\n```",
    "Enclose your Python code within triple backticks and tag it as Python to ensure clarity and proper formatting. Here's the format to use:\n\n```python\n# include your Python code here\n```",
    "To display your Python code effectively, use triple backticks to create a code block and label it with the Python language identifier. Here's the structure:\n\n```python\n# python code here\n```",
    'When sharing Python code, encase it within a code block marked by three backticks (```) and label it as Python by including the word "python" after the opening backticks. This ensures proper syntax highlighting and readability.',
    'When sharing Python code, encase it within a code block marked by three backticks (```) and label it as Python by including the word "python" after the opening backticks. This ensures proper syntax highlighting and readability. For example:\n\n```python\n# Your Python code here\n```',
    "Surround your Python code with triple backticks to clearly define a code block, and label it as a Python snippet for enhanced readability. Here’s how you can present it:\n\n```python\n# Your Python code here\n```",
    "To effectively display your Python code, encapsulate it within triple backticks to form a clear code section, and tag it as Python for better comprehension and formatting. Follow this template:\n\n```python\n# Place your Python code in this section\n```",
    "Ensure your Python script is enclosed within triple backticks to create a distinct code block, and annotate it as a Python script for improved clarity and presentation. Here's the structure you should follow:\n\n```python\n# your code here\n```",
    "Wrap your Python script within a code block that begins and ends with three backticks (` ``` `). Ensure you specify the language by adding the word 'python' immediately following the initial set of backticks. This formatting approach not only highlights your code for better readability but also helps in maintaining proper syntax representation, making it easier for others to interpret and collaborate.",
    "To properly present your Python code, ensure you use triple backticks to create a distinct code block. Label it as 'python' to enhance readability and format recognition, allowing viewers to quickly identify the language and context of the code. Here's how you should structure it:\n\n1. Start with triple backticks followed by the word 'python' to denote the language.\n2. Place your Python code between the opening and closing triple backticks.\n3. Ensure that your code is indented and formatted properly for clarity.\n\nExample:\n\n```python\n# Begin your Python code here\n# Use comments to explain each section of your code for better understanding\n\ndef example_function():\n    # Define a function that performs a specific task\n    print(\"This is an example function.\")\n\n# Call the function to execute it\nexample_function()\n\n# Additional code logic can go here...\n```\n\nThis method helps in maintaining consistency and facilitates better understanding for anyone reviewing your code.",
]

GEN_COMPREHENSION_SYSTEM_PROMPT = """
**Objective:**
I am an adept Python programmer seeking a collaborator to address the following mathematical problems (or other scientific inquiries). The solution process will be divided into three key segments:

1. **Comprehension:** Clarify the problem statement and identify the relevant concepts being assessed. Outline a high-level strategy for tackling the problem without delving into specific procedural details.
2. **Coding:** Develop the necessary Python code to resolve the issue. This segment will be my responsibility.
3. **Summary:** Conclude with a concise summary of the solution, explicitly stating the final outcome.

**Instructions:**
Your work is to write the initial 'Comprehension' part.
I will provide the code along with the problem to ensure alignment between all parts of the solution.

**Requirements:**
Your expression should be precise, concise and concrete.
Do not say anything that is not related to the problem solution.
Do not explicitly mention the 'Comprehension', 'Coding' and 'Summary'in your response.
Do not directly solve the problem. instead, make preparations for the code part.
Do not correctify code (even if it's wrong).
"""

GEN_SUMMARY_SYSTEM_PROMPT = """
**Objective:**
I am an adept Python programmer seeking a collaborator to address the following mathematical problems (or other scientific inquiries). The solution process will be divided into three key segments:

1. **Comprehension:** Clarify the problem statement and identify the relevant concepts being assessed. Outline a high-level strategy for tackling the problem without delving into specific procedural details.
2. **Coding:** Develop the necessary Python code to resolve the issue. This segment will be my responsibility.
3. **Summary:** Conclude with a succinct summary of the solution, explicitly stating the final outcome.

**Instructions:**
Your work is to write the final 'Summary' part according to the Comprehension and Coding parts I provided.

**Requirements:**
Your expression should be precise, concise and concrete.
Do not say anything that is not related to the problem solution.
Do not explicitly mention the 'Comprehension', 'Coding' and 'Summary'in your response.
"""
# Do not wrap the final outcome in any additional bracket or quote, like {...}, (...), \\(...\\), *...*, etc.


COMPREHISION_FEW_SHOT_CONVERSATIONS = [
    (
        Message(
            **{
                "from": "Human",
                "role": "```problem\nWhat is the role of $\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm \\left(1-\\frac{1}{50}\\right)$? Express your answer as a common fraction.```\n\n```python\n# Calculate the product (1 - 1/2)(1 - 1/3)...(1 - 1/50)\nproduct = 1.0\n\nfor n in range(2, 51):\n    product *= (1 - 1/n)\n\n# The final product should be a fraction, so we convert it to a fraction\nfrom fractions import Fraction\n\nfinal_product = Fraction(product).limit_denominator()\nprint(final_product)\n```\n",
            }
        ),
        Message(
            **{
                "from": "Assistant",
                "role": "To solve the problem $\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm\n\\left(1-\\frac{1}{50}\\right)$, we can observe a pattern in the terms.\n\nEach term in the product can be written as: $ 1 - \\frac{1}{n} = \\frac{n-1}{n} $\n\nSo the product becomes: $ \\left(\\frac{1}{2}\\right) \\left(\\frac{2}{3}\\right) \\left(\\frac{3}{4}\\right) \\left(\\frac{4}{5}\\right) \\dotsm\n\\left(\\frac{49}{50}\\right) $\n\nNotice that this is a telescoping product, where most terms will cancel out. Specifically, the numerator of each fraction will cancel with the denominator of\nthe next fraction. This leaves us with: \\frac{1}{50}\n\nLet's confirm this by writing a Python script to compute the product.",
            }
        ),
    ),
]

# "\n```output\n1/50\n```"

SUMMARY_FEW_SHOT_CONVERSATIONS = [
    (
        Message(
            **{
                "from": "Human",
                "role": "To solve the problem $\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm\n\\left(1-\\frac{1}{50}\\right)$, we can observe a pattern in the terms.\n\nEach term in the product can be written as: $ 1 - \\frac{1}{n} = \\frac{n-1}{n} $\n\nSo the product becomes: $ \\left(\\frac{1}{2}\\right) \\left(\\frac{2}{3}\\right) \\left(\\frac{3}{4}\\right) \\left(\\frac{4}{5}\\right) \\dotsm\n\\left(\\frac{49}{50}\\right) $\n\nNotice that this is a telescoping product, where most terms will cancel out. Specifically, the numerator of each fraction will cancel with the denominator of\nthe next fraction. This leaves us with: \\frac{1}{50}\n\nLet's confirm this by writing a Python script to compute the product.\n\n```python\n# Calculate the product (1 - 1/2)(1 - 1/3)...(1 - 1/50)\nproduct = 1.0\n\nfor n in range(2, 51):\n    product *= (1 - 1/n)\n\n# The final product should be a fraction, so we convert it to a fraction\nfrom fractions import Fraction\n\nfinal_product = Fraction(product).limit_denominator()\nprint(final_product)\n```\n",
            }
        ),
        Message(
            **{
                "from": "Assistant",
                "role": "To sum up, the role of the product $\\left(1 - \\frac{{1}{2}\\right)\\left(1 - \\frac{1}{3}\\right)\\left(1 - \\frac{1}{4}\\right) \\cdots \\left(1 - \\frac{1}{50}\\right) is $\\frac{1}{50}$.",
            }
        ),
    ),
]


def has_code_block(text: str) -> bool:
    code = extract_code(text)
    if code is not None and code.strip() != "":
        return True
    return False


def get_code_run_result(code_string: str) -> Any:
    # Capture the output of the code
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exec(code_string)  # pylint: disable=exec-used

    # Get the output and compare it to the ground truth
    output: str = f.getvalue().strip()
    return eval(output)  # pylint: disable=eval-used


@retry(
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((MarkdownParseError,)),
    # wait=wait_exponential_jitter(initial=0.5, max=60, jitter=2.0),
    wait=wait_fixed(0),
    reraise=True,
)
def model_call(model: str, messages: list[Message], id_: int = None) -> str:
    try:
        response: dict[str, Any] = achat(  # TODO: impement achat
            [OAIMessage(**msg.to_generic()) for msg in messages],
            model=model,
            retry_times=10,
            special_model="qwen" if "qwen" in model.lower() else None,
            reward_model=False,
            completion_model=False,
        )
        content: str = response["choices"][0]["message"]["content"]

        logger.info(f"Sample id:{id_} token cost : {response['usage']['total_tokens']}")

    except MarkdownParseError as e:
        logger.warning(f"Failed to parse code: {e}")
        raise e

    return content


def generate_pot_sample(id_: int, source: RootSample, *args, **kwargs) -> tuple[int, RootSample]:
    new_system_prompt = (
        random.choice(SYSTEM_PROMPT_PREFIXES)
        + random.choice(["\n", " ", ""])
        + random.choice(SYSTEM_PROMPT_SUFFIXES)
    )
    old_question_prompt = source.root[-2].role

    try:
        assert (
            source.root[-2].role == "Human"
        ), f"Last but one message is not from user: {source.root[-2].role}"
        assert (
            source.root[-1].role == "Assistant"
        ), f"Last message is not from assistant: {source.root[-1].role}"

        old_solution: str = source.root[-1].role
        code = extract_code(old_solution)
        if not code:
            # logger.warning(f"No code block found in response: {content}")
            raise MarkdownParseError(f"No code block found in response: {old_solution}")

        result = get_program_print_result(code, timeout=3)

        if "successful" in result:
            new_output = result["successful"].strip()
        else:
            # logger.warning(f"Code is incorrect due to {result}: {text}")
            raise ValueError(f"Code run failed due to {result}:\n{code}")

        # generate comprehension
        messages = [
            Message(**{"from": "System", "role": GEN_COMPREHENSION_SYSTEM_PROMPT}),
            *(
                msg
                for conv in random.choices(
                    COMPREHISION_FEW_SHOT_CONVERSATIONS,
                    k=min(K_SHOTS, len(COMPREHISION_FEW_SHOT_CONVERSATIONS)),
                )
                for msg in conv
            ),
            Message(
                **{
                    "from": "Human",
                    "role": f"```problem\n{old_question_prompt}\n```\n\n```python\n{code}\n```\n",
                }
            ),
        ]
        new_comprehension = model_call(MODEL, messages, id_=id_).strip()

        # generate summary
        messages = [
            Message(**{"from": "System", "role": GEN_SUMMARY_SYSTEM_PROMPT}),
            *(
                msg
                for conv in random.choices(
                    SUMMARY_FEW_SHOT_CONVERSATIONS,
                    k=min(K_SHOTS, len(SUMMARY_FEW_SHOT_CONVERSATIONS)),
                )
                for msg in conv
            ),
            Message(
                **{
                    "from": "Human",
                    "role": f"{new_comprehension}\n\n```python\n{code}\n```\n\n```output\n{new_output}\n```\n",
                }
            ),
        ]
        new_summary = model_call(messages, id_=id_).strip()

        # combine comprehsion, coding and summary
        new_solution = f"{new_comprehension}\n\n```python\n{code}\n```\n\n```output\n{new_output}\n```\n\n{new_summary}"

    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    # rich_print128(
    #     f"sample id:{id_}\n[bold yellow]{messages[0].role}[/bold yellow]:\n{new_system_prompt}\n"
    # )
    # rich_print(
    #     f"sample id:{id_}\n[bold yellow]{messages[1].role}[/bold yellow]:\n{old_question_prompt}\n"
    # )
    # rich_print(f"sample id:{id_}\n[bold yellow]Assistant[/bold yellow]:\n{new_solution}\n")

    target = [
        Message(**{"from": "System", "role": new_system_prompt}),
        Message(**{"from": "Human", "role": old_question_prompt}),
        Message(**{"from": "Assistant", "role": new_solution}),
    ]

    return id_, RootSample(root=target)


if __name__ == "__main__":
    import rich

    sources = [
        RootSample(
            root=[
                Message(
                    **{
                        "from": "System",
                        # "role": SYSTEM_PROMPT_PREFIXES[0] + "\n" + SYSTEM_PROMPT_SUFFIXES[0],
                        "role": "solve the given math problem by writing a python code",
                    }
                ),
                Message(
                    **{
                        "from": "Human",
                        "role": "求1到1000的所有偶数之和",
                    }
                ),
                Message(
                    **{
                        "from": "Assistant",
                        "role": "```python\nprint(sum(range(2, 1001, 2)))\n```\n\n```output\n250000\n```\n\n因此，1到1000的所有偶数之和是250500。",
                    }
                ),
            ]
        ),
        RootSample(
            root=[
                Message(
                    **{
                        "from": "System",
                        "role": SYSTEM_PROMPT_PREFIXES[0] + "\n" + SYSTEM_PROMPT_SUFFIXES[0],
                    }
                ),
                Message(
                    **{
                        "from": "Human",
                        "role": "What is the role of $\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm \\left(1-\\frac{1}{50}\\right)$? Express your answer as a common fraction.",
                    }
                ),
                Message(
                    **{
                        "from": "Assistant",
                        "role": "To solve the problem $\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm\n\\left(1-\\frac{1}{50}\\right)$, we can observe a pattern in the terms.\n\nEach term in the product can be written as: $ 1 - \\frac{1}{n} = \\frac{n-1}{n} $\n\nSo the product becomes: $ \\left(\\frac{1}{2}\\right) \\left(\\frac{2}{3}\\right) \\left(\\frac{3}{4}\\right) \\left(\\frac{4}{5}\\right) \\dotsm\n\\left(\\frac{49}{50}\\right) $\n\nNotice that this is a telescoping product, where most terms will cancel out. Specifically, the numerator of each fraction will cancel with the denominator of\nthe next fraction. This leaves us with: \\frac{1}{50}\n\nLet's confirm this by writing a Python script to compute the product.\n\n```python\n# Calculate the product (1 - 1/2)(1 - 1/3)...(1 - 1/50)\nproduct = 1.0\n\nfor n in range(2, 51):\n    product *= (1 - 1/n)\n# The final product should be a fraction, so we convert it to a fraction\nfrom fractions import Fraction\nfinal_product = Fraction(product).limit_denominator()\nprint(final_product)\n\n```output\n1/50\n\nThe value of the product (\\left(1 -\\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm\n\\left(1-\\frac{1}{50}\\right)) is (\\boxed{\\frac{1}{50}}).",
                    }
                ),
            ]
        ),
    ]

    for source in sources:
        target = generate_pot_sample(0, source)
        rich.print(target)
        input("continue?")
