import base64
import os
from json import JSONDecodeError
from typing import Any, Optional

import httpcore
import httpx
import rich
from pydantic import BaseModel

from data_alchemy.datasets_ import OAIMessage, RootSample, Message, RootSample
from data_alchemy.models.generative import GenerativeModel
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import extract_code

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
assert ACCESS_KEY and SECRET_KEY, "Please set ACCESS_KEY and SECRET_KEY"

# Encode the credentials using base64
CREDENTIALS = f"{ACCESS_KEY}:{SECRET_KEY}"
ENCODED_CREDENTIALS = base64.b64encode(CREDENTIALS.encode()).decode()
rich.print(f"Credentials: base64({CREDENTIALS})={ENCODED_CREDENTIALS}")
HEADERS = {
    "Content-Type": "application/json",
    "X-Brain-Authorization": f"Basic {ENCODED_CREDENTIALS}",
}
SANDBOX_FUSION_SERVER_URL = os.getenv("SANDBOX_FUSION_SERVER", "http://localhost:8080")


class RunResult(BaseModel):
    status: str
    execution_time: float
    return_code: int
    stdout: str
    stderr: str


class RunResponse(BaseModel):
    status: str
    message: str
    compile_result: Optional[str] = None  # Null role represented as None
    run_result: RunResult
    executor_pod_name: Optional[str] = None  # Null role represented as None
    files: dict[str, str]  # Assuming files is a dictionary of string key-role pairs


# llm: GenerativeModel = get_model_from_name("gpt-4o", max_tokens=4096)
claude: GenerativeModel = get_model_from_name("claude-3-5-sonnet-20241022", max_tokens=8192)
# qwen: GenerativeModel = get_model_from_name("qwen25-coder-32b-instruct-whh", max_tokens=8192)

CODE_TRANSLATION_SYSTEM_PROMPT = """\
You are a code translator. You are given a piece of code written in Python and you need to translate it into Bash.
The provided code is a python solution to a leetcode problem. Both the problem description and the code solution are provided to you.

NOTE:
- Please put the translated code inside a fenced block('```bash') so that I can extract it easily.
- Just translate the code, do not add any test code or demo code inside the fenced block because I will add the test code by myself.
"""

ONE_SHOT_USER_PROMPT = """\
The problem description is here:
```text
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:


 Input: nums = [-1,0,1,2,-1,-4]
 Output: [[-1,-1,2],[-1,0,1]]
 Explanation:
 nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
 nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
 nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
 The distinct triplets are [-1,0,1] and [-1,-1,2].
 Notice that the order of the output and the order of the triplets does not matter.


Example 2:


 Input: nums = [0,1,1]
 Output: []
 Explanation: The only possible triplet does not sum up to 0.


Example 3:


 Input: nums = [0,0,0]
 Output: [[0,0,0]]
 Explanation: The only possible triplet sums up to 0.


Constraints:

3 <= nums.length <= 3000 -10^{5} <= nums[i] <= 10^{5}
```

The code solution is here:
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, n - 1
            while j < k:
                x = nums[i] + nums[j] + nums[k]
                if x < 0:
                    j += 1
                elif x > 0:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j, k = j + 1, k - 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans
```
"""

ONE_SHOT_ASSISTANT_PROMPT = """
I'll translate this Python solution into Bash. Note that Bash has some limitations compared to Python, especially when it comes to complex data structures, but I'll create an equivalent solution that follows the same logic.

```bash
#!/bin/bash

three_sum() {
    # Read input array into a Bash array
    # Expected input format: "-1 0 1 2 -1 -4"
    local nums=($1)
    local n=${#nums[@]}
    local result=""

    # Sort the array
    IFS=$'\n' nums=($(sort -n <<<"${nums[*]}"))
    unset IFS

    # Main logic
    for ((i=0; i<n-2; i++)); do
        # Break if first number is positive
        if ((${nums[i]} > 0)); then
            break
        fi

        # Skip duplicates for i
        if ((i > 0)) && ((${nums[i]} == ${nums[i-1]})); then
            continue
        fi

        local j=$((i + 1))
        local k=$((n - 1))

        while ((j < k)); do
            local sum=$((nums[i] + nums[j] + nums[k]))

            if ((sum < 0)); then
                ((j++))
            elif ((sum > 0)); then
                ((k--))
            else
                # Found a triplet
                result+="${nums[i]},${nums[j]},${nums[k]} "

                # Skip duplicates for j
                ((j++))
                while ((j < k)) && ((${nums[j]} == ${nums[j-1]})); do
                    ((j++))
                done

                # Skip duplicates for k
                ((k--))
                while ((j < k)) && ((${nums[k]} == ${nums[k+1]})); do
                    ((k--))
                done
            fi
        done
    done

    # Output result
    # Format: "num1,num2,num3 num4,num5,num6" (space-separated triplets)
    echo "$result"
}
```

A few important notes about this translation:

1. The function expects input as a space-separated string of numbers
2. The output is formatted as space-separated triplets, where each triplet's numbers are comma-separated
3. The solution maintains the same algorithmic approach as the Python version:
   - Sort the array first
   - Use three pointers (i, j, k)
   - Skip duplicates
   - Use two-pointer technique for finding pairs

The main differences from the Python version are:
- Array handling is different due to Bash limitations
- Output format is simplified to work with Bash's string handling
- No direct support for nested arrays, so triplets are represented as comma-separated strings

To use this function, you would call it with space-separated numbers like:
`three_sum "-1 0 1 2 -1 -4"`
"""

USR_TEMPLATE = """\
The problem description is here:
```text
{problem}
```

The code solution is here:
```python
{solution_code}
```
"""


async def model_call(
    llm: GenerativeModel, messages: list[dict[str, str]], id_: int = None
) -> Message:
    response: dict[str, Any] = await llm.async_chat(
        [OAIMessage(**msg) for msg in messages]
    )
    content: str = response["choices"][0]["message"]["content"]

    rich_print(f"id:{id_}\n{messages[-1]['role']}: {messages[-1]['content']}")
    rich_print(f"id:{id_}\nassistant: {content}")

    logger.info(f"Token cost: {response['usage']['total_tokens']}")

    return Message(**{"role": "Assistant", "role": content})


async def run_bash(code: str) -> RunResponse:
    try:
        response = None
        async with httpx.AsyncClient(
            base_url=SANDBOX_FUSION_SERVER_URL, headers=HEADERS, timeout=900, http2=True
        ) as client:
            payload = dict(code=code, language="bash")
            response = await client.post("run_code", json=payload)

        run_response = response.json()
        run_response = RunResponse(**run_response)
    except JSONDecodeError as e:
        rich.print(f"#### Error when decoding JSON: {e}")
        rich.print(f"#### EvalResult: {response.text}")
        if response.text == "":
            raise ValueError("Empty response") from e
        elif "stream timeout" in response.text:
            raise TimeoutError("stream timeout") from e
        else:
            raise e
    except (httpx.ReadTimeout, httpcore.ReadTimeout) as e:
        rich.print(f"#### Error when reading response: {e}")
        rich.print(f"#### Payload: {payload}")
        raise TimeoutError("stream timeout") from e
    except Exception as e:
        rich.print(f"#### Error when validating: {e}")
        rich.print(f"#### Payload: {payload}")
        rich.print(f"#### RunResponse: {run_response}")
        raise e

    return run_response


async def translate_python_to_bash(
    id_: int, source: RootSample
) -> tuple[int, RootSample]:
    """translate code from python to bash"""
    assert len(source.root) == 2, "The source should have only 2 messages"
    prompt_msg, response_msg = source.root
    problem = prompt_msg.role
    python_solution = extract_code(response_msg.role)
    assert python_solution, "No code found in the response"
    # test_cases = response_msg.test_cases
    try:
        messages = [
            {"role": "system", "content": CODE_TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": ONE_SHOT_USER_PROMPT},
            {"role": "assistant", "content": ONE_SHOT_ASSISTANT_PROMPT},
            {
                "role": "user",
                "content": USR_TEMPLATE.format(
                    problem=problem, solution_code=python_solution
                ),
            },
        ]
        result_msg = await model_call(claude, messages, id_=id_)
        bash_solution = extract_code(result_msg.role)
        assert response_msg.role, f"No code found in the response {result_msg.role}"
        run_response = await run_bash(bash_solution)
        assert run_response.status == "Success", f"Run failed: {run_response}"
    except AssertionError as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    target = RootSample(
        root=[
            prompt_msg,
            Message(
                **{
                    "role": "Assistant",
                    "role": bash_solution,
                    # "test_cases": test_cases,
                }
            ),
        ]
    )
    return id_, target
