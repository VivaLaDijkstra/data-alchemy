import random
import re
from typing import Any

from fuzzywuzzy import fuzz

from data_alchemy.datasets_ import RootSample, Message, Sample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger

MODEL = "qwen20-72b-instruct-whh"
SIMILARITY_THRESHOLD = 50
random.seed(42)

TASK_DEFINITIONS = {
    "Generation": "Generate code from scratch based on a natural language description of an algorithm problem or requirement and some constraints or conditions.",
    "Completion": "Complete existing code given the context at the Project/File/Class/Function level.",
    "Explanation": "Provide a targeted natural language explanation for a given piece of code.",
    "QA": "Provide corresponding answers based on the given code and questions.",
    "Translation": "Generate target code that semantically and interface-wise corresponds to the source code in the given programming languages.",
    "Debug": "Identify bugs in the given code and provide the corrected version of the code.",
    "Optimization": "Provide code with the same functionality but improved time and space complexity.",
    "Style_Polish": "Polish the code format, comments and naming to make it more human-readable.",
    "Unit_Test_Generation": "Provide unit test code for the code being tested (technically a subclass of Generation, but listed separately due to mature evaluation tools for unit tests and integration with automation for reward-based learning).",
    "Other": "Classes that do not fall into the above categories or you are unsure about.",
}

NEW_LINE = "\n"

SYSTEM_PROMPT = f"""# Requirement:
Please act as a coding task classifier for coding samples. You will receive a piece of text with code snippets, and your task is to **classify the task type** the text is related to.

# Task Type Definitions:
Here are the task definitions, formatted as `- **<task type>**: <definition>`.
{NEW_LINE.join(f'- **{task}**: {definition}' for task, definition in TASK_DEFINITIONS.items())}

# Notes:
1. Before giving final answers, please provide a brief reason for your classification.
2. Provide your answer strictly in the format:
   `'reason: <reason>\ntask type: <task type>'`
3. Ensure that `<task type>` must match one of the listed option of the above definitions exactly.
4. Predict only one coding task at a time. If multiple tasks are possible, predict the one that is most likely.
5. If the code is not in any of the mentioned tasks, predict 'Other'.
5. Do not output any other information or comments in the <task type> part.
"""

FEW_SHOT_USER_PROMPT0 = """Human:
please help me write a quick sort function in python
Assistant:
Sure! Here's a simple implementation of the Quick Sort algorithm in Python:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # Choose the middle element as the pivot
    left = [x for x in arr if x < pivot]  # Elements smaller than pivot
    middle = [x for x in arr if x == pivot]  # Elements equal to pivot
    right = [x for x in arr if x > pivot]  # Elements greater than pivot
    return quick_sort(left) + middle + quick_sort(right)

# Example usage:
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print(sorted_arr)
```

For the input `[3, 6, 8, 10, 1, 2, 1]`, it will output the sorted list: `[1, 1, 2, 3, 6, 8, 10]`.

Let me know if you'd like further explanation or adjustments!
"""

FEW_SHOT_ASSISTANT_PROMPT0 = """reason: 'Human' role required the 'Assistant' role to write a Quick Sort function from scratch, which fullfills the definition of the Generation task
task type: Generation"""

FEW_SHOT_USER_PROMPT1 = """Human:
please help me rewrite this code and make it more human-readable
```python
def qS(o):
 if len(o)>1:
  p=o[len(o)//2]
  L=[i for i in o if i<p]
  E=[i for i in o if i==p]
  G=[i for i in o if i>p]
  return qS(L)+E+qS(G)
 else:
  return o

def qsort(l):return qS(l)

Assistant:
Here's a more readable version of the quicksort implementation with proper coding style:

```python
def quick_sort(array):
    \"""
    Sorts an array using the quicksort algorithm.

    Args:
        array: List of comparable elements to be sorted

    Returns:
        A new sorted list containing the same elements
    \"""
    if len(array) <= 1:
        return array

    # Choose the middle element as pivot
    pivot = array[len(array) // 2]

    # Partition the array into three parts
    less_than_pivot = [
        element for element in array
        if element < pivot
    ]
    equal_to_pivot = [
        element for element in array
        if element == pivot
    ]
    greater_than_pivot = [
        element for element in array
        if element > pivot
    ]

    # Recursively sort the partitions and combine them
    return (quick_sort(less_than_pivot) +
            equal_to_pivot +
            quick_sort(greater_than_pivot))


# Example usage
if __name__ == "__main__":
    test_array = [3, 6, 8, 10, 1, 2, 1]
    sorted_array = quick_sort(test_array)
    print(f"Original array: {test_array}")
    print(f"Sorted array: {sorted_array}")
```

Key improvements made:

1. **Meaningful Names**:
   - Used descriptive function name `quick_sort` instead of `qS`
   - Used meaningful variable names like `array`, `pivot`, `less_than_pivot`, etc.

2. **Proper Documentation**:
   - Added docstring explaining the function's purpose and parameters
   - Added comments explaining key steps in the algorithm

3. **Consistent Formatting**:
   - Used proper indentation (4 spaces, as per PEP 8)
   - Added appropriate spacing around operators
   - Split long lines for better readability

4. **Logical Structure**:
   - Clearly separated the partitioning step from the recursive calls
   - Made the code flow more obvious with proper spacing and organization

5. **Example Usage**:
   - Added a demonstration section protected by `if __name__ == "__main__"`
   - Included clear output formatting

6. **Removed Unnecessary Wrapper**:
   - Eliminated the redundant wrapper function

This version maintains the exact same functionality but is much easier to read, understand, and maintain.
"""

FEW_SHOT_ASSISTANT_PROMPT1 = """reason: The task involves rewriting the given code to enhance its readability, focusing on style improvements such as naming, formatting, and documentation.
task type: Style_Polish"""

FEW_SHOT_USER_PROMPT2 = """Human:
check out this quick sort implementation
```python
def quick_sort(array):
    \"""
    Sorts an array using the quicksort algorithm.

    Args:
        array: List of comparable elements to be sorted

    Returns:
        A new sorted list containing the same elements
    \"""
    if len(array) <= 1:
        return array

    # Choose the middle element as pivot
    pivot = array[0]

    # Partition the array into three parts
    less_than_pivot = [
        element for element in array
        if element < pivot
    ]
    equal_to_pivot = [
        element for element in array
        if element == pivot
    ]
    greater_than_pivot = [
        element for element in array
        if element > pivot
    ]

    # Recursively sort the partitions and combine them
    return (quick_sort(less_than_pivot) +
            equal_to_pivot +
            quick_sort(greater_than_pivot))

```
my question is in which case would this implementation code has the worst time complexity?

Assistant:
This implementation of the quick sort algorithm has the **worst-case time complexity of \\( O(n^2) \\)**. The worst-case scenario occurs when the pivot choice results in highly unbalanced partitions.

### Analysis of the Worst Case
The pivot in this implementation is chosen as the first element of the array:

```python
pivot = array[0]
```

However, worst-case performance is not determined solely by the pivot's position but rather by how the elements are distributed relative to the pivot. The worst case arises when:
1. **All elements (except the pivot) fall into one partition.** This can happen if the array is:
   - Already sorted in ascending or descending order, and the pivot fails to split the array evenly.
   - Contains all identical elements (though this is a degenerate case where all partitions end up empty except for `equal_to_pivot`).

In such cases, the recursion depth will reach \\( O(n) \\), and each level of recursion will require \\( O(n) \\) work to partition the array, leading to \\( O(n^2) \\) overall complexity.

### Example of Worst-Case Input
1. For a sorted array in ascending order:
   ```python
   array = [1, 2, 3, 4, 5]
   ```
   At each step, the middle pivot will divide the array such that one of the partitions is empty, leading to maximum recursion depth.

2. For a sorted array in descending order:
   ```python
   array = [5, 4, 3, 2, 1]
   ```

3. For an array with all identical elements:
   ```python
   array = [5, 5, 5, 5, 5]
   ```

In these cases, the partitions will not significantly reduce the size of the problem, leading to inefficiencies.

### Key Takeaway
To mitigate the worst-case time complexity, consider using a pivot-selection strategy that better balances partitions, such as:
- Randomly choosing the pivot (Randomized Quicksort).
- Using the "median-of-three" rule to choose a pivot.
"""

FEW_SHOT_ASSISTANT_PROMPT2 = """reason: The task is about providing an answer to a question based on the provided code snippet.
task type: QA"""


# @async_model_cache(cache_dir=Path("~/.cache/data-alchemy").expanduser())
async def model_call(history: list[Message], model: str) -> Message:
    plain_text = "\n".join(f"{msg.role}:\n{msg.content}\n" for msg in history)

    few_shots = [
        (
            {"role": "user", "content": FEW_SHOT_USER_PROMPT0},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT_PROMPT0},
        ),
        (
            {"role": "user", "content": FEW_SHOT_USER_PROMPT1},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT_PROMPT1},
        ),
        (
            {"role": "user", "content": FEW_SHOT_USER_PROMPT2},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT_PROMPT2},
        ),
    ]
    random.shuffle(few_shots)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *sum(few_shots, ()),  # Unpack and flatten the few_shots list into the messages list
        {"role": "user", "content": plain_text},
    ]

    response = await achat(
        messages,
        model=model,
        retry_times=10,
        special_model="qwen" if "qwen" in model.lower() else None,
        reward_model=False,
        completion_model=False,
    )

    # logger.info(f"Token cost: {response['usage']['total_tokens']}")

    return response


def parse_result(response: dict[str, Any]) -> str:
    text = response["choices"][0]["text"]
    rich_print("[yellow bold]response:[/yellow bold]", text)

    res = re.match(r"(.*)\ntask type: (.*)", text, re.DOTALL)
    assert res, f"Regex Parsing Error, got '{text}'"

    _, task = res.groups()
    task = task.strip().lower()
    assert len(task) < 42, f"Output too long: {task}"

    tasks = [task.lower() for task in TASK_DEFINITIONS]
    similarities = [fuzz.ratio(task, tsk.lower()) for tsk in tasks]
    max_similarity = max(similarities)
    idx = similarities.index(max_similarity)
    result = tasks[idx]

    assert (
        max_similarity > SIMILARITY_THRESHOLD
    ), f"Similarity too low: sim({task}, {result})={max_similarity}"

    if result in task:
        return result
    return "other"


async def classify_coding_task(id_: int, source: RootSample) -> tuple[int, RootSample]:
    messages = source.root

    try:
        response = await model_call(history=messages, model=MODEL)
        result = parse_result(response)
    except AssertionError as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    messages[0].tag = {"coding_task": result}
    target = Sample(root=messages)
    if hasattr(target.root[0], "tag"):
        target.root[0].tag.update({"coding_task": result})
    else:
        target.root[0].tag = {"coding_task": result}
    return id_, target


if __name__ == "__main__":
    rich_print(SYSTEM_PROMPT)
    rich_print(FEW_SHOT_USER_PROMPT0)
    rich_print(FEW_SHOT_ASSISTANT_PROMPT0)
    input("Press Enter to continue...")

    rich_print(SYSTEM_PROMPT)
    rich_print(FEW_SHOT_USER_PROMPT1)
    rich_print(FEW_SHOT_ASSISTANT_PROMPT1)
    input("Press Enter to continue...")

    rich_print(SYSTEM_PROMPT)
    rich_print(FEW_SHOT_USER_PROMPT2)
    rich_print(FEW_SHOT_ASSISTANT_PROMPT2)
