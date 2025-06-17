import re
from typing import Any

from fuzzywuzzy import fuzz
from data_alchemy.datasets_ import RootSample, Message, Sample
from data_alchemy.utils.auto_resume import model_cache
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger

MODEL = "qwen20-72b-instruct-whh"

PROGRAMMING_LANGUAGES = {
    "Bash": "Bash command(s) and shell scripting",
    "C": "Standard C programming language. Notice pure C does not support C++ features",
    "C#": "C# programming language developed by Microsoft. Do not call it 'C Sharp'. Just 'C#'",
    "C++": "Standard C++ programming language. Notice the difference between C++ and C",
    "Fortran": "Fortran programming language",
    "Go": "Go programming language",
    "HTML": "HTML (HyperText Markup Language), the standard markup language used for creating web pages and web applications.",
    "Java": "Java programming language and its extensions, such as Android development, JSP, etc.",
    "JavaScript": "JavaScript programming language and its extensions, such as Node.js, React, Vue.js, etc. Just call them 'JavaScript'.",
    "Kotlin": "Kotlin programming language, commonly used for Android development and modern JVM-based applications.",
    "LaTeX": "LaTeX typesetting system, widely used for creating scientific and technical documents, especially for mathematics and academic papers.",
    "LISP": "LISP programming language, known for its powerful macro system and use in AI research.",
    "Lua": "Lua programming language, lightweight and often used for scripting in games and embedded systems.",
    "MATLAB": "MATLAB programming language, primarily used for numerical computation, data analysis, and visualization.",
    "Objective-C": "Objective-C programming language, traditionally used for macOS and iOS development before Swift.",
    "Pascal": "Pascal programming language and its extensions, such as Delphi.",
    "Perl": "Perl programming language, known for text processing and system administration tasks.",
    "PHP": "PHP programming language, widely used for web development and server-side scripting.",
    "Python": "Python programming language, known for its simplicity and versatility across many domains.",
    "R": "R programming language, primarily used for statistical computing, data analysis, and visualization.",
    "Ruby": "Ruby programming language, commonly used for web development, especially with Ruby on Rails.",
    "Rust": "Rust programming language, known for its focus on safety, concurrency, and performance.",
    "SQL": "SQL (Structured Query Language), used for querying and managing relational databases.",
    "Swift": "Swift programming language, used for macOS and iOS development, known for its modern features.",
    "TypeScript": "TypeScript programming language, a typed superset of JavaScript commonly used for large-scale web development. Notice the difference between TypeScript and JavaScript.",
    "Visual Basic": "Visual Basic programming language, often used for Windows application development with .NET.",
    # languages not considerd
    "Other": "Any programming language not listed above. If unsure, predict 'Other'.",
}

NEWLINE = "\n"

SYSTEM_PROMPT = f"""# Requirement:
You will receive a piece of conversation text with code snippets in it.
Your task is to act as a programming language classifier.
The following programming languages are considered:
{NEWLINE.join(f"- **{k}**: {v}" for k, v in PROGRAMMING_LANGUAGES.items())}.

Data format protocols like JSON, YAML, XML, etc. are not considered as programming languages.

# Notes:
1. Before giving final answers, please provide a brief reason for your classification.
2. Provide your answer strictly in the format:
   `'reason: <reason>\nprogramming language: <programming language>'`
3. Predict only one programming language at a time. If multiple languages are possible, predict the one that is most likely.
4. Output <programming language> with the names listed above.
5. Do not output any other information or comments in the <programming language> part.
6. If the code is not in any of the mentioned languages, predict 'Other'.
"""

ONE_SHOT_USER_PROMPT = """\
Human:
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

### Example:
For the input `[3, 6, 8, 10, 1, 2, 1]`, it will output the sorted list: `[1, 1, 2, 3, 6, 8, 10]`.

Let me know if you'd like further explanation or adjustments!
"""

ONE_SHOT_ASSISTANT_PROMPT = (
    "reason: the 'Human' role explicitly mentioned using Python and the markdown code block also mentioned Python\n"
    "programming language: Python"
)

achat = model_cache(cache_dir=".cache/achat")(achat)

async def model_call(history: list[Message], model: str) -> Message:
    plain_text = "\n".join(f"{msg.role}:\n{msg.content}\n" for msg in history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ONE_SHOT_USER_PROMPT},
        {"role": "assistant", "content": ONE_SHOT_ASSISTANT_PROMPT},
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

    res = re.match(r"(.*)\nprogramming language: (.*)", text, re.DOTALL)
    assert res, f"Regex Parsing Error, got '{text}'"

    _, lang = res.groups()
    lang = lang.strip().lower()
    assert len(lang) < 42, f"Output too long: {lang}"

    languages = list(pl.lower() for pl in PROGRAMMING_LANGUAGES)
    similarities = [fuzz.ratio(lang, pl.lower()) for pl in PROGRAMMING_LANGUAGES]
    max_similarity = max(similarities)
    idx = similarities.index(max_similarity)
    result = languages[idx]

    assert max_similarity > 50, f"Similarity too low: sim({lang}, {result})={max_similarity}"

    if result in lang:
        return result
    return "other"


async def classify_programming_language(id_: int, source: RootSample) -> tuple[int, RootSample]:
    messages = source.root

    try:
        response = await model_call(history=messages, model=MODEL)
        result = parse_result(response)
    except AssertionError as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    messages[0].tag = {"programming_language": result}
    target = Sample(root=messages)
    return id_, target
