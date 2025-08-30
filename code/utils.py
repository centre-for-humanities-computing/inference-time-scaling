# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import math
from sympy import simplify, N
from sympy.parsing.latex import parse_latex


def score_gsm8k(model_answer: str, expected_answer: str) -> int:
    """Score the answer based on the expected answer."""
    match = re.search(r"\n#### (\d+)$", answer)
    if match:
        expected_answer = match.group(1)

    return score


def score_math500(model_answer: str, expected_answer: str) -> int:
    """Score the answer based on the expected answer."""
    return score


def format_dataset(sample, config, miner):
    """Format the dataset for the model."""
    question = sample["question"]
    system = [
        {
            "role": "system",
            "content": config.promt_style,
        }
    ]

    prompt = miner.tokenizer.apply_chat_template(
        system + [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sample["formatted_prompt"] = prompt

    return sample


def scorer():
    """Score the answer based on the expected answer."""
    model_answer = re.findall(r"\\boxed\{([^}]+)\}", log_data["model_response"])[0]

    import regex

    pattern = r"\\boxed\{(?P<content>(?:[^{}]|(?P>content))*\}"
    match = regex.search(pattern, log_data["model_response"])
    if match:
        print(match.group("content"))

    if config.dataset == "gsm8k":
        accuracy = score_gsm8k(model_answer=model_answer, expected_answer=answer)
    elif config.dataset == "math500":
        accuracy = score_math500(model_answer=model_answer, expected_answer=answer)

    logging.info(f"Accuracy: {accuracy}")
