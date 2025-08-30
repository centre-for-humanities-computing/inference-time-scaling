# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pandas as pd
import torch
import re
import json
from pathlib import Path
from grader import math_equal
from qwen_math_parser import extract_answer, strip_string
from transformers import AutoTokenizer

results_df = pd.DataFrame([])

for model_id in [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
        results_path = (
            Path.cwd().parent / "results" / model_id / dataset / "forward-pass" / "4"
        )

        for file in results_path.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)

            kl = torch.load(
                results_path / f"{re.search(r'\d+', file.name).group()}_logits.pt"
            )

            for question_idx in range(8):
                model_response = data[f"model_response_{question_idx}"]
                if dataset in ["gsm8k-en", "gsm8k-da"]:
                    expected_answer = strip_string(
                        re.search(r"####(.*)$", data["expected_answer"]).group(1)
                    )
                else:
                    expected_answer = strip_string(data["expected_answer"])
                expected_answer = strip_string(data["expected_answer"])
                model_output = extract_answer(model_response, data_name=dataset)
                accuracy = math_equal(model_output, expected_answer)

                # extract KL divergences
                mask = [
                    el in tokenizer.all_special_ids
                    for el in tokenizer.encode(model_response)
                ]
                kl[question_idx]

                [
                    tokenizer.encode("user")[0] == el
                    for el in tokenizer.encode(model_response)
                ]

                tokenizer.encode(model_response).index(tokenizer.encode("user")[0]) + 1
