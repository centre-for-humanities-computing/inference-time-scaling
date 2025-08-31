# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pandas as pd
import torch
import re
import json
from pathlib import Path
from grader import math_equal
from qwen_math_parser import extract_answer, strip_string
from transformers import AutoTokenizer
from thoughtminers.miners import Config
import numpy as np
from tqdm import tqdm

results_df = pd.DataFrame([])
config = Config()

for model_id in [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for dataset in tqdm(["math500", "gsm8k-en", "gsm8k-da"]):
        results_path = (
            Path.cwd().parent / "results" / model_id / dataset / "forward-pass" / "4"
        )

        for file in tqdm(results_path.glob("*.json")):
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

                # extract KL divergences for each reasoning step
                response_length = len(kl[question_idx])
                tokens = tokenizer.encode(model_response)[-response_length:]
                if tokenizer.encode("<|im_end|>")[0] in tokens:
                    response_ends = tokens.index(tokenizer.encode("<|im_end|>")[0])
                    tokens = np.array(tokens[:response_ends])
                    divergences = kl[question_idx][:response_ends].cpu().numpy()

                    reasoning_token_ids = [
                        tokenizer.encode(reasoning_token)[0]
                        for reasoning_token in config.reasoning_token
                    ]
                    split_vector = np.where(
                        np.array([tokens == rt for rt in reasoning_token_ids]).any(
                            axis=0
                        )
                    )[0]

                    reasonin_step = 0
                    for start_idx, stop_idx in zip(
                        np.insert(split_vector, 0, 0),
                        np.append(split_vector, len(divergences)),
                    ):
                        average_divergence = divergences[start_idx:stop_idx].mean()

                        results_df = pd.concat(
                            [
                                results_df,
                                pd.DataFrame(
                                    {
                                        "model_id": model_id,
                                        "dataset": dataset,
                                        "question_idx": question_idx,
                                        "accuracy": [accuracy],
                                        "reasonin_step": [reasonin_step],
                                        "average_divergence": [average_divergence],
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )

                        reasonin_step += 1

results_df.to_csv(Path.cwd().parent / "results" / "merged_results.csv", index=False)
