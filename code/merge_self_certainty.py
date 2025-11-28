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
from dotenv import load_dotenv
from huggingface_hub import login
import os

results_df = pd.DataFrame([])
config = Config()

# Logging - The Hugging Face token is expected to be in an .env file in root
load_dotenv(Path(Path.cwd().parent, ".env"))  # Load .env file
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=token)


for model_id in [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
        results_path = (
            Path.cwd().parent / "results" / model_id / dataset / "forward-pass" / "4"
        )

        for file_path in tqdm(results_path.glob("*.json")):
            with open(file_path, "r") as f:
                data = json.load(f)

            kl = torch.load(
                results_path / f"{re.search(r'\d+', file_path.name).group()}_logits.pt"
            )

            for response_number in range(8):
                model_response = data[f"model_response_{response_number}"]
                if dataset in ["gsm8k-en", "gsm8k-da"]:
                    expected_answer = strip_string(
                        re.search(r"####(.*)$", data["expected_answer"]).group(1)
                    )
                else:
                    expected_answer = strip_string(data["expected_answer"])
                model_output = extract_answer(model_response, data_name=dataset)
                accuracy = math_equal(model_output, expected_answer)

                # extract KL divergences for each reasoning step
                response_length = len(kl[response_number])
                tokens = tokenizer.encode(model_response)[-response_length:]
                if tokenizer.encode("<|im_end|>")[0] in tokens:
                    response_ends = tokens.index(tokenizer.encode("<|im_end|>")[0])
                elif tokenizer.encode("<|end_of_text|>")[1] in tokens:
                    response_ends = tokens.index(tokenizer.encode("<|end_of_text|>")[1])
                else:
                    pass

                tokens = np.array(tokens[:response_ends])
                divergences = kl[response_number][:response_ends].cpu().numpy()

                if model_id[:10] == "meta-llama":
                    reasoning_token_ids = [
                        tokenizer.encode(reasoning_token)[1]
                        for reasoning_token in config.reasoning_token
                        if len(tokenizer.encode(reasoning_token)) == 2
                    ]
                else:
                    reasoning_token_ids = [
                        tokenizer.encode(reasoning_token)[0]
                        for reasoning_token in config.reasoning_token
                    ]

                split_vector = np.where(
                    np.array([tokens == rt for rt in reasoning_token_ids]).any(axis=0)
                )[0]

                reasoning_step, n_reasoning_steps = 0, len(split_vector) + 1
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
                                    "file_path": file_path.name,
                                    "response_number": response_number,
                                    "accuracy": [accuracy],
                                    "reasoning_step": [reasoning_step],
                                    "average_divergence": [average_divergence],
                                    "n_reasoning_steps": [n_reasoning_steps],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

                    reasoning_step += 1

results_df.to_csv(
    Path.cwd().parent / "results" / "self_certainty_forward_pass.csv", index=False
)


results_df = pd.DataFrame([])
for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
    for model, model_path in zip(
        [
            "Qwen2.5-0.5B-Instruct",
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
        ],
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
    ):
        for n_samples in [2, 4, 8]:
            result_path = (
                Path.cwd().parent
                / "results"
                / model_path
                / dataset
                / "kl-divergence"
                / str(n_samples)
            )
            for file_path in tqdm(result_path.glob("*.json")):
                # load JSON files
                with open(file_path, "r") as f:
                    data = json.load(f)

                model_response = data["model_response"]

                if dataset in ["gsm8k-en", "gsm8k-da"]:
                    answer = strip_string(
                        re.search(r"####(.*)$", data["expected_answer"]).group(1)
                    )
                else:
                    answer = strip_string(data["expected_answer"])
                model_output = extract_answer(model_response, data_name=dataset)
                accuracy = math_equal(model_output, answer)

                reasoning_steps = len(data["reasoning_steps"])

                for key in data["reasoning_steps"].keys():
                    try:
                        kls = np.array(
                            data["reasoning_steps"][key]["proposals_KL_divergence"]
                        )
                        kl = kls.max()
                        kl_benefit = (kls.max() - kls).mean()

                    except KeyError:
                        kl = np.nan
                        kl_benefit = np.nan
                        kl_std = np.nan

                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame(
                                {
                                    "method": "kl-divergence",
                                    "model": model,
                                    "dataset": dataset,
                                    "n_samples": n_samples,
                                    "file_path": file_path.name,
                                    "model_response": model_response,
                                    "expected_answer": answer,
                                    "model_output": model_output,
                                    "accuracy": "Correct" if accuracy else "Incorrect",
                                    "n_question": data["n_question"],
                                    "n_reasoning_steps": reasoning_steps,
                                    "reasoning_step": key,
                                    "kl": kl,
                                    "kl_benefit": kl_benefit,
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )

results_df.to_csv(
    Path.cwd().parent / "results" / "self_certainty_kl_divergences.csv", index=False
)
