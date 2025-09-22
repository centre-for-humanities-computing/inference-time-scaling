# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pathlib import Path
import pandas as pd
import json
import itertools
from qwen_math_parser import extract_answer, strip_string
from grader import math_equal
import re
from tqdm import tqdm

result_df = pd.DataFrame([])
for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
    for model_id in [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]:
        result_path = (
            Path.cwd().parent / "results" / model_id / dataset / "forward-pass" / "4"
        )

        for file_path in tqdm(result_path.glob("*.json")):
            # load JSON files
            with open(file_path, "r") as f:
                data = json.load(f)
            for response_number in range(8):
                model_response = data[f"model_response_{response_number}"]

                if dataset in ["gsm8k-en", "gsm8k-da"]:
                    answer = strip_string(
                        re.search(r"####(.*)$", data["expected_answer"]).group(1)
                    )
                else:
                    answer = strip_string(data["expected_answer"])
                model_output = extract_answer(model_response, data_name=dataset)
                accuracy = math_equal(model_output, answer)

                result_df = pd.concat(
                    [
                        result_df,
                        pd.DataFrame(
                            {
                                "method": "forward-pass",
                                "model_id": model_id,
                                "dataset": dataset,
                                "n_samples": 1,
                                "file_path": file_path.name,
                                "model_response": model_response,
                                "response_number": response_number,
                                "expected_answer": answer,
                                "model_output": model_output,
                                "accuracy": accuracy,
                                "n_question": data["n_question"],
                                "n_reasoning_steps": [1],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )


for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
    for model_id in [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]:
        for n_samples in [2, 4, 8]:
            result_path = (
                Path.cwd().parent
                / "results"
                / model_id
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

                result_df = pd.concat(
                    [
                        result_df,
                        pd.DataFrame(
                            {
                                "method": "kl-divergence",
                                "model_id": model_id,
                                "dataset": dataset,
                                "n_samples": n_samples,
                                "file_path": file_path.name,
                                "model_response": model_response,
                                "response_number": [1],
                                "expected_answer": answer,
                                "model_output": model_output,
                                "accuracy": accuracy,
                                "n_question": data["n_question"],
                                "n_reasoning_steps": reasoning_steps,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

for dataset in ["math500", "gsm8k-en", "gsm8k-da"]:
    for model_id in [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]:
        result_path = (
            Path.cwd().parent / "results" / model_id / dataset / "forward-pass" / "4"
        )

        for file_path in tqdm(result_path.glob("*.json")):
            # load JSON files
            with open(file_path, "r") as f:
                data = json.load(f)

            if dataset in ["gsm8k-en", "gsm8k-da"]:
                answer = strip_string(
                    re.search(r"####(.*)$", data["expected_answer"]).group(1)
                )
            else:
                answer = strip_string(data["expected_answer"])

            for n_samples in [2, 4, 8]:
                combinations = list(itertools.combinations(range(8), n_samples))

                for response_number, combination in enumerate(combinations[:5]):
                    model_output = []
                    for resp_idx in combination:
                        model_response = data[f"model_response_{resp_idx}"]
                        model_output.append(
                            extract_answer(model_response, data_name=dataset)
                        )

                    # Compute the self-consistency by comparing the model responses with each other
                    similarities = []
                    for resp in model_output:
                        similarities.append(
                            sum([math_equal(resp, out) for out in model_output])
                        )

                    # select the most similar response (or the first in the list)
                    idx = similarities.index(max(similarities))

                    model_response = data[f"model_response_{idx}"]

                    model_output = extract_answer(model_response, data_name=dataset)
                    accuracy = math_equal(model_output, answer)

                    result_df = pd.concat(
                        [
                            result_df,
                            pd.DataFrame(
                                {
                                    "method": "majority-vote",
                                    "model_id": model_id,
                                    "dataset": dataset,
                                    "n_samples": n_samples,
                                    "file_path": file_path.name,
                                    "model_response": model_response,
                                    "response_number": response_number,
                                    "expected_answer": answer,
                                    "model_output": model_output,
                                    "accuracy": accuracy,
                                    "n_question": data["n_question"],
                                    "n_reasoning_steps": [1],
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )

result_df.to_csv(Path.cwd().parent / "results" / "merged_accuracy.csv", index=False)
