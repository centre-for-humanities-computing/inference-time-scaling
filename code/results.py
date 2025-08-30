import json
import pandas as pd
from pathlib import Path
from qwen_math_parser import extract_answer, strip_string
from grader import math_equal

# load baseline scores

result_df = pd.DataFrame([])

for method in [
    "baseline",
    "entropy",
    "expected-entropy",
    "curvature",
    "expected-similarity",
]:
    result_path = (
        Path.cwd()
        / "results"
        / "Qwen"
        / "Qwen2.5-Math-1.5B-Instruct"
        / method
        / "outputs"
    )
    for file in result_path.glob("*.json"):
        # load JSON files
        with open(file, "r") as f:
            data = json.load(f)

        model_response = data["model_response"]
        answer = strip_string(data["answer"])
        model_output = extract_answer(model_response, data_name="math500")
        accuracy = math_equal(model_output, answer)

        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "method": method,
                        "file": file.name,
                        "model_response": model_response,
                        "expected_answer": answer,
                        "model_output": model_output,
                        "accuracy": accuracy,
                        "n_question": data["n_question"],
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
