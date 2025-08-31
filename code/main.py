# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import torch
import json
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from config import parse_args
from thoughtminers.miners import ThoughtMiner
import os
from huggingface_hub import login
from dataclasses import asdict
from utils import format_dataset
from dotenv import load_dotenv
import torch.nn.functional as F


def main():
    """Run the main analysis."""
    config = parse_args()  # Get config from command line

    # Logging - The Hugging Face token is expected to be in an .env file in root
    load_dotenv(Path(Path.cwd().parent, ".env"))  # Load .env file
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    login(token=token)

    print(
        f"---- Starting evaluation - Model: {config.model_id} - Dataset: {config.dataset} - Method: {config.method} - n_proposal: {config.n_proposal} ----"
    )

    # Dataset --------------------------------------------------------------------------
    if config.dataset == "math500":
        # Reads JSONL
        dataset = Dataset.from_list(
            list(
                load_dataset(
                    "json",
                    split="train",
                    data_files=str(Path.cwd().parent / "datasets" / "math500.jsonl"),
                )
            )
        )
        dataset = dataset.rename_column("problem", "question")
    elif config.dataset == "math500r":
        # Reads JSONL
        dataset = Dataset.from_list(
            list(
                load_dataset(
                    "json",
                    split="train",
                    data_files=str(
                        Path.cwd().parent
                        / "datasets"
                        / "math500_100randomquestions.jsonl"
                    ),
                )
            )
        )
        dataset = dataset.rename_column("problem", "question")
    elif config.dataset == "gsm8k-en":
        # load the GSM8K dataset
        dataset = Dataset.from_list(
            list(load_dataset("openai/gsm8k", "main", split="test"))
        )

        # load the danish translation to get the ids to keep
        dataset_da = Dataset.from_list(
            list(
                load_dataset(
                    "json",
                    split="train",
                    data_files=str(
                        Path.cwd().parent
                        / "datasets"
                        / "symbolic-translated"
                        / "*.json"
                    ),
                )
            )
        )
        ids_to_keep = dataset_da["id_orig"]

        # filter the GSM8K dataset to keep only the ids that are translated
        dataset = dataset.select(ids_to_keep)

    elif config.dataset == "gsm8k-da":
        dataset = Dataset.from_list(
            list(
                load_dataset(
                    "json",
                    split="train",
                    data_files=str(
                        Path.cwd().parent
                        / "datasets"
                        / "symbolic-translated"
                        / "*.json"
                    ),
                )
            )
        )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Instance of the thought miner
    miner = ThoughtMiner(
        model=model,
        tokenizer=tokenizer,
        prompt_style=config.promt_style,
    )

    # format the dataset
    dataset = dataset.map(
        format_dataset,
        fn_kwargs={"config": config, "miner": miner},
        batched=False,
    )

    prompts = dataset["formatted_prompt"][: config.n_questions]
    answers = dataset["answer"][: config.n_questions]
    n = len(prompts)

    for question, expected_answer, question_idx in tqdm(
        zip(prompts, answers, range(n)), total=n
    ):
        # Save to JSON
        output_path = (
            Path.cwd().parent
            / config.output_path
            / config.model_id
            / config.dataset
            / config.method
            / str(config.n_proposal)
        )

        if (
            not Path(output_path, f"{question_idx}results.json").exists()
        ) or config.overwrite:
            miner.log_data = {
                "n_question": question_idx,
                "question": question,
                "expected_answer": expected_answer,
            }

            # get the response from the model
            if config.method == "forward-pass":
                n_resp, kl_divergences = 0, []
                for _ in range(2):
                    outputs = miner.get_response(
                        question=question,
                        method=config.method,
                        max_reasoning_steps=config.max_reasoning_steps,
                        n_proposal=config.n_proposal,
                        model_id=config.model_id,
                        is_an_answer_pattern=config.is_an_answer_pattern,
                    )
                    for i in range(outputs.sequences.shape[0]):
                        model_response = "".join(
                            miner.tokenizer.batch_decode(outputs.sequences[i])
                        )
                        miner.log_data[f"model_response_{n_resp}"] = model_response
                        n_resp += 1

                    n_batch, vocab_size = outputs.logits[0].shape
                    uniform = (torch.ones(vocab_size) / vocab_size).to(
                        miner.model.device
                    )

                    for batch in range(n_batch):
                        kl_divergences.append(
                            torch.stack(
                                [
                                    F.kl_div(
                                        torch.log(
                                            F.softmax(
                                                outputs.logits[step][batch], dim=-1
                                            )
                                        ),
                                        uniform,
                                        reduction="none",
                                        log_target=False,
                                    ).sum()
                                    for step in range(len(outputs.logits))
                                ]
                            )
                        )

                Path(output_path / f"{question_idx}_logits.pt").parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(kl_divergences, output_path / f"{question_idx}_logits.pt")
            else:
                model_response = "".join(miner.tokenizer.batch_decode(outputs))
                miner.log_data["model_response"] = model_response

            # create the output directory if it does not exist
            Path(output_path, f"{question_idx}results.json").parent.mkdir(
                parents=True, exist_ok=True
            )

            # Save the results to JSON
            with open(Path(output_path, f"{question_idx}results.json"), "w") as f:
                json.dump(asdict(config) | miner.log_data, f, indent=4)


if __name__ == "__main__":
    main()
