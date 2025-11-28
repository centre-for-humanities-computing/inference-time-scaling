# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from config import parse_args
from thoughtminers.miners import ThoughtMiner
import os
from huggingface_hub import login
from utils import format_dataset
from dotenv import load_dotenv
from methods import forward_pass


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
                forward_pass(miner, question, question_idx, config, output_path)

            elif config.method == "kl-divergence":
                _ = miner.get_response(
                    method=config.method,
                    question=question,
                    max_reasoning_steps=config.max_reasoning_steps,
                    n_proposal=config.n_proposal,
                    is_an_answer_pattern=config.is_an_answer_pattern,
                    output_path=output_path,
                    question_idx=question_idx,
                    config=config,
                    max_sampling_steps=config.max_sampling_steps,
                )

            elif config.method == "early-stop":
                output_path = output_path / str(config.max_sampling_steps)

                _ = miner.get_response(
                    method="kl-divergence",
                    question=question,
                    max_reasoning_steps=config.max_reasoning_steps,
                    n_proposal=config.n_proposal,
                    is_an_answer_pattern=config.is_an_answer_pattern,
                    output_path=output_path,
                    question_idx=question_idx,
                    config=config,
                    max_sampling_steps=config.max_sampling_steps,
                )

            else:
                raise ValueError(f"Unknown method: {config.method}")


if __name__ == "__main__":
    main()
