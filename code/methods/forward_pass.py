# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pathlib import Path
import torch
import json
from torch.nn import functional as F
from dataclasses import asdict


def forward_pass(miner, question, question_idx, config, output_path):
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
            model_response = "".join(miner.tokenizer.batch_decode(outputs.sequences[i]))
            miner.log_data[f"model_response_{n_resp}"] = model_response
            n_resp += 1

        n_batch, vocab_size = outputs.logits[0].shape
        uniform = (torch.ones(vocab_size) / vocab_size).to(miner.model.device)

        for batch in range(n_batch):
            kl_divergences.append(
                torch.stack(
                    [
                        F.kl_div(
                            torch.log(F.softmax(outputs.logits[step][batch], dim=-1)),
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

    # create the output directory if it does not exist
    Path(output_path, f"{question_idx}results.json").parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save the results to JSON
    with open(Path(output_path, f"{question_idx}results.json"), "w") as f:
        json.dump(asdict(config) | miner.log_data, f, indent=4)
