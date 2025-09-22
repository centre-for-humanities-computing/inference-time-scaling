# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import torch
from typing import TYPE_CHECKING
from qwen_math_parser import extract_answer
from grader import math_equal

if TYPE_CHECKING:
    from thoughtminers.miners import ThoughtMiner


def self_consistency(
    miner: "ThoughtMiner", question: str, num_return_sequences: int = 2, **kwargs
) -> torch.Tensor:
    """Get the response to one question from the model using n forward pass."""
    inputs = miner.tokenizer([question], return_tensors="pt").to(miner.model.device)

    outputs = miner.model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=2000,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=False,
        output_logits=False,
        output_hidden_states=False,
        output_attentions=False,
    )

    # Collect the model responses
    model_output = []
    for i in range(num_return_sequences):
        model_response = "".join(miner.tokenizer.batch_decode(outputs[i, :]))
        model_output.append(extract_answer(model_response, data_name="math500"))

    # Compute the self-consistency by comparing the model responses with each other
    similarities = []
    for resp in model_output:
        similarities.append(sum([math_equal(resp, out) for out in model_output]))

    # select the most similar response (here the first in the list)
    idx = similarities.index(max(similarities))

    return outputs[idx, :]
