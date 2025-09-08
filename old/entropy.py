# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import torch
from thoughtminers.miners import ThoughtMiner


def get_response_entropy(
    question: str,
    config,
    miner: ThoughtMiner,
    log_data: dict,
) -> dict:
    """Get the response to one question from the model."""
    inputs = miner.tokenizer([question], return_tensors="pt").to(miner.model.device)
    current_input_ids = inputs.input_ids

    log_data["reasoning_steps"] = {}
    for i in range(config.max_reasoning_steps):
        try:
            new_proposal, this_logs = get_new_proposal(
                miner=miner,
                current_input_ids=current_input_ids,
                config=config,
            )
        except RuntimeError:
            new_proposal = None

        if new_proposal is None:
            break

        # update the current input with the new proposal
        current_input_ids = torch.cat(
            (current_input_ids.squeeze(), new_proposal)
        ).unsqueeze(0)

        log_data["reasoning_steps"][i] = this_logs

        # if the answer was provided in the current proposal, stop here
        model_response = "".join(miner.tokenizer.batch_decode(new_proposal))
        log_data["reasoning_steps"][i]["model_response"] = model_response
        if re.search(
            config.is_an_answer_pattern,
            "".join(
                miner.tokenizer.batch_decode(
                    current_input_ids[:, inputs.input_ids.shape[1] :]
                )
            ),
        ):
            break
    log_data["model_response"] = model_response

    return log_data


def get_new_proposal(
    miner: ThoughtMiner,
    current_input_ids: torch.Tensor,
    config,
):
    """Get new reasoning step proposal.

    This function samples reasoning steps from the model and selects the best
    proposal based on the expected entropy of the reasoning steps, the expected
    dissimilarity and the curvature of the reasoning trajectory.

    Parameters
    ----------
    current_input_ids :
        Token ids of the input question.
    hidden_layer :
        Which hidden layer to use for the reasoning step. Default is -1 (last
        hidden layer).
    n_samples :
        Number of reasoning chains to sample in parallel for each proposal.

    Returns
    -------
    proposal_token_ids :
        List of token ids of the reasoning steps.
    this_logs :
        Dictionary containing the logs of the reasoning steps.

    """
    this_logs = {}
    # 1 - sample 5 reasoning steps proposals -------------------------------------------
    _, proposal_token_ids, proposal_probs = miner.sample_thoughts(
        input_ids=current_input_ids,
        num_return_sequences=config.n_proposal,
    )

    if proposal_token_ids is None or len(proposal_token_ids) == 0:
        return None, None

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    this_logs["n_tokens"] = n_tokens

    # trajectory-level entropy
    proposals_entropies = torch.stack(
        [
            -torch.log(
                torch.cat(
                    [
                        this_prob.gather(0, this_tokens.unsqueeze(-1))
                        for this_prob, this_tokens in zip(
                            proposal_probs[n_batch], proposal_token_ids[n_batch]
                        )
                    ]
                )
            ).sum()
            / proposal_token_ids[n_batch].shape[0]
            for n_batch in range(config.n_proposal)
        ]
    )

    # estimate the entropy of the proposals
    this_logs["proposals_entropies"] = (
        proposals_entropies.detach().cpu().float().tolist()
    )

    # 2 - Select best step proposal

    # sum the entropies ans scale by cosine dissimilarities - lower = better
    weigths = proposals_entropies
    select_id = torch.argmin(weigths)

    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    return proposal_token_ids[select_id], this_logs
