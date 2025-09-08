# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import logging
import torch
from thoughtminers.miners import ThoughtMiner
import torch.nn.functional as F
from config import Config


def get_response_entropy_weigthed_similarity(
    question: str,
    config: Config,
    miner: ThoughtMiner,
    log_data: dict,
    logger: logging.Logger = logging.getLogger(__name__),
) -> dict:
    """Get the response to one question from the model."""
    logger.info(f"Question: {question}")
    inputs = miner.tokenizer([question], return_tensors="pt").to(miner.model.device)
    current_input_ids = inputs.input_ids
    previous_input_length = 0

    log_data["reasoning_steps"] = {}
    for i in range(config.max_reasoning_steps):
        new_proposal, this_logs = get_new_proposal(
            miner=miner,
            config=config,
            current_input_ids=current_input_ids,
            previous_input_length=previous_input_length,
            num_return_sequences_proposal=5,
            logger=logger,
        )

        # update the current input with the new proposal
        previous_input_length = current_input_ids.shape[1]  # save previous step size
        current_input_ids = torch.cat(
            (current_input_ids.squeeze(), new_proposal)
        ).unsqueeze(0)

        log_data["reasoning_steps"][i] = this_logs

        # if the answer was provided in the current proposal, stop here
        model_response = "".join(miner.tokenizer.batch_decode(new_proposal))
        logger.info(f"Proposal: {model_response}")
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
    config: Config,
    current_input_ids: torch.Tensor,
    previous_input_length: int,
    num_return_sequences_proposal: int = 10,
    hidden_layer: int = -1,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Get new reasoning step proposal.

    This function samples reasoning steps from the model and selects the best
    proposal based on the expected entropy of the reasoning steps, the expected
    dissimilarity and the curvature of the reasoning trajectory.

    Parameters
    ----------
    current_input_ids :
        Token ids of the input question.
    n_proposal :
        Number of reasoning chains to sample in parallel.
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
    continuous_step, proposal_token_ids, proposal_probs = miner.sample_thoughts(
        input_ids=current_input_ids,
        num_return_sequences=num_return_sequences_proposal,
        hidden_layer=hidden_layer,
    )

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    logging.info(
        f"Sampling {num_return_sequences_proposal} proposals - n_tokens = {n_tokens}"
    )
    this_logs["n_tokens"] = n_tokens

    # 2 - trajectory-level entropy -----------------------------------------------------
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
            for n_batch in range(num_return_sequences_proposal)
        ]
    )

    # estimate the entropy of the proposals
    logger.info(f"Token output entropy: {proposals_entropies}.")
    this_logs["proposals_entropies"] = (
        proposals_entropies.detach().cpu().float().tolist()
    )

    # 3 - proposal dissimilarity -------------------------------------------------------
    # get the embeddings of the previous proposals - all the same across the batch
    h_0 = torch.stack([tk for tk in miner.outputs.hidden_states[0][hidden_layer]])[
        :, previous_input_length:, :
    ].mean(1)
    similarities = torch.cat(
        [
            F.cosine_similarity(h_0[1].unsqueeze(0), y.unsqueeze(0))
            for y in continuous_step
        ]
    )
    logger.info(f"Proposal cosine similarities: {similarities}.")
    this_logs["proposals_similarities"] = similarities.detach().cpu().float().tolist()

    # 4 - Select best step proposal
    # weigth the entropies with cosine similarities - lower = better
    weigths = proposals_entropies * similarities
    select_id = torch.argmin(weigths)

    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    logger.info(f"Final weights: {weigths.detach().cpu().float().tolist()}")
    logger.info(
        f"Selecting proposal number {select_id.detach().cpu().float().tolist()}"
    )

    return proposal_token_ids[select_id], this_logs
