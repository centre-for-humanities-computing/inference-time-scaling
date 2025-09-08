# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import logging
import torch
from thoughtminers.miners import ThoughtMiner
from config import Config


def get_response_expected_entropy(
    question: str,
    config,
    miner: ThoughtMiner,
    log_data: dict,
    logger: logging.Logger = logging.getLogger(__name__),
) -> dict:
    """Get the response to one question from the model."""
    logger.info(f"Question: {question}")
    inputs = miner.tokenizer([question], return_tensors="pt").to(miner.model.device)
    current_input_ids = inputs.input_ids

    log_data["reasoning_steps"] = {}
    for i in range(config.max_reasoning_steps):
        try:
            new_proposal, this_logs = get_new_proposal(
                miner=miner,
                current_input_ids=current_input_ids,
                logger=logger,
                config=config,
            )
        except RuntimeError:
            new_proposal = None

        if new_proposal is None:
            logger.info("No new proposal generated, stopping.")
            break

        # update the current input with the new proposal
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
        num_return_sequences=num_return_sequences_proposal,
        hidden_layer=hidden_layer,
        save_outputs=False,
    )

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    logging.info(
        f"Sampling {num_return_sequences_proposal} proposals - n_tokens = {n_tokens}"
    )
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
            for n_batch in range(num_return_sequences_proposal)
        ]
    )

    # estimate the entropy of the proposals
    logger.info(f"Token output entropy: {proposals_entropies}.")
    this_logs["proposals_entropies"] = (
        proposals_entropies.detach().cpu().float().tolist()
    )

    # 2 - Get an estimate of the expected entropy for each of these steps
    average_entropies = []
    for proposal_token, n in zip(
        proposal_token_ids, range(num_return_sequences_proposal)
    ):
        # if the proposal is a final answer, stop here
        if re.search(
            config.is_an_answer_pattern,
            "".join(miner.tokenizer.batch_decode(proposal_token)),
        ):
            average_entropies.append(torch.tensor(0.0).to(miner.model.device))
            continue

        # add the proposal step to the input
        concat_inputs = torch.cat(
            (current_input_ids.squeeze(), proposal_token)
        ).unsqueeze(0)

        # sample possible proposals from this step -------------------------------------
        continuous_step, tokens, probs = miner.sample_thoughts(
            input_ids=concat_inputs,
            num_return_sequences=num_return_sequences_expected,
            save_outputs=False,
        )

        logger.info(
            (
                f"--- Proposal {n} - Sampling {num_return_sequences_expected} expected steps - "
                f"n_tokens = {[p.shape[0] for p in probs]} ---"
            )
        )

        # early stop if no steps or missing steps are sampled
        if not continuous_step:
            average_entropies.append(torch.tensor(0.0).to(miner.model.device))
            continue

        # trajectory-level entropy
        average_entropies.append(
            torch.stack(
                [
                    -torch.log(
                        torch.cat(
                            [
                                this_prob.gather(0, this_tokens.unsqueeze(-1))
                                for this_prob, this_tokens in zip(
                                    probs[n_batch], tokens[n_batch]
                                )
                            ]
                        )
                    ).sum()
                    / tokens[n_batch].shape[0]
                    for n_batch in range(num_return_sequences_expected)
                ]
            ).mean()
        )
    logger.info(f"Proposal expected entropies: {torch.stack(average_entropies)}")

    # save in log data
    this_logs["average_entropies"] = (
        torch.stack(average_entropies).detach().cpu().float().tolist()
    )

    # 3 - Select best step proposal

    # sum the entropies - lower = better
    weigths = proposals_entropies + torch.stack(average_entropies)
    select_id = torch.argmin(weigths)

    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    logger.info(f"Final weights: {weigths.detach().cpu().float().tolist()}")
    logger.info(
        f"Selecting proposal number {select_id.detach().cpu().float().tolist()}"
    )

    return proposal_token_ids[select_id], this_logs
