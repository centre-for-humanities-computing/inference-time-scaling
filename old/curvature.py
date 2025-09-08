# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import logging
import torch
from thoughtminers.miners import ThoughtMiner
from config import Config


def get_response_curvature(
    question: str,
    config: Config,
    miner: ThoughtMiner,
    log_data: dict,
    logger: logging.Logger = logging.getLogger(__name__),
) -> dict:
    """Get the response to one question from the model."""
    inputs = miner.tokenizer([question], return_tensors="pt").to(miner.model.device)
    current_input_ids = inputs.input_ids

    log_data["reasoning_steps"] = {}
    for i in range(config.max_reasoning_steps):
        new_proposal, this_logs = get_new_proposal(
            miner=miner,
            config=config,
            current_input_ids=current_input_ids,
            n_proposal=5,
            n_samples=5,
            logger=logger,
        )

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
    n_proposal: int = 5,
    hidden_layer: int = -1,
    n_samples: int = 5,
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
    proposal_continuous_steps, proposal_token_ids, _ = miner.sample_thoughts(
        input_ids=current_input_ids,
        num_return_sequences=n_proposal,
    )

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    logging.info(f"Sampling {n_proposal} proposals - n_tokens = {n_tokens}")
    this_logs["n_tokens"] = n_tokens

    # 2 - Get an estimate of the expected entropy for each of these steps
    average_curvature = []
    for proposal_continuous_step, proposal_token, n in zip(
        proposal_continuous_steps, proposal_token_ids, range(n_proposal)
    ):
        # if the proposal is a final answer, stop here
        if re.search(
            config.is_an_answer_pattern,
            "".join(miner.tokenizer.batch_decode(proposal_token)),
        ):
            average_curvature.append(torch.tensor(0.0).to(miner.model.device))
            continue

        # add the proposal step to the input
        concat_inputs = torch.cat(
            (current_input_ids.squeeze(), proposal_token)
        ).unsqueeze(0)

        # sample possible proposals from this step -------------------------------------
        continuous_step, _, probs = miner.sample_thoughts(
            input_ids=concat_inputs,
            num_return_sequences=n_samples,
            save_outputs=False,
        )

        logger.info(
            (
                f"--- Proposal {n} - Sampling {n_samples} expected steps - "
                f"n_tokens = {[p.shape[0] for p in probs]} ---"
            )
        )

        # if the next step is empty, stop here
        if not continuous_step:
            average_curvature.append(torch.tensor(0.0).to(miner.model.device))
            continue

        # estimate the curvature of the proposals --------------------------------------
        # hidden state embedding of the input question
        h_0 = torch.stack(
            [tk for tk in miner.outputs.hidden_states[0][hidden_layer][n]]
        )
        assert h_0.shape[0] == current_input_ids.shape[1]
        h_0 = h_0.mean(0)

        # hidden state embedding of the proposal
        h_1 = proposal_continuous_step

        # hidden state embedding of the expected future steps
        h_2 = torch.stack(continuous_step)
        average_curvature.append(torch.norm(h_2 - 2 * h_1 + h_0, dim=1).mean())

    logger.info(f"Proposal expected curvatures: {torch.stack(average_curvature)}")

    # save in log data
    this_logs["average_curvature"] = (
        torch.stack(average_curvature).detach().cpu().float().tolist()
    )

    # 3 - Select best step proposal

    # average expected curvature - lower = better
    weigths = torch.stack(average_curvature)
    select_id = torch.argmin(weigths)

    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    logger.info(f"Final weights: {weigths.detach().cpu().float().tolist()}")
    logger.info(
        f"Selecting proposal number {select_id.detach().cpu().float().tolist()}"
    )

    return proposal_token_ids[select_id], this_logs
