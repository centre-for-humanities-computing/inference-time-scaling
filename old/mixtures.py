# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import logging
import torch
from thoughtminers.miners import ThoughtMiner
import torch.nn.functional as F


def get_response_entropy(
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
        new_proposal, this_logs = get_new_proposal(
            miner=miner,
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
        if re.search(config.is_an_answer_pattern, model_response):
            break

    return log_data


def get_new_proposal(
    miner: ThoughtMiner,
    current_input_ids: torch.Tensor,
    n_proposal: int = 10,
    hidden_layer: int = -1,
    n_samples: int = 20,
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
    proposal_continuous_steps, proposal_token_ids, proposal_probs = (
        miner.sample_thoughts(
            input_ids=current_input_ids,
            num_return_sequences=n_proposal,
        )
    )

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    logging.info(f"Sampling {n_proposal} proposals - n_tokens = {n_tokens}")
    this_logs["n_tokens"] = n_tokens

    proposals_entropies = torch.stack(
        [-(ps * torch.log(ps)).sum() / ps.shape[0] for ps in proposal_probs]
    )

    # estimate the entropy of the proposals
    logger.info(f"Token output entropy: {proposals_entropies}.")
    this_logs["proposals_entropies"] = (
        proposals_entropies.detach().cpu().float().tolist()
    )

    # 2 - Get an estimate of the expected entropy for each of these steps
    average_cosine_similarities, average_entropies, average_curvature = [], [], []
    for proposal_continuous_step, proposal_token, n in zip(
        proposal_continuous_steps, proposal_token_ids, range(n_proposal)
    ):
        # add the proposal step to the input
        concat_inputs = torch.cat(
            (current_input_ids.squeeze(), proposal_token)
        ).unsqueeze(0)

        # sample possible proposals from this step ---------------------------------
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

        # early stop if no steps or missing steps are sampled
        if not continuous_step:
            average_cosine_similarities.append(torch.tensor(1.0).to(miner.model.device))
            average_entropies.append(torch.tensor(0.0).to(miner.model.device))
            average_curvature.append(torch.tensor(0.0).to(miner.model.device))

            continue

        # average and store cosine similarities ------------------------------------
        embeddings = F.normalize(torch.stack(continuous_step), p=2, dim=1)
        cosine_similarities = torch.mm(embeddings, embeddings.t())
        average_cosine_similarities.append(
            torch.triu(cosine_similarities, diagonal=1).mean()
        )

        # estimate the entropy of the proposals ------------------------------------
        average_entropies.append(
            torch.stack(
                [-(ps * torch.log(ps)).sum() / ps.shape[0] for ps in probs]
            ).mean()
        )

        # estimate the curvature of the proposals ----------------------------------

        # hidden state embedding of the input question
        h_0 = torch.stack(
            [hs[hidden_layer][n] for hs in miner.outputs.hidden_states[0]]
        ).mean(0)

        # hidden state embedding of the proposal
        h_1 = proposal_continuous_step[n]

        # hidden state embedding of the expected future steps
        h_2 = torch.stack(continuous_step)
        average_curvature.append(torch.norm(h_2 - 2 * h_1 + h_0, dim=1).mean())

    logger.info(
        f"Average cosine similarities: {torch.stack(average_cosine_similarities)}"
    )
    logger.info(f"Proposal expected entropies: {torch.stack(average_entropies)}")
    logger.info(f"Proposal expected curvatures: {torch.stack(average_curvature)}")

    # save in log data
    this_logs["average_cosine_similarities"] = (
        torch.stack(average_cosine_similarities).detach().cpu().float().tolist()
    )
    this_logs["average_entropies"] = (
        torch.stack(average_entropies).detach().cpu().float().tolist()
    )
    this_logs["average_curvature"] = (
        torch.stack(average_curvature).detach().cpu().float().tolist()
    )

    # 3 - Select best step proposal

    # sum the entropies ans scale by cosine dissimilarities - lower = better
    weigths = (
        (proposals_entropies + torch.stack(average_entropies))
        * (1 - torch.stack(average_cosine_similarities))
        * torch.softmax(torch.stack(average_curvature), dim=0)
    )
    logger.info(f"Final weights: {weigths.detach().cpu().float().tolist()}")
    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    select_id = torch.argmin(weigths)
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    logger.info(
        f"Selecting proposal number {select_id.detach().cpu().float().tolist()}"
    )

    return proposal_token_ids[select_id], this_logs
