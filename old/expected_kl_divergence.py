# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import re
import torch
from thoughtminers.miners import ThoughtMiner
from config import Config
import torch.nn.functional as F


def get_response_expected_kl_divergence(
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
    config: Config,
    current_input_ids: torch.Tensor,
):
    """Get new reasoning step proposal.

    This function samples reasoning steps from the model and selects the best
    proposal based on the expected KL divergence of the next reasoning steps.

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
        save_outputs=False,
    )

    # clear cache to avoid OOM errors
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if proposal_token_ids is None or len(proposal_token_ids) == 0:
        return None, None

    # saving the number of tokens in each proposal
    n_tokens = [len(id) for id in proposal_token_ids]
    this_logs["n_tokens"] = n_tokens

    # trajectory-level KL divergence
    uniform = (
        torch.ones(miner.model.config.vocab_size) / miner.model.config.vocab_size
    ).to(miner.model.device)
    proposal_divergences = torch.stack(
        [
            F.kl_div(torch.log(proposal_probs[n_batch]), uniform, reduction="batchmean")
            for n_batch in range(config.n_proposal)
        ]
    )

    # estimate the entropy of the proposals
    this_logs["proposals_kl_divergences"] = (
        proposal_divergences.detach().cpu().float().tolist()
    )

    # 2 - Get an estimate of the expected KL divergence for each of these steps
    expected_kl_divergence = []
    for proposal_token, n in zip(proposal_token_ids, range(config.n_proposal)):
        # if the proposal is a final answer, stop here
        # use the previous KL divergence as the expected divergence
        if re.search(
            config.is_an_answer_pattern,
            "".join(miner.tokenizer.batch_decode(proposal_token)),
        ):
            expected_kl_divergence.append(torch.tensor(100.0).to(miner.model.device))
            continue

        # add the proposal step to the input
        concat_inputs = torch.cat(
            (current_input_ids.squeeze(), proposal_token)
        ).unsqueeze(0)

        # sample possible proposals from this step -------------------------------------
        _, _, expected_probs = miner.sample_thoughts(
            input_ids=concat_inputs,
            num_return_sequences=config.n_proposal,
            save_outputs=False,
        )

        # early stop if no steps or missing steps are sampled
        if not expected_probs:
            expected_kl_divergence.append(torch.tensor(100.0).to(miner.model.device))
            continue

        # trajectory-level KL divergence
        expected_kl_divergence.append(
            torch.stack(
                [
                    F.kl_div(
                        torch.log(expected_probs[n_batch]),
                        uniform,
                        reduction="batchmean",
                    )
                    for n_batch in range(config.n_proposal)
                ]
            ).max()
        )

    # save in log data
    this_logs["expected_kl_divergences"] = (
        torch.stack(expected_kl_divergence).detach().cpu().float().tolist()
    )

    # 3 - Select best step proposal

    # sum the proposal divergence and next expected divergence - higher = better
    weigths = proposal_divergences + torch.stack(expected_kl_divergence)
    select_id = torch.argmax(weigths)

    this_logs["final_weigths"] = weigths.detach().cpu().float().tolist()
    this_logs["select_id"] = select_id.detach().cpu().float().tolist()

    return proposal_token_ids[select_id], this_logs
