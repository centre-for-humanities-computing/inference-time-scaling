# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from dataclasses import dataclass
from argparse import ArgumentParser
import torch


@dataclass
class Config:
    """Configuration class for the script."""

    method: str
    model_id: str
    device: str
    max_reasoning_steps: int
    output_path: str
    n_questions: int
    dataset: str
    n_proposal: int
    is_an_answer_pattern: str = ""
    promt_style: str = ""
    overwrite: bool = False

    def __post_init__(self):
        """Set dependent fields after initialization."""
        self.is_an_answer_pattern = r"\\boxed\{.*?\}"
        if self.dataset == "gsm8k-da":
            self.promt_style = "Løs følgende matematiske problem effektivt og tydeligt:\n\n- For nemme problemer (2 trin eller færre):\nGiv en præcis løsning med minimal forklaring.\n\n- For komplekse problemer (3 trin eller mere):\nBrug dette trinvise format:\n\n## Trin 1: [Koncis beskrivelse]\n[Kort forklaring og beregninger]\n\n## Trin 2: [Koncis beskrivelse]\n[Kort forklaring og beregninger]\n\n...\n\nUanset fremgangsmåde, afslut altid med:\n\nDerfor er det endelige svar: $\\boxed{answer}$. Jeg håber, det er korrekt.\n\nHvor [answer] er blot det endelige tal eller udtryk, der løser problemet."
        else:
            self.promt_style = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."


def parse_args() -> Config:
    """Parse command line arguments and return a Config object."""
    parser = ArgumentParser()

    parser.add_argument("--method", type=str, default="forward-pass")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--max_reasoning_steps", type=int, default=30)
    parser.add_argument("--n_questions", type=int, default=-1)  # Fixed type
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--n_proposal", type=int, default=8)
    parser.add_argument("--overwrite", type=bool, default=False)

    args = parser.parse_args()

    return Config(
        method=args.method,
        model_id=args.model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",  # Dynamic device
        max_reasoning_steps=args.max_reasoning_steps,
        output_path=args.output_path,
        n_questions=args.n_questions,
        dataset=args.dataset,
        n_proposal=args.n_proposal,
        overwrite=args.overwrite,
    )
