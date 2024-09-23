import argparse
import asyncio
from os import listdir
from pathlib import Path
from typing import Any, List, cast

from llama_index.llms.openai import OpenAI

from arc_finetuning_st.cli.evaluation import batch_runner
from arc_finetuning_st.workflows.arc_task_solver import (
    ARCTaskSolverWorkflow,
    WorkflowOutput,
)


def handle_evaluate(
    llm: str,
    batch_size: int,
    num_workers: int,
    verbose: bool,
    sleep: int,
    **kwargs: Any,
) -> None:
    data_path = Path(
        Path(__file__).parents[2].absolute(), "data", "evaluation"
    )
    task_paths = [data_path / t for t in listdir(data_path)]
    llm = OpenAI(llm)
    w = ARCTaskSolverWorkflow(llm=llm, timeout=None)
    results = asyncio.run(
        batch_runner(
            w,
            task_paths[:10],
            verbose=verbose,
            batch_size=batch_size,
            num_workers=num_workers,
            sleep=sleep,
        )
    )
    results = cast(List[WorkflowOutput], results)
    num_solved = sum(el.passing for el in results)
    print(
        f"Solved: {num_solved}\nTotal Tasks:{len(results)}\nAverage Solve Rate: {float(num_solved) / len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="arc-finetuning cli tool.")

    # Subparsers
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True
    )

    # evaluation command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluation of ARC Task predictions with LLM and ARCTaskSolverWorkflow.",
    )
    evaluate_parser.add_argument(
        "-m",
        "--llm",
        type=str,
        default="gpt-4o",
        help="The OpenAI LLM model to use with the Workflow.",
    )
    evaluate_parser.add_argument("-b", "--batch-size", type=int, default=5)
    evaluate_parser.add_argument("-w", "--num-workers", type=int, default=3)
    evaluate_parser.add_argument(
        "-v", "--verbose", action=argparse.BooleanOptionalAction
    )
    evaluate_parser.add_argument("-s", "--sleep", type=int, default=10)
    evaluate_parser.set_defaults(
        func=lambda args: handle_evaluate(**vars(args))
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()
