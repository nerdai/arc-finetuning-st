# prepare jsonl from finetuning_examples/

# get finetuning model name if exists

# submit finetune job

# check on current job id

from os import listdir
from pathlib import Path

from llama_index.finetuning import OpenAIFinetuneEngine

SINGLE_EXAMPLE_JSON_PATH = Path(
    Path(__file__).parents[1].absolute(), "finetuning_examples"
)

FINETUNING_ASSETS_PATH = Path(
    Path(__file__).parents[1].absolute(), "finetuning_assets"
)

FINETUNE_JSONL_FILENAME = "finetuning.jsonl"
FINETUNE_JOBS_FILENAME = "finetuning_jobs.txt"


def prepare_finetuning_jsonl_file(json_path: Path = SINGLE_EXAMPLE_JSON_PATH):
    """Read all json files from data path and write a jsonl file."""
    with open(
        FINETUNING_ASSETS_PATH / FINETUNE_JSONL_FILENAME, "w"
    ) as jsonl_out:
        for json_name in listdir(json_path):
            with open(SINGLE_EXAMPLE_JSON_PATH / json_name) as f:
                for line in f:
                    jsonl_out.write(line)
                    jsonl_out.write("\n")


def submit_finetune_job():
    """Submit finetuning job."""

    try:
        with open(FINETUNING_ASSETS_PATH / FINETUNE_JOBS_FILENAME) as f:
            lines = f.read().splitlines()
            current_job_id = lines[-1]
    except FileNotFoundError:
        # no previous finetune model
        current_job_id = None

    finetune_engine = OpenAIFinetuneEngine(
        "gpt-4o-2024-08-06",
        (FINETUNING_ASSETS_PATH / FINETUNE_JSONL_FILENAME).as_posix(),
        start_job_id=current_job_id,
        validate_json=False,
    )
    finetune_engine.finetune()

    with open(FINETUNING_ASSETS_PATH / FINETUNE_JOBS_FILENAME, "a+") as f:
        f.write(finetune_engine._start_job.id)
        f.write("\n")

    print(finetune_engine.get_current_job())


def check_latest_job_status():
    """Check on status of most recent submitted finetuning job."""
    try:
        with open(FINETUNING_ASSETS_PATH / FINETUNE_JOBS_FILENAME) as f:
            lines = f.read().splitlines()
            current_job_id = lines[-1]
    except FileNotFoundError:
        raise ValueError(
            "No finetuning_jobs.txt file exists. You likely haven't submitted a job yet."
        )

    finetune_engine = OpenAIFinetuneEngine(
        "gpt-4o-2024-08-06",
        (FINETUNING_ASSETS_PATH / FINETUNE_JSONL_FILENAME).as_posix(),
        start_job_id=current_job_id,
        validate_json=False,
    )

    print(finetune_engine.get_current_job())


if __name__ == "__main__":
    FINETUNING_ASSETS_PATH.mkdir(exist_ok=True, parents=True)
    prepare_finetuning_jsonl_file()
    # submit_finetune_job()
    check_latest_job_status()
