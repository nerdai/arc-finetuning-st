# prepare jsonl from finetuning_examples/

# get finetuning model name if exists

# submit finetune job

# check on current job id

from os import listdir
from pathlib import Path

SINGLE_EXAMPLE_JSON_PATH = Path(
    Path(__file__).parents[1].absolute(), "finetuning_examples"
)

FINETUNING_ASSETS_PATH = Path(
    Path(__file__).parents[1].absolute(), "finetuning_assets"
)


def prepare_finetuning_jsonl_file(json_path: Path = SINGLE_EXAMPLE_JSON_PATH):
    """Read all json files from data path and write a jsonl file."""
    with open(FINETUNING_ASSETS_PATH / "finetuning.jsonl", "w") as jsonl_out:
        for json_name in listdir(json_path):
            with open(SINGLE_EXAMPLE_JSON_PATH / json_name) as f:
                for line in f:
                    jsonl_out.write(line)
                    jsonl_out.write("\n")


if __name__ == "__main__":
    FINETUNING_ASSETS_PATH.mkdir(exist_ok=True, parents=True)
    prepare_finetuning_jsonl_file()
