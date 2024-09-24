import subprocess

commands = [
    "wget https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip -O ./master.zip",
    "unzip ./master.zip -d ./",
    "rm -rf ./data",
    "mv ARC-AGI-master/data ./",
    "rm -rf ARC-AGI-master",
    "rm master.zip",
]


def download_data() -> None:
    subprocess.run("; ".join(commands), capture_output=True, shell=True)


if __name__ == "__main__":
    download_data()
