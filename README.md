# ARC Task LLM Solver With Human Input

The Abstraction and Reasoning Corpus (ARC) for Artificial General Intelligence
benchmark aims to measure an AI system's ability to efficiently learn new skills.
Each task within the ARC benchmark contains a unique puzzle for which the systems
attempt to solve. Currently, the best AI systems achieve 34% solve rates, whereas
humans are able to achieve 85% ([source](https://www.kaggle.com/competitions/arc-prize-2024/overview/prizes)).

Motivated by this large disparity, we built this app with the goal of injecting
human-level reasoning on this benchmark to LLMs. Specifically, this app enables
the collaboration of LLMs and humans to solve an ARC task; and these collaborations
can then be used for fine-tuning the LLM.

[example ARC task]

## Running The App

Before running the streamlit app, we first must download the ARC dataset. The
below command will download the dataset and store it in a directory named `data/`:

```sh
wget https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip -O ./master.zip
unzip ./master.zip -d ./
mv ARC-AGI-master/data ./
rm -rf ARC-AGI-master
rm master.zip
```

Next, we must install the app's dependencies. To do so, we can use `poetry`:

```sh
poetry shell
poetry install
```

Finally, to run the streamlit app:

```sh
export OPENAI_API_KEY=<FILL-IN> && streamlit run arc_finetuning_st/streamlit/app.py
```

## How To Use The App

### Solving An ARC Task

### Saving Solutions For Fine-Tuning

## A Note On The Underlying LlamaIndex Workflow
