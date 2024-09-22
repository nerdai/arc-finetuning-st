# ARC Task (LLM) Solver With Human Input

The Abstraction and Reasoning Corpus (ARC) for Artificial General Intelligence
benchmark aims to measure an AI system's ability to efficiently learn new skills.
Each task within the ARC benchmark contains a unique puzzle for which the systems
attempt to solve. Currently, the best AI systems achieve 34% solve rates, whereas
humans are able to achieve 85% ([source](https://www.kaggle.com/competitions/arc-prize-2024/overview/prizes)).

<p align="center">
  <img height="300" src="https://d3ddy8balm3goa.cloudfront.net/arc-task-solver-st-demo/arc-task.svg" alt="cover">
</p>

Motivated by this large disparity, we built this app with the goal of injecting
human-level reasoning on this benchmark to LLMs. Specifically, this app enables
the collaboration of LLMs and humans to solve an ARC task; and these collaborations
can then be used for fine-tuning the LLM.

The Solver itself is a LlamaIndex `Workflow` that relies on successive runs for
which `Context` is maintained from previous runs. Doing so allows for an
effective implementation of the Human In the Loop Pattern.

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

In the next two sections, we discuss how to use the app in order to solve a given
ARC task.

<p align="center">
  <img height="500" src="https://d3ddy8balm3goa.cloudfront.net/arc-task-solver-st-demo/arc-task-solver-app.svg" alt="cover">
</p>

### Solving An ARC Task

Each ARC task consists of training examples, each of which consist of input and
output pairs. There exists a common pattern between these input and output pairs,
and the problem is solved by uncovering this pattern, which can be verified by
the included test examples.

To solve the task, we cycle through the following three steps:

1. Prediction (of test output grid)
2. Evaluation
3. Critique (human in the loop)

(Under the hood a LlamaIndex `Workflow` implements these three `steps`.)

Step 1. makes use of an LLM to produce the Prediction whereas Step 2. is
deterministic and is a mere comparison between the ground truth test output and
the Prediction. If the Prediction doesn't match the ground truth grid, then Step 3.
is performed. Similar to step 1. an LLM is prompted to generate a Critique on the
Prediction as to why it may not match the pattern underlying the train input and
output pairs. However, we also allow for a human in the loop to override this
LLM generated Critique.

The Critique is carried on from a previous cycle onto the next in order to
generate an improved and hopefully correct next Prediction.

To begin, click the `Start` button found in the top-right corner. If the
prediction is incorrect, you can view the Critique produced by the LLM in the
designated text area. You can choose to use this Critique or supply your own by
overwriting the text and applying the change. Once ready to produce the next
prediction, hit the `Continue` button.

### Saving Solutions For Fine-Tuning

Any collaboration session involving the LLM and human can be saved and used to
finetune an LLM. In this app, we use OpenAI LLMs, and so the finetuning examples
adhere to the [OpenAI fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset).
Click the `fine-tuning example` button during a session to see the current
example that can be used for fine-tuning.

<p align="center">
  <img height="500" src="https://d3ddy8balm3goa.cloudfront.net/arc-task-solver-st-demo/finetuning-arc-example.svg" alt="cover">
</p>
