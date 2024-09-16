from typing import List
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field

PREDICTION_PROMPT_TEMPLATE = PromptTemplate(
    """You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
Identify the pattern in the training examples and predict the output for the provided TEST INPUT.

EXAMPLES:
{examples}

TEST INPUT:
{test_input}

OUTPUT FORMAT:
{{
    "output": 
}}

Return your response in JSON format given above. DO NOT RETURN markdown code.
"""
)


REFLECTION_PROMPT_TEMPLATE = PromptTemplate(
    """You are a bot that is very good at solving puzzles. Below is a list of input and output pairs that share a
common pattern. The TEST INPUT also shares this common pattern, and you've previously predicted the output for it.
Your task now is critique the prediction on why it might not fit the pattern inherent in the example input/output pairs.

EXAMPLES:
{examples}

TEST INPUT:
{test_input}

PREDICTED OUTPUT:
{predicted_output}

OUTPUT FORMAT:
{{
    "critique": ...
}}

Return your response in JSON format given above. DO NOT RETURN markdown code."""
)

CORRECTION_PROMPT_TEMPLATE = PromptTemplate(
    """You are a bot that is very good at solving puzzles. Below is a list of input and output pairs that share a
common pattern. The TEST INPUT also shares this common pattern, and you've previously predicted the output for it.
The predicted output was found to be incorrect and a critique has been articulated offering a potential
reason as to why it may have been a flawed prediction.

Your task now to create a new prediction that corrects the previous one using the critique.

EXAMPLES:
{examples}

TEST INPUT:
{test_input}

PREDICTED OUTPUT:
{predicted_output}

CRITIQUE:
{critique}

OUTPUT FORMAT:
{{
    "correction": ...
}}

Return your response in JSON format given above. DO NOT RETURN markdown code."""
)


class Prediction(BaseModel):
    rationale: str = Field(
        description="Brief description of pattern and why prediction was made. Limit to 150 words."
    )
    prediction: str = Field(
        description="Predicted grid as a single string. e.g. '0,0,1\n1,1,1\n0,0,0'"
    )

    @staticmethod
    def prediction_str_to_int_array(prediction: str) -> List[List[int]]:
        return [[int(a) for a in el.split(",")] for el in prediction.split("\n")]


class Critique(BaseModel):
    critique: str = Field(
        description="Brief critique of the previous prediction and rationale. Limit to 150 words."
    )


class Correction(BaseModel):
    correction: str = Field(
        description="Corrected prediction as a single string. e.g. '0,0,1\n1,1,1\n0,0,0'"
    )
