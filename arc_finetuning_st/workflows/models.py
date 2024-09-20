from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field


class Prediction(BaseModel):
    rationale: str = Field(
        description="Brief description of pattern and why prediction was made. Limit to 150 words."
    )
    prediction: str = Field(
        description="Predicted grid as a single string. e.g. '0,0,1\n1,1,1\n0,0,0'"
    )

    @staticmethod
    def prediction_str_to_int_array(prediction: str) -> List[List[int]]:
        return [
            [int(a) for a in el.split(",")] for el in prediction.split("\n")
        ]


class Critique(BaseModel):
    critique: str = Field(
        description="Brief critique of the previous prediction and rationale. Limit to 150 words."
    )


class Correction(BaseModel):
    correction: str = Field(
        description="Corrected prediction as a single string. e.g. '0,0,1\n1,1,1\n0,0,0'"
    )
