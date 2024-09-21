import json
from typing import Annotated, Any, Callable, List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel, WrapSerializer

from arc_finetuning_st.finetuning.templates import (
    ASSISTANT_TEMPLATE,
    SYSTEM_MESSAGE,
    USER_CRITIQUE_TEMPLATE,
    USER_TASK_TEMPLATE,
)
from arc_finetuning_st.workflows.models import Attempt


def remove_additional_kwargs(value: Any, handler: Callable, info: Any) -> Any:
    partial_result = handler(value, info)
    del partial_result["additional_kwargs"]
    return partial_result


class FineTuningExample(BaseModel):
    messages: List[
        Annotated[ChatMessage, WrapSerializer(remove_additional_kwargs)]
    ]

    @classmethod
    def from_attempts(
        cls,
        examples: str,
        test_input: str,
        attempts: List[Attempt],
        system_message: str = SYSTEM_MESSAGE,
        user_task_template: str = USER_TASK_TEMPLATE,
        user_critique_template: str = USER_CRITIQUE_TEMPLATE,
        assistant_template: str = ASSISTANT_TEMPLATE,
    ) -> "FineTuningExample":
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_message),
            ChatMessage(
                role=MessageRole.USER,
                content=user_task_template.format(
                    examples=examples, test_input=test_input
                ),
            ),
        ]
        for a in attempts:
            messages.extend(
                [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=assistant_template.format(
                            predicted_output=str(a.prediction),
                            rationale=a.prediction.rationale,
                        ),
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=user_critique_template.format(
                            critique=str(a.critique)
                        ),
                    ),
                ]
            )
        return cls(messages=messages)

    def to_json(self) -> str:
        data = self.model_dump()
        return json.dumps(data, indent=4)
