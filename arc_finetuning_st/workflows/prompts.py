from llama_index.core.prompts import PromptTemplate

SYSTEM_PROMPTS = {
    "reasoning": """You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
Identify the pattern in the training examples, and articulate it in words.
{task_string}
Your response:
"""
}
