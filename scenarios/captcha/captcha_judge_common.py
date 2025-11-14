from pydantic import BaseModel
from typing import Optional

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


class CaptchaAttempt(BaseModel):
    """Result of a single CAPTCHA solving attempt."""
    filename: str
    actual_text: str
    predicted_text: str
    correct: bool
    character_accuracy: float  # Percentage of correct characters


class CaptchaEval(BaseModel):
    """Complete evaluation result for CAPTCHA benchmark."""
    total_attempts: int
    correct_predictions: int
    overall_accuracy: float  # Percentage of fully correct predictions
    average_character_accuracy: float  # Average per-character accuracy
    attempts: list[CaptchaAttempt]


def captcha_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Generate agent card for CAPTCHA judge."""
    skill = AgentSkill(
        id='evaluate_captcha_solving',
        name='Evaluates CAPTCHA solving capability',
        description='Evaluate an agent\'s ability to solve CAPTCHA puzzles by testing on a dataset of CAPTCHA images.',
        tags=['captcha', 'ocr', 'vision', 'benchmark'],
        examples=["""
{
  "participants": {
    "captcha_solver": "https://captcha-solver.example.com:443"
  },
  "config": {
    "num_samples": 10,
    "dataset_path": "assets/kaggle-captcha-v2-images"
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Evaluate an agent\'s ability to solve CAPTCHA puzzles from image files. The ground truth is encoded in the filename.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text', 'image'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
