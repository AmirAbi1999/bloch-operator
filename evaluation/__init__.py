"""Scoring a trained operator: checkpoint loading and split evaluation."""

from .evaluator import evaluate, load_checkpoint

__all__ = [
    "evaluate",
    "load_checkpoint",
]
