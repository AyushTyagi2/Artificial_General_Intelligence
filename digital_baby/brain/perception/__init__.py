"""Perception package — screen-based and Wikipedia-based knowledge acquisition."""
from .screen_tracker import ScreenTracker, PerceptionEvent, PerceptionTriple
from .wiki_reader import WikiReader, WikiTriple, WikiArticleEvent

__all__ = [
    "ScreenTracker", "PerceptionEvent", "PerceptionTriple",
    "WikiReader", "WikiTriple", "WikiArticleEvent",
]