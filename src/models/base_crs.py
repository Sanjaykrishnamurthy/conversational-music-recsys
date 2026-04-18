"""Abstract base class for Conversational Recommender Systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single dialogue turn."""
    turn_number: int
    user_utterance: str
    system_response: Optional[str] = None
    recommended_tracks: list[str] = field(default_factory=list)  # track_ids


@dataclass
class Session:
    """A full multi-turn conversation session."""
    session_id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def dialogue_history(self) -> list[dict]:
        """Return conversation history as list of role/content dicts."""
        history = []
        for turn in self.turns:
            history.append({"role": "user", "content": turn.user_utterance})
            if turn.system_response:
                history.append({"role": "assistant", "content": turn.system_response})
        return history

    @property
    def context_text(self) -> str:
        """Flatten dialogue history into a single context string."""
        parts = []
        for turn in self.turns:
            parts.append(f"User: {turn.user_utterance}")
            if turn.system_response:
                parts.append(f"System: {turn.system_response}")
        return "\n".join(parts)


class BaseCRS(ABC):
    """Abstract base for all CRS models in this challenge."""

    @abstractmethod
    def recommend(
        self,
        session: Session,
        top_k: int = 20,
    ) -> list[str]:
        """Return top-k recommended track_ids for current session state.

        Args:
            session: Current conversation session with history
            top_k: Number of tracks to recommend

        Returns:
            Ordered list of track_ids (best first)
        """
        ...

    @abstractmethod
    def generate_response(
        self,
        session: Session,
        recommended_tracks: list[str],
    ) -> str:
        """Generate a natural language response.

        Args:
            session: Current conversation session
            recommended_tracks: Track IDs already recommended this turn

        Returns:
            Generated response string
        """
        ...

    def step(
        self,
        session: Session,
        top_k: int = 20,
    ) -> tuple[list[str], str]:
        """Run one full recommendation + response generation step.

        Returns:
            (recommended_track_ids, response_text)
        """
        track_ids = self.recommend(session, top_k=top_k)
        response = self.generate_response(session, track_ids)
        return track_ids, response
