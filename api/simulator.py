import pandas as pd
import numpy as np
import asyncio
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR
from utils.logger import get_logger

log = get_logger("simulator")


class DataSimulator:
    """
    Replays NSL-KDD test set as a live event stream.
    Simulates real-time network traffic for demo purposes.
    """

    def __init__(
        self,
        speed      : float = 1.0,   # events per second
        attack_boost: bool = True,   # oversample attacks for demo
    ):
        self.speed       = speed
        self.attack_boost= attack_boost
        self.running     = False
        self._df         = None

    def load(self) -> None:
        """Load test dataset."""
        path = PROCESSED_DIR / "nslkdd_test_features.csv"
        self._df = pd.read_csv(path)
        log.info(f"Simulator loaded {len(self._df):,} events")

        if self.attack_boost:
            # Oversample attacks 3x for better demo visibility
            attacks = self._df[self._df["label_binary"] == 1]
            self._df = pd.concat(
                [self._df, attacks, attacks],
                ignore_index=True
            ).sample(frac=1, random_state=42).reset_index(drop=True)
            log.info(f"After attack boost: {len(self._df):,} events")

    def get_next_event(self, idx: int) -> dict:
        """Get event at index as dict, cycling through dataset."""
        row = self._df.iloc[idx % len(self._df)]
        event = row.to_dict()
        # Remove label from event (model shouldn't see it)
        event.pop("label_binary", None)
        event.pop("attack_type", None)
        return event

    async def stream(
        self,
        callback,
        max_events: int = None,
    ) -> None:
        """
        Async generator that yields events at self.speed per second.

        Args:
            callback  : async function called with each event dict
            max_events: stop after this many events (None = run forever)
        """
        if self._df is None:
            self.load()

        self.running = True
        idx          = 0
        count        = 0

        log.info(
            f"Simulator starting — "
            f"{self.speed} events/sec"
        )

        while self.running:
            if max_events and count >= max_events:
                break

            event = self.get_next_event(idx)
            await callback(event)

            idx   += 1
            count += 1
            await asyncio.sleep(1.0 / self.speed)

        log.info(f"Simulator stopped after {count} events")

    def stop(self) -> None:
        self.running = False


# Global simulator instance
simulator = DataSimulator(speed=2.0, attack_boost=True)