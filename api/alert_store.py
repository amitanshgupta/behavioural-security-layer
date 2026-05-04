from collections import deque
from datetime import datetime
import threading

class AlertStore:
    """
    Thread-safe in-memory alert queue.
    Stores last 100 alerts — no database needed.
    """
    def __init__(self, maxlen: int = 100):
        self._store    = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._listeners= []

    def add(self, alert: dict) -> None:
        with self._lock:
            self._store.append(alert)
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception:
                pass

    def get_all(self) -> list:
        with self._lock:
            return list(self._store)

    def get_recent(self, n: int = 20) -> list:
        with self._lock:
            items = list(self._store)
            return items[-n:]

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict:
        alerts = self.get_all()
        if not alerts:
            return {
                "total": 0, "critical": 0,
                "high": 0, "medium": 0, "low": 0,
            }
        return {
            "total"   : len(alerts),
            "critical": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
            "high"    : sum(1 for a in alerts if a["severity"] == "HIGH"),
            "medium"  : sum(1 for a in alerts if a["severity"] == "MEDIUM"),
            "low"     : sum(1 for a in alerts if a["severity"] == "LOW"),
        }

# Global singleton
alert_store = AlertStore()