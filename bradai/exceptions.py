from __future__ import annotations

from typing import Any


class CancelException(Exception):
    def __init__(self, *args: Any, name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
