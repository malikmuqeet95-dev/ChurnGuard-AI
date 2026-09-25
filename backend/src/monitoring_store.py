
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MonitoringStore:
    """
    Lightweight JSONL storage for monitoring reports.

    One JSON object is stored per line.

    This implementation is intended for local development,
    testing, and small-scale monitoring history.
    """

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)

    def _ensure_parent_directory(self) -> None:
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Append one monitoring report to the JSONL file.
        """

        if not isinstance(report, dict):
            raise TypeError("Monitoring report must be a dictionary.")

        self._ensure_parent_directory()

        with self.storage_path.open(
            mode="a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        return report

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all valid monitoring records.

        Empty lines are ignored.
        Invalid JSON records raise ValueError to prevent
        silently hiding corrupted monitoring history.
        """

        if not self.storage_path.exists():
            return []

        records: List[Dict[str, Any]] = []

        with self.storage_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Monitoring record at line {line_number} "
                        "must be a JSON object."
                    )

                records.append(record)

        return records

    def count(self) -> int:
        """
        Return the number of stored monitoring records.
        """

        return len(self.read_all())

    def latest(self) -> Dict[str, Any] | None:
        """
        Return the latest monitoring record.

        Returns None when no records exist.
        """

        records = self.read_all()

        if not records:
            return None

        return records[-1]

    def clear(self) -> None:
        """
        Delete the monitoring history file.

        Primarily useful for tests and local maintenance.
        """

        if self.storage_path.exists():
            self.storage_path.unlink()