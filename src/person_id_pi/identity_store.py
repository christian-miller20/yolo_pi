from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import numpy as np


@dataclass
class UserTemplates:
    user_id: str
    templates: List[List[float]] = field(default_factory=list)
    display_name: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "user_id": self.user_id,
            "templates": self.templates,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "UserTemplates":
        return cls(
            user_id=str(payload["user_id"]),
            templates=[list(vec) for vec in payload.get("templates", [])],
            display_name=(
                str(payload["display_name"]).strip()
                if payload.get("display_name")
                else None
            ),
        )


class IdentityStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, UserTemplates] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or not self.path.read_text().strip():
            self._users = {}
            return
        self._users = self._read_users()

    def _read_users(self) -> Dict[str, UserTemplates]:
        if not self.path.exists() or not self.path.read_text().strip():
            return {}
        data = json.loads(self.path.read_text())
        users: Dict[str, UserTemplates] = {}
        for entry in data:
            try:
                record = UserTemplates.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                continue
            users[record.user_id] = record
        return users

    def save(self, *, merge_external_labels: bool = True) -> None:
        if merge_external_labels:
            for user_id, disk_record in self._read_users().items():
                if user_id in self._users and disk_record.display_name:
                    self._users[user_id].display_name = disk_record.display_name
        payload = [user.to_dict() for user in self._users.values()]
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, indent=2))
            temp_path = Path(handle.name)
        temp_path.replace(self.path)

    def list_users(self) -> List[str]:
        return sorted(self._users.keys())

    def has_user(self, user_id: str) -> bool:
        return user_id in self._users

    def get_display_name(self, user_id: str, *, refresh: bool = True) -> Optional[str]:
        record = self._users.get(user_id)
        if record is None:
            return None
        if refresh:
            disk_record = self._read_users().get(user_id)
            if disk_record is not None:
                record.display_name = disk_record.display_name
        return record.display_name

    def display_label(self, user_id: str) -> str:
        return self.get_display_name(user_id) or user_id

    def set_display_name(self, user_id: str, display_name: str) -> bool:
        name = display_name.strip()
        if not name or user_id not in self._users:
            return False
        for other_id, record in self._users.items():
            if other_id == user_id or record.display_name is None:
                continue
            if record.display_name.casefold() == name.casefold():
                return False
        self._users[user_id].display_name = name
        self.save(merge_external_labels=False)
        return True

    def template_count(self, user_id: str) -> int:
        record = self._users.get(user_id)
        return len(record.templates) if record is not None else 0

    def get_templates(self, user_id: str) -> List[np.ndarray]:
        user = self._users.get(user_id)
        if user is None:
            return []
        return [np.asarray(vec, dtype=np.float32) for vec in user.templates]

    def add_user(self, user_id: str) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
            self.save()

    def delete_user(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False
        del self._users[user_id]
        self.save()
        return True

    def add_template(self, user_id: str, embedding: np.ndarray) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
        user = self._users[user_id]
        user.templates.append(embedding.astype(np.float32).tolist())
        self.save()

    def replace_templates(self, user_id: str, templates: List[np.ndarray]) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
        self._users[user_id].templates = [
            t.astype(np.float32).tolist() for t in templates
        ]
        self.save()

    def rename_user(self, current_user_id: str, new_user_id: str) -> bool:
        if current_user_id not in self._users:
            return False
        if current_user_id == new_user_id:
            return True
        if new_user_id in self._users:
            return False
        record = self._users.pop(current_user_id)
        record.user_id = new_user_id
        self._users[new_user_id] = record
        self.save()
        return True

    def generate_new_user_id(self) -> str:
        pattern = re.compile(r"^user_(\d+)$")
        max_idx = -1
        for user_id in self._users.keys():
            match = pattern.match(user_id)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return f"user_{max_idx + 1:04d}"
