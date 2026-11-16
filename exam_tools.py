"""Three local agent tools specified in the NLP laboratory assignment.

The context variable binds each Gradio visitor to their own exam state, so
concurrent sessions do not share the current topic.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TypedDict, Literal

BASE_DIR = Path(__file__).resolve().parent
STUDENTS_FILE = BASE_DIR / "data" / "students.json"
EXAMS_DIR = BASE_DIR / "data" / "exams"

TOPICS = [
    "Токенізація та нормалізація тексту",
    "Векторизація тексту: Bag of Words і TF-IDF",
    "Word2Vec та семантичні векторні представлення",
    "Класифікація тексту та метрики якості",
    "Рекурентні нейронні мережі та LSTM",
    "Transformer і механізм self-attention",
    "NER: розпізнавання іменованих сутностей",
    "Мовні моделі та генерація тексту",
    "Seq2Seq і машинний переклад",
    "LLM: промпти, fine-tuning та обмеження",
]


class Message(TypedDict):
    role: Literal["system", "user", "tool_call"]
    content: str
    datetime: str


class StudentNotFoundError(ValueError):
    """The name/email pair is not in the student register."""


_SESSION: ContextVar[dict | None] = ContextVar("exam_session", default=None)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def bind_session(session: dict) -> Iterator[None]:
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)


def _session() -> dict:
    result = _SESSION.get()
    if result is None:
        raise RuntimeError("Інструмент викликано поза контекстом іспиту")
    return result


def _student_exists(email: str, name: str) -> bool:
    students = json.loads(STUDENTS_FILE.read_text(encoding="utf-8"))
    return any(
        item["email"].strip().casefold() == email.strip().casefold()
        and " ".join(item["name"].split()).casefold() == " ".join(name.split()).casefold()
        for item in students
    )


def save_session(session: dict) -> None:
    """Write an exam atomically to its own JSON file."""
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXAMS_DIR / f"{session['exam_id']}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=EXAMS_DIR, suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(session, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, target)


def start_exam(email: str, name: str) -> list[str]:
    """Validate a student, save the start record, select 2–3 random topics."""
    session = _session()
    if not _student_exists(email, name):
        raise StudentNotFoundError(
            "Студента не знайдено в списку. Перевірте ім’я та email і спробуйте ще раз."
        )
    topics = random.sample(TOPICS, k=random.choice((2, 3)))
    session.clear()
    session.update(
        exam_id=uuid.uuid4().hex,
        email=email.strip().casefold(),
        name=" ".join(name.split()),
        topics=topics,
        topic_index=-1,
        topic_scores={},
        topic_feedback={},
        topic_dialogue={},
        turn_count=0,
        status="in_progress",
        started_at=timestamp(),
        finished_at=None,
        score=None,
        history=[],
        chat=[],
    )
    save_session(session)
    return topics


def get_next_topic() -> str:
    """Advance to the next topic, or raise when there is none."""
    session = _session()
    if session.get("status") != "in_progress":
        raise RuntimeError("Іспит не розпочато або вже завершено")
    index = session["topic_index"] + 1
    if index >= len(session["topics"]):
        raise StopIteration("Усі теми вже пройдено")
    session["topic_index"] = index
    session["turn_count"] = 0
    topic = session["topics"][index]
    session["topic_dialogue"].setdefault(topic, [])
    save_session(session)
    return topic


def end_exam(email: str, score: float, history: list[Message]) -> None:
    """Save the 0–10 score together with the full timestamped dialogue."""
    session = _session()
    if session.get("status") != "in_progress":
        raise RuntimeError("Немає активного іспиту")
    if email.strip().casefold() != session["email"]:
        raise ValueError("Email не збігається з email учасника іспиту")
    if not (0 <= score <= 10):
        raise ValueError("Оцінка має бути у межах 0..10")
    if len(session["topic_scores"]) != len(session["topics"]):
        raise ValueError("Не всі теми пройдено")
    if not history or any(
        item.get("role") not in {"system", "user", "tool_call"}
        or not isinstance(item.get("content"), str)
        or not isinstance(item.get("datetime"), str)
        for item in history
    ):
        raise ValueError("Історія має містити role, content, datetime")
    session["score"] = round(float(score), 2)
    session["history"] = history
    session["status"] = "completed"
    session["finished_at"] = timestamp()
    save_session(session)
