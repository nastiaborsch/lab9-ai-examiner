"""Offline tests: no real API calls are performed."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import exam_tools
from agent import ExamAgent, MAX_TURNS_PER_TOPIC


class FakeCompletions:
    def __init__(self, scores=None):
        self.scores = iter(scores or [])
        self.called_tools = []
        self.evaluations = 0

    def create(self, **kwargs):
        if kwargs.get("tools"):
            tool = kwargs["tools"][0]["function"]["name"]
            self.called_tools.append(tool)
            message = SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(name=tool, arguments="{}")
                    )
                ],
                content=None,
            )
        elif kwargs.get("response_format"):
            self.evaluations += 1
            score = next(self.scores, 9.0)
            payload = {
                "score": score,
                "feedback": "Пояснення частково правильне." if score < 7.5 else "Добра відповідь.",
                "can_finish": score >= 7.5,
                "follow_up": "Яка головна перевага цього підходу?",
            }
            message = SimpleNamespace(tool_calls=None, content=json.dumps(payload))
        else:
            prompt = kwargs["messages"][0]["content"]
            if "ОДНЕ чітке питання" in prompt:
                content = "Що це таке і для чого використовується?"
            else:
                content = "Варто повторити практичні приклади застосування методів NLP."
            message = SimpleNamespace(tool_calls=None, content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ExamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch_dir = patch.object(exam_tools, "EXAMS_DIR", Path(self.temp.name))
        self.patch_dir.start()

    def tearDown(self):
        self.patch_dir.stop()
        self.temp.cleanup()

    def test_start_selects_two_or_three_unique_topics_and_persists(self):
        session = {}
        with exam_tools.bind_session(session):
            topics = exam_tools.start_exam("TEST.student@example.com", "Тестовий Студент")
        self.assertIn(len(topics), (2, 3))
        self.assertEqual(len(set(topics)), len(topics))
        self.assertTrue(all(t in exam_tools.TOPICS for t in topics))
        self.assertEqual(session["email"], "test.student@example.com")
        saved = json.loads((Path(self.temp.name) / f"{session['exam_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "in_progress")
        self.assertIsNone(saved["score"])

    def test_unknown_student_must_be_rejected(self):
        session = {}
        with exam_tools.bind_session(session):
            with self.assertRaises(exam_tools.StudentNotFoundError):
                exam_tools.start_exam("nobody@example.com", "Невідомий")
        self.assertEqual(session, {})

    def test_next_topic_advances_and_stops(self):
        session = {}
        with exam_tools.bind_session(session):
            topics = exam_tools.start_exam("test.student@example.com", "Тестовий Студент")
            returned = [exam_tools.get_next_topic() for _ in topics]
            with self.assertRaises(StopIteration):
                exam_tools.get_next_topic()
        self.assertEqual(returned, topics)

    def test_end_exam_requires_all_topics_and_saves_history(self):
        session = {}
        with exam_tools.bind_session(session):
            topics = exam_tools.start_exam("test.student@example.com", "Тестовий Студент")
            history = [
                {"role": "system", "content": "Привіт", "datetime": exam_tools.timestamp()},
                {"role": "user", "content": "Тест", "datetime": exam_tools.timestamp()},
                {"role": "tool_call", "content": "start_exam", "datetime": exam_tools.timestamp()},
            ]
            with self.assertRaises(ValueError):
                exam_tools.end_exam(session["email"], 7.0, history)
            for topic in topics:
                session["topic_scores"][topic] = 7.0
            exam_tools.end_exam(session["email"], 7.0, history)
        saved = json.loads((Path(self.temp.name) / f"{session['exam_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["score"], 7.0)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["history"], history)
        self.assertIsNotNone(saved["finished_at"])

    def test_complete_agent_uses_every_tool_and_saves_history(self):
        fake = FakeCompletions(scores=[5.0, 8.0] + [9.0] * 4)
        agent = ExamAgent(client=SimpleNamespace(chat=SimpleNamespace(completions=fake)))
        session = agent.start("Тестовий Студент", "test.student@example.com")
        topics = session["topics"]
        self.assertEqual(fake.called_tools, ["start_exam", "get_next_topic"])
        self.assertIn("Тема 1/", session["chat"][-1]["content"])

        agent.answer(session, "Приблизна відповідь")
        self.assertEqual(session["topic_index"], 0)
        self.assertIn("Уточнення", session["chat"][-1]["content"])
        agent.answer(session, "Детальніша відповідь з прикладом")
        self.assertEqual(session["topic_index"], 1)
        for _ in range(len(topics) - 1):
            agent.answer(session, "Відповідь достатньої якості")
        self.assertEqual(session["status"], "completed")
        self.assertEqual(fake.called_tools.count("start_exam"), 1)
        self.assertEqual(fake.called_tools.count("get_next_topic"), len(topics))
        self.assertEqual(fake.called_tools.count("end_exam"), 1)
        self.assertEqual(len(session["topic_scores"]), len(topics))
        self.assertTrue(0 <= session["score"] <= 10)
        self.assertEqual(session["history"][-1]["role"], "tool_call")
        self.assertIn("Іспит завершено", session["chat"][-1]["content"])
        saved_text = (Path(self.temp.name) / f"{session['exam_id']}.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(saved_text)["status"], "completed")
        self.assertNotIn("gsk_", saved_text)
        for entry in session["history"]:
            self.assertEqual(set(entry), {"role", "content", "datetime"})

    def test_give_up_moves_forward_even_if_score_is_low(self):
        fake = FakeCompletions(scores=[0.0])
        agent = ExamAgent(client=SimpleNamespace(chat=SimpleNamespace(completions=fake)))
        session = agent.start("Тестовий Студент", "test.student@example.com")
        agent.answer(session, "не знаю")
        self.assertEqual(session["topic_index"], 1)
        self.assertEqual(session["topic_scores"][session["topics"][0]], 0.0)

    def test_round_limit_forces_next_topic(self):
        fake = FakeCompletions(scores=[2.0, 2.0, 2.0])
        agent = ExamAgent(client=SimpleNamespace(chat=SimpleNamespace(completions=fake)))
        session = agent.start("Тестовий Студент", "test.student@example.com")
        for i in range(MAX_TURNS_PER_TOPIC):
            agent.answer(session, f"моя відповідь номер {i}")
        self.assertEqual(session["topic_index"], 1)
        self.assertEqual(session["topic_scores"][session["topics"][0]], 2.0)


if __name__ == "__main__":
    unittest.main()
