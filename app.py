"""Launch: python app.py"""

from __future__ import annotations

import os

import gradio as gr

from agent import ExamAgent
from exam_tools import StudentNotFoundError

WELCOME = [
    {
        "role": "assistant",
        "content": (
            "Привіт! Я AI Examiner Agent. Введіть ім’я, email зі списку "
            "та Groq API key, а тоді натисніть **«Почати іспит»**."
        ),
    }
]


def start_exam_ui(name: str, email: str, api_key: str):
    try:
        agent = ExamAgent(api_key)
        session = agent.start(name, email)
        return session["chat"], session, f"Іспит розпочато. Тем: {len(session['topics'])}."
    except StudentNotFoundError as exc:
        return WELCOME, None, f"❌ {exc}"
    except Exception as exc:
        return WELCOME, None, f"❌ Не вдалося почати іспит: {exc}"


def submit_answer_ui(message: str, api_key: str, session: dict | None):
    if not session:
        return WELCOME, None, "Спочатку натисніть «Почати іспит».", message
    if session.get("status") == "completed":
        return session["chat"], session, "Іспит уже завершено. Почніть новий.", message
    try:
        agent = ExamAgent(api_key)
        session = agent.answer(session, message)
        if session["status"] == "completed":
            status = f"Іспит завершено. Результат: {session['score']}/10."
        else:
            status = f"Тема {session['topic_index'] + 1} з {len(session['topics'])}."
        return session["chat"], session, status, ""
    except Exception as exc:
        return session["chat"], session, f"❌ Помилка: {exc}. Можна повторити запит.", message


def reset_exam_ui():
    return WELCOME, None, "Новий іспит: введіть дані та натисніть «Почати іспит».", ""


with gr.Blocks(title="AI Examiner Agent | NLP") as demo:
    gr.Markdown(
        "# AI Examiner Agent — іспит з NLP\n"
        "LLM-агент із трьома Python-інструментами: `start_exam`, "
        "`get_next_topic`, `end_exam`. Він обирає 2–3 теми, ставить запитання, "
        "оцінює відповіді та зберігає результати у JSON.\n\n"
        "**Демо-студент:** Тестовий Студент — `test.student@example.com`. "
        "У власному списку студентів можна змінити дані в `data/students.json`."
    )
    with gr.Row():
        name = gr.Textbox(label="Ім’я та прізвище", placeholder="Тестовий Студент")
        email = gr.Textbox(label="Email", placeholder="test.student@example.com")
    api_key = gr.Textbox(
        label="Groq API key", type="password", placeholder="gsk_...",
        info="Ключ не записується у файли з результатами іспитів.",
    )
    with gr.Row():
        start_button = gr.Button("Почати іспит", variant="primary")
        reset_button = gr.Button("Новий іспит")
    chat = gr.Chatbot(label="Діалог іспиту", value=WELCOME, height=430)
    answer = gr.Textbox(label="Ваша відповідь", placeholder="Введіть відповідь українською...")
    send_button = gr.Button("Надіслати відповідь", variant="primary")
    status = gr.Markdown("Заповніть дані для початку іспиту.")
    session_state = gr.State(value=None)

    start_button.click(
        start_exam_ui, inputs=[name, email, api_key],
        outputs=[chat, session_state, status],
    )
    send_button.click(
        submit_answer_ui, inputs=[answer, api_key, session_state],
        outputs=[chat, session_state, status, answer],
    )
    answer.submit(
        submit_answer_ui, inputs=[answer, api_key, session_state],
        outputs=[chat, session_state, status, answer],
    )
    reset_button.click(
        reset_exam_ui, outputs=[chat, session_state, status, answer],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
