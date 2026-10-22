"""Streamlit chat interface for the self-evaluating lesson agent."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from Content_Generator_Agent import (
    MAX_ALLOWED_RETRIES,
    _create_agent,
)


def _clear_session() -> None:
    for key in (
        "lesson_agent",
        "lesson_result",
        "topic",
        "messages",
        "clarification_pending",
        "clarification_question",
    ):
        st.session_state.pop(key, None)


def _chat_history(messages: list[dict[str, str]]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    return history


def _complete_lesson(lesson_agent: Any, topic: str) -> None:
    try:
        with st.spinner(f"Creating and evaluating a lesson about {topic}..."):
            result = lesson_agent.run(topic)
    except Exception as exc:
        st.error(f"Could not start the lesson session: {exc}")
        return

    st.session_state.lesson_agent = lesson_agent
    st.session_state.lesson_result = result
    st.session_state.topic = topic
    st.session_state.messages = [
        {"role": "user", "content": f"Teach me about {topic}."},
        {"role": "assistant", "content": result["lesson"]},
    ]
    st.session_state.pop("clarification_pending", None)
    st.session_state.pop("clarification_question", None)


def _start_lesson(topic: str, max_retries: int) -> None:
    topic = topic.strip()
    if not topic:
        st.error("Please enter a topic to begin.")
        return

    try:
        lesson_agent = _create_agent(max_retries)
        with st.spinner("Checking whether the topic needs clarification..."):
            clarification_question = lesson_agent.clarify_topic(topic)
    except Exception as exc:
        st.error(f"Could not start the lesson session: {exc}")
        return

    if clarification_question:
        st.session_state.lesson_agent = lesson_agent
        st.session_state.topic = topic
        st.session_state.clarification_question = clarification_question
        st.session_state.clarification_pending = True
        return
    _complete_lesson(lesson_agent, topic)


def main() -> None:
    st.set_page_config(page_title="Beginner Lesson Tutor", page_icon="📘")
    st.title("📘 Beginner Lesson Tutor")
    st.caption(
        "Enter any topic. The agent creates and checks a beginner lesson, "
        "asks for clarification if needed, then answers your follow-up questions."
    )

    if st.session_state.get("clarification_pending", False):
        st.sidebar.subheader("Session")
        st.sidebar.write(f"**Topic:** {st.session_state.topic}")
        if st.sidebar.button("Start a new topic", use_container_width=True):
            _clear_session()
            st.rerun()
        if st.sidebar.button("Exit session", use_container_width=True):
            _clear_session()
            st.session_state.session_exited = True
            st.rerun()
        st.subheader("Let's clarify your topic")
        with st.chat_message("assistant"):
            st.markdown(st.session_state.clarification_question)
        clarification = st.chat_input("Type the meaning you want to learn about")
        if clarification:
            resolved_topic = (
                f"{st.session_state.topic} (user clarified: {clarification.strip()})"
            )
            _complete_lesson(st.session_state.lesson_agent, resolved_topic)
            if st.session_state.get("lesson_result") is not None:
                st.rerun()
        return

    if st.session_state.get("lesson_agent") is None:
        if st.session_state.get("session_exited", False):
            st.info("Session ended. Start a new topic whenever you are ready.")
        with st.form("topic_form"):
            topic = st.text_input(
                "What would you like to learn about?",
                placeholder="For example: photosynthesis, budgeting, or RAG",
            )
            max_retries = st.selectbox(
                "Maximum retries after the first draft",
                options=range(MAX_ALLOWED_RETRIES + 1),
                index=MAX_ALLOWED_RETRIES,
            )
            submitted = st.form_submit_button("Start lesson")

        if submitted:
            _start_lesson(topic, max_retries)
            if (
                st.session_state.get("lesson_agent") is not None
                and not st.session_state.get("clarification_pending", False)
            ):
                st.session_state.session_exited = False
                st.rerun()
            if st.session_state.get("clarification_pending", False):
                st.session_state.session_exited = False
                st.rerun()
        return

    st.sidebar.subheader("Session")
    st.sidebar.write(f"**Topic:** {st.session_state.topic}")
    if st.sidebar.button("Start a new topic", use_container_width=True):
        _clear_session()
        st.rerun()
    if st.sidebar.button("Exit session", use_container_width=True):
        _clear_session()
        st.session_state.session_exited = True
        st.rerun()

    result: dict[str, Any] = st.session_state.lesson_result
    if result["passed"]:
        st.success(
            f"Lesson passed all checks after {result['attempts_used']} "
            f"attempt(s). Grounding: {result['grounding']}."
        )
    else:
        st.warning(
            f"The lesson did not pass after {result['attempts_used']} attempt(s). "
            "You can still ask questions; the rejection details are below."
        )
    if result["rejection_log"]:
        with st.expander("Review the rejection log"):
            st.json(result["rejection_log"])
    follow_up_evaluations = result.setdefault("follow_up_evaluations", [])
    if follow_up_evaluations:
        with st.expander("Review follow-up evaluations"):
            st.json(follow_up_evaluations)

    messages: list[dict[str, str]] = st.session_state.messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a follow-up question, or use Exit session")
    if question:
        prior_history = _chat_history(messages)
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Thinking..."):
                follow_up_result = st.session_state.lesson_agent.answer_follow_up(
                    topic=st.session_state.topic,
                    lesson=result["lesson"],
                    question=question,
                    chat_history=prior_history,
                )
        except Exception as exc:
            st.error(f"Could not answer the follow-up: {exc}")
            return

        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": follow_up_result["answer"]},
            ]
        )
        st.session_state.messages = messages
        evaluation_record = {
            "question": question,
            **follow_up_result,
        }
        follow_up_evaluations.append(evaluation_record)
        result["follow_up_evaluations"] = follow_up_evaluations
        st.session_state.lesson_result = result
        with st.chat_message("assistant"):
            st.markdown(follow_up_result["answer"])
            if follow_up_result["passed"]:
                st.success("This follow-up answer passed its evaluation.")
            else:
                st.warning("This follow-up answer did not pass every evaluation check.")
            with st.expander("Follow-up evaluation details"):
                st.write(follow_up_result["summary"])
                st.json(follow_up_result["checks"])

    st.download_button(
        "Download lesson and evaluation",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="lesson_result.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
