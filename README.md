# Self-Evaluating Beginner Lesson Tutor

A Python application using one LangChain tools agent to create and evaluate
beginner lessons. The learner enters a topic, the agent checks whether the
topic is ambiguous, then drafts a lesson and checks it against a pass/fail
rubric. It regenerates failed lessons up to a selected retry limit. In the
Streamlit UI, learners can ask follow-up questions; each answer is also
self-evaluated and included in the downloadable result.

The example topic is **Introduction to RAG**, but the lesson topic and rubric
are not hard-coded to RAG.

## Requirements

- Python 3.10 or later
- An OpenAI API key
- A LangSmith API key is optional (required only for LangSmith traces and
  feedback)

## Setup

Open Command Prompt in this project folder:

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Optionally set `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`, and `LANGSMITH_TRACING`. Never commit `.env` or API keys.
The default model is `gpt-4o-mini`; set `OPENAI_MODEL` in `.env` to use another
model supported by the configured provider.

## Run the Streamlit tutor

```cmd
streamlit run app.py
```

Enter a topic and select the maximum number of lesson retries (0–2). The agent
checks for ambiguous topics first. For example, if the learner enters `RAG`,
the tutor may ask whether they mean Retrieval-Augmented Generation in AI or
Red-Amber-Green status reporting. After clarification, it creates and evaluates
the lesson.

The chat supports follow-up questions in the current app session. Each answer
is evaluated for:

- Directly answering the question and staying on topic.
- Accuracy, uncertainty, and avoiding unsupported claims.
- Beginner-friendly language.

The UI displays follow-up evaluation details and has **Start a new topic** and
**Exit session** controls. Conversations are kept in Streamlit session state;
there is no saved-session list or cross-restart chat history. Use **Download
lesson and evaluation** to save the lesson, initial evaluation, rejection log,
and follow-up questions, answers, and evaluations as JSON.

The UI does not require the learner to upload source material and does not
search the web. Without an external source, the model uses general knowledge;
the app labels this as not independently verified.

## Run the command-line interface

The CLI asks for a topic if `--topic` is omitted:

```cmd
python Content_Generator_Agent.py
```

Or provide the topic and options directly:

```cmd
python Content_Generator_Agent.py --topic "Photosynthesis"
python Content_Generator_Agent.py --topic "Introduction to RAG in AI" --max-retries 1 --output outputs\lesson.json
```

By default, the CLI writes `lesson_result.json` beside
`Content_Generator_Agent.py`. It creates or overwrites that file on each run.
Use `--output` to choose a different location. The Streamlit UI instead offers
a download button; it does not automatically write a result JSON to the
project folder.

## How it works

1. **One LangChain agent:** `build_lesson_agent` creates a
   `ChatPromptTemplate`, a tools agent with `create_openai_tools_agent`, and an
   `AgentExecutor`. The same executor handles topic clarification, lesson
   generation, follow-up answers, and self-evaluation. It is one agent, not
   separate generator/evaluator agents.
2. **Topic clarification:** the agent returns `TOPIC_CLEAR` for a clear topic
   or a plain-text question when multiple meanings are likely. Clarification is
   not a guarantee that every ambiguity will be detected.
3. **Lesson generation and retries:** for each draft, the agent writes the
   lesson and returns pass/fail decisions for all seven rubric checks. Python
   validates the response and treats missing, duplicate, or unexpected check
   IDs as a failure. Failed checks are included in the next prompt. The retry
   ceiling is two retries (three total drafts).
4. **Follow-up evaluation:** the same agent answers each follow-up with the
   lesson and chat history as context, and returns pass/fail results for the
   three follow-up checks. These records are kept in the current Streamlit
   session and included in the JSON download.
5. **Persistent failure memory:** `LessonMemory` stores lesson-attempt
   evaluations and aggregate counts/reasons for failed rubric checks in local
   SQLite (`lesson_memory.sqlite3` beside the agent file by default). Frequent
   failures become reminders in later lesson prompts. This is failure-informed
   prompting; it does not automatically edit the rubric or system prompt.
   Override the database location with `LESSON_MEMORY_DB`.
6. **LangSmith observability:** `@traceable` records lesson-generation,
   clarification, and follow-up runs when tracing is enabled. Lesson rubric
   results are also attached to the lesson-attempt run as
   `lesson_rubric_pass` feedback. If no LangSmith key is configured, local
   generation and SQLite memory still work, but traces and feedback are not
   sent.

## Lesson rubric

Each check is all-or-nothing:

- Explains the topic and why it matters.
- Is accurate and acknowledges uncertainty.
- Explains the main idea or process in a sensible order.
- Uses beginner-friendly language and explains technical terms.
- Teaches with a relevant example.
- Covers relevant limitations, exceptions, risks, or uncertainty.
- Has a coherent, standalone flow and recap.

The agent evaluates its own output, so a passing score is **not independent
fact verification**. General-knowledge answers may still be incorrect,
especially for specialized or time-sensitive topics. For high-stakes use,
verify important claims with authoritative sources and human review.


## Demo walkthrough

1. Run the Streamlit app and enter a topic. Demonstrate ambiguity clarification
   with `RAG`, then clarify the intended meaning.
2. Show the lesson, rubric result, and rejection log.
3. Ask a follow-up question and show its evaluation details.
4. Download the JSON and show the lesson attempts and follow-up evaluations.
5. Demonstrate the bounded retry loop with an offline test or a case where the
   model rejects its first draft.
6. Show the LangSmith trace and `lesson_rubric_pass` feedback if LangSmith is
   configured.
7. Explain how SQLite remembers recurring lesson rubric failures and how those
   reminders affect later prompts.
