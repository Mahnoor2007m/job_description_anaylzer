# AI Job Description Analyzer (LangGraph)

Paste a job description and get a structured hiring report, built with a 4-node
**LangGraph** workflow, **Gemini or Groq**, **Pydantic** and a **Streamlit** UI.

## Workflow

```
Job Description
   │
   ▼
Node 1  information_extraction  → title, company, experience, required skills
   ▼
Node 2  job_classification      → job category, employment type
   ▼
Node 3  skill_analysis          → technical skills, soft skills, top 3 skills
   ▼
Node 4  hiring_report           → final structured report
   ▼
END  (Hiring Report + JSON export)
```

State is shared between nodes via a `JDState` TypedDict. If any node fails, the
graph routes straight to `END` and the error is shown nicely.

## Features

- All 4 LangGraph nodes with state passing and conditional error routing
- Pydantic structured output for every LLM step
- Handles empty / too short / too long / non-job-description input gracefully
- Streamlit UI with colored **skill tags**
- **Export report as JSON** (download button in UI, file in CLI)
- Optional LangSmith tracing (set `LANGCHAIN_TRACING_V2=true` in `.env`)
- Offline unit tests using a fake LLM (no API key needed)

## Project structure

```
jd-analyzer/
├── app/
│   ├── config.py      # env/config
│   ├── schemas.py     # Pydantic models (+ HiringReport)
│   ├── state.py       # LangGraph state
│   ├── llm.py         # Groq / Gemini factory
│   ├── nodes.py       # the 4 nodes
│   ├── graph.py       # graph wiring
│   ├── analyzer.py    # public entry point (never raises)
│   └── utils.py       # validation + normalization helpers
├── samples/sample_jd.txt
├── tests/test_pipeline.py
├── main.py            # CLI
├── streamlit_app.py   # UI
└── requirements.txt
```

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set `LLM_PROVIDER` plus the matching key:
- Groq key: https://console.groq.com/keys
- Gemini key: https://aistudio.google.com/apikey

(On Mac/Linux: `source venv/bin/activate` and `cp .env.example .env`.)

## Run

```bat
streamlit run streamlit_app.py
```

or from the terminal:

```bat
python main.py --file samples/sample_jd.txt --provider groq
```

The CLI prints the report and saves `output/hiring_report.json`.

## Tests

```bat
pytest -q
```

## Example output

```
Job Title: AI Engineer
Company: ABC Technologies
Experience: 2-4 Years
Required Skills: Python, FastAPI, LangGraph, SQL
Job Category: Artificial Intelligence
Employment Type: Full-time, Hybrid
Technical Skills: Python, FastAPI, LangGraph, SQL
Soft Skills: Communication, Teamwork, Problem Solving
Top 3 Skills:
  1. Python
  2. LangGraph
  3. FastAPI
```

## Notes

- If a model name stops working, change `GROQ_MODEL` / `GEMINI_MODEL` in `.env`.
