"""Streamlit UI: streamlit run streamlit_app.py"""
import html
import re
from pathlib import Path

import streamlit as st

from app.analyzer import analyze_job_description
from app.config import DEFAULT_PROVIDER

st.set_page_config(page_title="AI Job Description Analyzer", page_icon="🧠", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "samples" / "sample_jd.txt"

st.markdown(
    """
    <style>
    .tag {display:inline-block; padding:4px 12px; margin:3px 6px 3px 0; border-radius:999px;
          font-size:0.85rem; font-weight:500; border:1px solid transparent;}
    .tag-req  {background:#e0e7ff; color:#3730a3; border-color:#c7d2fe;}
    .tag-tech {background:#dcfce7; color:#166534; border-color:#bbf7d0;}
    .tag-soft {background:#fef3c7; color:#92400e; border-color:#fde68a;}
    .tag-top  {background:#fee2e2; color:#991b1b; border-color:#fecaca; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def tags(items, css_class: str) -> str:
    if not items:
        return "<em>Not specified</em>"
    return "".join(
        f'<span class="tag {css_class}">{html.escape(str(i))}</span>' for i in items
    )


def load_sample():
    st.session_state["jd_text"] = SAMPLE_PATH.read_text(encoding="utf-8")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox(
        "LLM provider", ["groq", "gemini"],
        index=0 if DEFAULT_PROVIDER != "gemini" else 1,
    )
    api_key = st.text_input(
        "API key (optional)", type="password",
        help="Leave empty to use the key from your .env file.",
    )
    st.divider()
    st.caption("Pipeline: Extraction → Classification → Skill Analysis → Hiring Report")

# ---------------- Main ----------------
st.title("🧠 AI Job Description Analyzer")
st.caption("Paste a job description and get a structured hiring report (LangGraph + Gemini/Groq).")

st.text_area(
    "Job description", key="jd_text", height=260,
    placeholder="Paste the full job description here...",
)

col_a, col_b, _ = st.columns([1, 1, 4])
analyze_clicked = col_a.button("🔍 Analyze", type="primary", use_container_width=True)
col_b.button("📄 Load sample", on_click=load_sample, use_container_width=True)

if analyze_clicked:
    with st.spinner("Running the LangGraph pipeline..."):
        result = analyze_job_description(
            st.session_state.get("jd_text", ""),
            provider=provider,
            api_key=api_key.strip() or None,
        )
    st.session_state["result"] = result

result = st.session_state.get("result")

if result is not None:
    if not result.ok:
        st.error(result.error)
    else:
        report = result.report
        st.success("Hiring report generated.")

        tab_report, tab_steps, tab_json = st.tabs(["📋 Hiring Report", "🔗 Pipeline Steps", "{ } JSON"])

        with tab_report:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Job Title", report.job_title)
            m2.metric("Company", report.company)
            m3.metric("Experience", report.experience)
            m4.metric("Category", report.job_category)

            st.markdown("**Employment Type**")
            st.markdown(tags(report.employment_type, "tag-req"), unsafe_allow_html=True)
            st.markdown("**Required Skills**")
            st.markdown(tags(report.required_skills, "tag-req"), unsafe_allow_html=True)
            st.markdown("**Technical Skills**")
            st.markdown(tags(report.technical_skills, "tag-tech"), unsafe_allow_html=True)
            st.markdown("**Soft Skills**")
            st.markdown(tags(report.soft_skills, "tag-soft"), unsafe_allow_html=True)
            st.markdown("**Top 3 Most Important Skills**")
            for i, skill in enumerate(report.top_3_skills, 1):
                st.markdown(
                    f'{i}. <span class="tag tag-top">{html.escape(skill)}</span>',
                    unsafe_allow_html=True,
                )

        with tab_steps:
            state = result.state or {}
            st.markdown("**Node 1 – Information Extraction**")
            st.json(state["extraction"].model_dump())
            st.markdown("**Node 2 – Job Classification**")
            st.json(state["classification"].model_dump())
            st.markdown("**Node 3 – Skill Analysis**")
            st.json(state["skill_analysis"].model_dump())
            st.markdown("**Node 4 – Hiring Report (text)**")
            st.code(report.to_text(), language="text")

        with tab_json:
            json_text = report.model_dump_json(indent=2)
            st.code(json_text, language="json")
            slug = re.sub(r"[^a-z0-9]+", "_", report.job_title.lower()).strip("_") or "report"
            st.download_button(
                "⬇️ Download report as JSON",
                data=json_text,
                file_name=f"hiring_report_{slug}.json",
                mime="application/json",
            )
