"""Offline tests: a fake LLM replaces Gemini/Groq, so no API key or internet is needed."""
import pytest

from app.analyzer import analyze_job_description
from app.schemas import ClassificationResult, ExtractionResult, SkillAnalysis
from app.utils import (
    finalize_top3, normalize_category, normalize_employment, validate_input,
)

JD = "AI Engineer at ABC Technologies. Full-time hybrid. 2-4 years. Python, FastAPI, LangGraph, SQL."


class FakeLLM:
    """Mimics llm.with_structured_output(schema).invoke(messages)."""

    def __init__(self, is_jd=True):
        self.is_jd = is_jd

    def with_structured_output(self, schema):
        outer = self

        class _Runner:
            def invoke(self, _messages):
                if schema is ExtractionResult:
                    return ExtractionResult(
                        is_job_description=outer.is_jd, job_title="AI Engineer",
                        company_name="ABC Technologies", required_experience="2-4 Years",
                        required_skills=["Python", "FastAPI", "LangGraph", "SQL", "python"],
                    )
                if schema is ClassificationResult:
                    return ClassificationResult(
                        job_category="AI", employment_type=["full time", "Hybrid"]
                    )
                return SkillAnalysis(
                    technical_skills=["Python", "FastAPI", "LangGraph", "SQL"],
                    soft_skills=["Communication", "Teamwork", "Problem Solving"],
                    top_3_skills=["Python", "LangGraph", "FastAPI", "SQL"],
                )

        return _Runner()


def test_full_pipeline_with_fake_llm():
    result = analyze_job_description(JD, llm=FakeLLM())
    assert result.ok
    r = result.report
    assert r.job_title == "AI Engineer"
    assert r.required_skills == ["Python", "FastAPI", "LangGraph", "SQL"]  # de-duplicated
    assert r.job_category == "Artificial Intelligence"
    assert r.employment_type == ["Full-time", "Hybrid"]
    assert r.top_3_skills == ["Python", "LangGraph", "FastAPI"]  # capped at 3
    assert "Top 3 Skills" in r.to_text()


def test_not_a_job_description_stops_graph():
    result = analyze_job_description("Today the weather is really nice in the city.", llm=FakeLLM(is_jd=False))
    assert not result.ok
    assert "job description" in result.error.lower()


@pytest.mark.parametrize("bad", [None, "", "   ", "hi", "12345 67890 !!!! 12345 67890 !!!! 12345"])
def test_invalid_input_is_handled(bad):
    result = analyze_job_description(bad, llm=FakeLLM())
    assert not result.ok and result.error


def test_validate_input_ok():
    assert validate_input("  " + JD + "  ") == JD


def test_normalizers():
    assert normalize_category("web development") == "Web Development"
    assert normalize_category("gardening") == "Other"
    assert normalize_employment(["onsite", "Intern"]) == ["On-site", "Internship"]
    assert normalize_employment([]) == ["Not specified"]
    assert finalize_top3(["A"], ["B", "C", "D"]) == ["A", "B", "C"]
