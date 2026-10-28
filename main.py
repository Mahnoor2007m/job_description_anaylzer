"""CLI: python main.py --file samples/sample_jd.txt [--provider groq|gemini]"""
import argparse
import sys
from pathlib import Path

from app.analyzer import analyze_job_description


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Job Description Analyzer")
    parser.add_argument("--file", help="Path to a text file containing the job description")
    parser.add_argument("--provider", choices=["groq", "gemini"], help="LLM provider")
    parser.add_argument("--out", default="output/hiring_report.json", help="JSON output path")
    args = parser.parse_args()

    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Could not read file: {exc}")
            return 1
    else:
        print("Paste the job description, then press Ctrl+Z + Enter (Windows) or Ctrl+D (Mac/Linux):")
        text = sys.stdin.read()

    result = analyze_job_description(text, provider=args.provider)
    if not result.ok:
        print(f"\nError: {result.error}")
        return 1

    print("\n===== HIRING REPORT =====")
    print(result.report.to_text())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nJSON saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
