"""
main.py — ProcEx CLI
Usage:
    python main.py --input paper.pdf --resolution 1080p --minutes 7
    python main.py --input paper.pdf --topic "Attention Mechanism" --resolution 4K --minutes 10
    python main.py --resume output/checkpoints/attention_mechanism_checkpoint.json
"""
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# ── Force UTF-8 output on Windows (cmd/PowerShell default to cp1252) ──────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure we can import from the procex root ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="ProcEx — Procedural Cinematic Explainer Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input attention_paper.pdf
  python main.py --input diabetes_textbook.pdf --topic "Pathophysiology of Diabetes" --minutes 10
  python main.py --input renal_chapter.pdf --topic "Anatomy of the Nephron" --resolution 4K
  python main.py --resume output/checkpoints/my_topic_checkpoint.json
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input PDF file. If omitted and --topic is given, a deep research report is auto-generated."
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="",
        help="Topic hint for naming the output, or the research topic when --input is omitted."
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        choices=["720p", "1080p", "4K", "720p_v", "1080p_v", "4K_v"],
        default="1080p",
        help="Output resolution: 720p/1080p/4K (landscape) or 720p_v/1080p_v/4K_v (portrait 9:16)"
    )
    parser.add_argument(
        "--minutes", "-m",
        type=float,
        default=5.0,
        help="Target video duration in minutes (default: 5.0, max: 10.0)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint JSON file to resume a failed run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory (default: output/)"
    )

    args = parser.parse_args()

    # Validate
    if not args.resume and not args.input and not args.topic:
        parser.error("Either --input, --topic (for deep research), or --resume is required")

    if args.input and not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.minutes > 10.0:
        print(f"⚠ Warning: {args.minutes} minutes is long — consider splitting the topic.")
        print("   Proceeding anyway...")

    # ── Run ───────────────────────────────────────────────────────────────────
    from config import ProcExConfig
    from orchestrator import ProcExOrchestrator

    cfg = ProcExConfig(output_root=args.output_dir)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           ProcEx — Procedural Cinematic Explainer         ║
╚══════════════════════════════════════════════════════════╝
  Input:       {args.input or 'N/A (deep research mode)'}
  Topic:       {args.topic or 'auto-detect'}
  Resolution:  {args.resolution}
  Duration:    {args.minutes} min (target)
  Output dir:  {args.output_dir}/videos/
""")

    # ── Deep research mode: no --input, topic provided ────────────────────────
    input_path = args.input or ""
    pipeline   = ProcExOrchestrator(cfg)   # init once — owns the llm client

    if not input_path and args.topic and not args.resume:
        print("[ProcEx] No --input provided — entering deep research mode...")
        from agents.deep_research import DeepResearchAgent

        research   = DeepResearchAgent(cfg, pipeline.llm)  # reuse orchestrator's llm
        input_path = research.research(
            topic          = args.topic,
            target_minutes = args.minutes,
        )
        print(f"[ProcEx] ✓ Research report generated: {input_path}")
        print(f"[ProcEx] ▶ Handing off to video pipeline...")

    try:
        output_path = pipeline.run(
            input_path          = input_path,
            topic_hint          = args.topic,
            resolution          = args.resolution,
            target_minutes      = args.minutes,
            resume_checkpoint   = args.resume,
        )
    except KeyboardInterrupt:
        print("\n\n[ProcEx] ⚠ Interrupted by user — saving checkpoint...")
        print("[ProcEx] Resume with:")
        slug = args.topic.lower().replace(" ", "_")[:50] if args.topic else "run"
        print(f"  python main.py --resume output/checkpoints/{slug}_checkpoint.json")
        sys.exit(0)

    print(f"\n✅ Done! Video saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())