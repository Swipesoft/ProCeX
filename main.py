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

    parser.add_argument(
        "--style", "-s",
        type=str,
        default="auto",
        help=(
            "Presentation style. Options: "
            "auto (LLM picks based on topic+duration), "
            "tiktok-scifi (sci-fi dramatic, <1.5min), "
            "tiktok-thriller (historical story, <2min), "
            "youtube-tutorial (deep explainer, 4-10min)"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["research", "documentary"],
        default=None,
        help=(
            "Generation mode when no --input is given. "
            "'research' = academic deep research report (default when only --topic given). "
            "'documentary' = Netflix-style multi-voice documentary with Tavily web search."
        )
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemma"],
        default=None,
        help=(
            "LLM provider override. 'gemma' activates Gemma 4 31B end-to-end "
            "(research mode only). All agents except TTS route through Gemma. "
            "Image generation is disabled. Resolution forced to 1080p landscape."
        )
    )

    args = parser.parse_args()

    # Validate
    if not args.resume and not args.input and not args.topic:
        parser.error("Either --input, --topic, or --resume is required")

    if args.mode and args.input:
        parser.error("--mode and --input are mutually exclusive. Use --mode for research/documentary, --input for PDF teaching.")

    if getattr(args, "provider", None) == "gemma" and args.mode == "documentary":
        parser.error("--provider gemma is only supported with --mode research (not documentary).")

    if args.input and not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.minutes > 10.0:
        print(f"⚠ Warning: {args.minutes} minutes is long — consider splitting the topic.")
        print("   Proceeding anyway...")

    # ── Run ───────────────────────────────────────────────────────────────────
    from config import ProcExConfig
    from orchestrator import ProcExOrchestrator

    cfg = ProcExConfig(output_root=args.output_dir, presentation_style=args.style)

    # ── Gemma provider mode ───────────────────────────────────────────────────
    if getattr(args, "provider", None) == "gemma":
        cfg.gemma_provider = True
        # Force landscape 1080p — portrait has too many layout constraints
        # that can cause over-rejection in VLMCritic when image_gen is off.
        if args.resolution.endswith("_v"):
            print("[ProcEx] ℹ Gemma mode: portrait resolution overridden → 1080p")
            args.resolution = "1080p"
        print("[ProcEx] 🟣 Gemma 4 31B mode active")
        print("[ProcEx]   • All text/vision/code agents → gemma-4-31b-it")
        print("[ProcEx]   • Image generation → disabled")
        print("[ProcEx]   • TTS → Gemini (unchanged)")
        print("[ProcEx]   • Research → agentic Tavily function-calling loop")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           ProcEx — Procedural Cinematic Explainer         ║
╚══════════════════════════════════════════════════════════╝
  Input:       {args.input or f"N/A ({args.mode or 'research'} mode)"}
  Topic:       {args.topic or 'auto-detect'}
  Resolution:  {args.resolution}
  Duration:    {args.minutes} min (target)
  Style:       {args.style}
  Output dir:  {args.output_dir}/videos/
""")

    # ── Mode dispatch: research / documentary / PDF input ────────────────────
    input_path = args.input or ""
    pipeline   = ProcExOrchestrator(cfg)   # init once — owns the llm client

    if not input_path and args.topic and not args.resume:
        mode = args.mode or "research"   # default to research if --mode not given

        if mode == "documentary":
            print("[ProcEx] ▶ Entering deep documentary mode (Tavily + multi-voice)...")
            from agents.deep_documentary import DeepDocumentaryAgent
            doc_agent  = DeepDocumentaryAgent(cfg, pipeline.llm)
            input_path = doc_agent.research(
                topic          = args.topic,
                target_minutes = args.minutes,
            )
            print(f"[ProcEx] ✓ Documentary script generated: {input_path}")
            print(f"[ProcEx] ▶ Handing off to video pipeline...")
            # Auto-select tiktok-thriller if style is auto for documentary
            if args.style == "auto":
                cfg = cfg.__class__(
                    output_root        = args.output_dir,
                    presentation_style = "tiktok-thriller",
                )
                pipeline = ProcExOrchestrator(cfg)
                print("[ProcEx] ℹ Auto-selected style: tiktok-thriller for documentary")

        else:  # research (default)
            print("[ProcEx] ▶ No --input provided — entering deep research mode...")
            from agents.deep_research import DeepResearchAgent
            research   = DeepResearchAgent(cfg, pipeline.llm)
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
            presentation_style  = args.style,
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