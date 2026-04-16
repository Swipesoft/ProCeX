# ProcEx — Procedural Cinematic Explainer Pipeline

> Generate **cinematic, perfectly-synced educational videos** from a topic or PDF in minutes.
> Powered by Gemma 4 31B · Gemini TTS · Manim · Modal H100 inference.

---

## What It Does

ProcEx turns a topic string or a PDF document into a fully produced educational video — complete with narration audio, animated Manim scenes, code snippets, subtitles, and music. A 4-minute video with 10 scenes takes roughly **5 minutes** to generate end-to-end at a cost of **under $1.50**.

Unlike black-box video tools, every scene is generated as editable Python (Manim) code. You can open any `output/manim/scene_03.py`, tweak the animation, re-render just that clip, and drop it back into the final video.

---

## Architecture

```
CLI / API
   │
   ├─ --mode research (default)          ├─ --mode documentary (PDF input)
   │                                     │
   ▼                                     ▼
[DeepResearchAgent]              [DocumentaryParser]
   Gemma 4 31B agentic loop         Parses tagged PDF into scenes
   Tavily web search (2-3 calls)    (NARRATOR / STORY / TECHNICAL / VOICE)
   Summarises results               Preserves speaker attribution
   Writes full script               Builds bridge clauses between speakers
   │                                     │
   └──────────────────┬──────────────────┘
                      ▼
              [DomainRouter]
                  Classifies domain → ML_MATH / MEDICAL / CS / NCLEX
                  Loads domain skill pack (skills/*.yaml)
                  Sets image_gen_enabled, manim_style, element palette
                      │
                      ▼
              [ScriptWriter]  ← bypassed in research mode (script already written)
                  LLM → structured narration scenes JSON
                  ~2.5 scenes/minute, cinematic spoken-word style
                      │
                      ▼
              [SlopRefiner]  ← Stage 2.5 (post-script, pre-TTS)
                  8-pattern regex pre-filter (em-dash overuse, hype cadence,
                  throat-clearing openers, high-probability tokens, etc.)
                  LLM correction for flagged scenes only
                  Recalculates duration after rewrite
                      │
              ┌────────────────────────────────────┐
              │         PARALLEL STAGE A           │
              ├────────────────────────────────────┤
              │ [TTSAgent]        [VisualDirector] │
              │  Gemini TTS        Per-scene:      │
              │  word timestamps   MANIM           │
              │  audio.mp3         IMAGE_GEN       │
              │                    TEXT_ANIMATION  │
              └────────────────────────────────────┘
                      │
              ┌────────────────────────────────────────────────────┐
              │                  PARALLEL STAGE B                  │
              ├──────────────────┬──────────────────┬──────────────┤
              │  [ManimCoder]    │  [ImageGenAgent] │[VideoGenAgent]│
              │  Gemma 4 31B     │  Gemini Imagen   │  Novita I2V  │
              │  Manim Python    │  Ken-burns       │  Seedance    │
              │  per scene       │  fallback        │  (doc mode)  │
              ├──────────────────┴──────────────────┴──────────────┤
              │              [RendererAgent]                        │
              │   manim -qh → .mp4  |  FFmpeg ken-burns            │
              │              [VLMCritic]                            │
              │   Gemini vision → 5-point rubric → patch or reroute│
              └────────────────────────────────────────────────────┘
                      │
              [AssemblerAgent]
                  FFmpeg concat + audio overlay + music mix + SRT subtitles
                      │
              output/videos/<topic>.mp4
```

---

## Key Features

### Gemma 4 31B Provider Mode (`--provider gemma`)
- All LLM work routes to a **self-hosted Gemma 4 31B endpoint** on Modal (2× H100 SXM)
- 16 concurrent requests via vLLM continuous batching
- Agentic web research loop with Tavily (2-3 targeted searches)
- Native function calling for tool use
- Falls back gracefully to standard research path if research loop produces no script
- Image generation disabled in Gemma mode (Gemini Imagen not available)
- Opening hook upgrade skipped in Gemma mode

### Teaching Context (`--context`)
Every agent prompt is wrapped with your context string — top and bottom — so the entire pipeline stays locked to the intended audience, scope, and exclusions. TTSAgent strips the context tags before audio generation so it is never read aloud.

```powershell
python main.py `
  --topic "Lifetimes in Rust" `
  --provider gemma `
  --minutes 4 `
  --context "Teaching Rust lifetimes to a student who knows borrow checking.
             Focus on lifetime annotations in function signatures only.
             Exclude structs with lifetime params, trait objects, subtyping."
```

### VLMCritic (5-Point Rubric)
After every Manim render, Gemini vision inspects 3 keyframes at 75/88/97% of the clip:
- **Score 5**: Clean — accepted
- **Score 4**: Minor crowding — accepted
- **Score 3**: Moderate overlap — Claude patches the layout
- **Score 2**: Major collision — rerouted to VisualDirector for re-planning
- **Score 1**: Catastrophic — rerouted immediately

The critic also enforces margin rules: elements touching `x < -5.0` or `x > 5.0` in landscape (or `x < -3.5` / `x > 1.0` in portrait) are flagged as critical even without element overlaps.

### SlopRefiner
Eight anti-slop patterns applied between ScriptWriter and TTSAgent:

| Pattern | Example Bad | Fix |
|---|---|---|
| Em-dash overuse | `well — better than expected — on` | parentheses or restructure |
| X didn't just Y | `She didn't just finish, she also...` | two clean sentences |
| Town-crier hype | `And the wildest part?` | grounded reaction |
| Negative stacks | `No fuel. No gas. Just solar.` | single full sentence |
| High-probability tokens | `delve`, `state-of-the-art`, `groundbreaking` | specific concrete language |
| Throat-clearing opener | `In this video we will explore...` | start with the substance |
| Filler transitions | `Having said that,` / `With that in mind,` | remove entirely |
| Unlinked sentences | `X improved. Y also dropped.` | add conjunction |

### Checkpoint & Resume
Every stage writes a checkpoint. Resume any failed run:
```bash
python main.py --resume output/checkpoints/lifetimes_in_rust_checkpoint.json
```

### Parallel Execution
Stage A (TTSAgent + VisualDirector) and Stage B (ManimCoder + Render + Critic) run in parallel thread pools. Worker counts are configurable in `config.py`.

---

## Providers & LLM Routing

| Task | Gemma Mode | Standard Mode |
|---|---|---|
| Deep research | Gemma 4 31B (Modal) | Gemini / Claude |
| ScriptWriter | Gemma 4 31B | Claude → Gemini → GPT-4o |
| VisualDirector | Gemma 4 31B | Gemini → Claude |
| ManimCoder | Gemma 4 31B | Claude → Gemini |
| SlopRefiner | Gemma 4 31B | Claude |
| TTSAgent | Gemini TTS (always) | Gemini TTS |
| VLMCritic | Gemma 4 31B | Gemini Vision |
| Image generation | Disabled | Gemini Imagen |
| Video generation | Disabled | Novita Seedance |

---

## Quick Start

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg texlive-latex-recommended texlive-fonts-extra texlive-latex-extra

# macOS
brew install ffmpeg
# Install MacTeX from https://tug.org/mactex/
```

**Windows (PowerShell):**
```powershell
# Python 3.12 required — Manim does not support 3.13
winget install Python.Python.3.12

# Create and activate venv
py -3.12 -m venv venv

# Allow script execution
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

.\venv\Scripts\activate
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
pip install manim   # large dependency — install separately
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# TTS
# Gemini TTS is used by default — GEMINI_API_KEY above covers it

# Web Search (for research mode)
TAVILY_API_KEY=tvly-...

# Image & Video Generation
NOVITA_API_KEY=...          # Novita Seedance I2V (documentary mode)

# Modal Gemma Endpoint (for --provider gemma)
MODAL_GEMMA_URL=https://your-workspace--gemma4-31b-serve.modal.run
MODAL_INTERNAL_SECRET=your-64-char-secret

# Optional model name override (default: google/gemma-4-31B-it)
GEMMA_MODEL_NAME=google/gemma-4-31B-it
```

### 4. Run

```powershell
# Research mode — Gemma provider, portrait video
python main.py `
  --topic "Lifetimes in Rust" `
  --mode research `
  --provider gemma `
  --minutes 4 `
  --style youtube-tutorial `
  --resolution 1080p_v `
  --context "Teaching Rust lifetimes to a student who knows borrow checking
             and Rust datatypes. Focus on lifetime annotations in function
             signatures and return types only."

# Documentary mode — PDF input
python main.py `
  --input attention_is_all_you_need.pdf `
  --topic "Attention Mechanism" `
  --minutes 5

# Resume a failed run
python main.py --resume output/checkpoints/lifetimes_in_rust_checkpoint.json
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--topic` | required | Video topic |
| `--input` | none | PDF input path (documentary mode) |
| `--mode` | `research` | `research` or `documentary` |
| `--provider` | `standard` | `gemma` (Modal H100) or `standard` (Claude/Gemini/GPT-4o) |
| `--minutes` | `5.0` | Target video duration |
| `--style` | `youtube-tutorial` | Skill pack style (see `skills/styles/`) |
| `--resolution` | `1080p` | `1080p`, `1080p_v` (portrait), `4K` |
| `--context` | none | Teaching perspective/audience/scope string |
| `--resume` | none | Path to checkpoint JSON |

---

## Skill Packs

Each domain and style has a `skills/*.yaml` file that controls the entire pipeline behaviour:

```
skills/
├── domains/
│   ├── ml_math.yaml        # ML & Mathematics — Manim-first, image gen disabled
│   ├── medical.yaml        # Medical — IMAGE_GEN for anatomy, MANIM for pathophysiology
│   ├── cs.yaml             # Computer Science — code-heavy Manim
│   └── nclex.yaml          # Nursing — clinical judgment framework
├── styles/
│   ├── youtube-tutorial.yaml   # Engaging tutorial style with code focus
│   ├── tiktok-thriller.yaml    # Fast-paced, dramatic hooks
│   └── documentary.yaml        # Long-form, narrative-driven
└── refine/
    └── anti_slop.yaml          # 8 slop patterns for SlopRefiner
```

Each domain yaml includes:
- `image_gen_enabled` — hard gate on image generation for the domain
- `manim_style` — colour palette, animation preferences
- `manim_elements` — available Manim primitives
- `script_instructions` — ScriptWriter personality
- `image_gen_style` — image generation prompt style guide

---

## Visual Strategy Decision Logic

The VisualDirector reads the domain skill pack and decides per scene:

| Scene Content | Strategy | Why |
|---|---|---|
| Gradient descent, matrix ops, proofs | `MANIM` | Mathematical — exact animations |
| PyTorch / Rust / Python code snippets | `MANIM` | Code() objects, syntax highlighted |
| Pathophysiology mechanisms | `MANIM` | Flowchart — no anatomy needed |
| Anatomy of the nephron | `IMAGE_GEN` | Spatial structure needs real imagery |
| Cross-section of the heart | `IMAGE_GEN` | Anatomical layout |
| Opening hook / cinematic transition | `TEXT_ANIMATION` | Title card in Manim |
| Historical figure speaking | `VIDEO_GEN` | I2V via Novita Seedance (doc mode) |

**Rule: when in doubt → MANIM. IMAGE_GEN only for spatial anatomy or real-world scenes.**

---

## Output Files

```
output/
├── videos/
│   ├── lifetimes_in_rust.mp4         ← final assembled video
│   └── lifetimes_in_rust.srt         ← subtitle file
├── audio/
│   └── lifetimes_in_rust.mp3         ← Gemini TTS audio
├── scenes/
│   ├── scene_01.mp4                  ← individual rendered clips
│   └── scene_02.mp4
├── manim/
│   ├── scene_01.py                   ← generated Manim code (fully editable)
│   └── scene_02.py
├── images/
│   ├── scene_05_raw.png              ← Gemini Imagen raw output
│   └── scene_05_labeled.png          ← with label overlay
└── checkpoints/
    └── lifetimes_in_rust_checkpoint.json
```

---

## Editing & Re-rendering

Because every scene is generated code, you can fix any scene without re-running the whole pipeline:

```bash
# 1. Edit the Manim code
code output/manim/scene_03.py

# 2. Re-render just that scene
manim -qh output/manim/scene_03.py Scene03

# 3. Drop the new clip into place
cp media/videos/scene_03/.../Scene03.mp4 output/scenes/scene_03.mp4

# 4. Re-assemble the final video (use checkpoint — skips all other stages)
python main.py --resume output/checkpoints/your_checkpoint.json
```

---

## Modal Deployment (Gemma 4 31B)

The `modal_serve.py` script deploys Gemma 4 31B on 2× H100 SXM via Modal's serverless platform:

```bash
# One-time secret creation
modal secret create modal-internal-secret MODAL_INTERNAL_SECRET=<random-64-char-string>

# Deploy
modal deploy modal_serve.py

# Smoke test
MODAL_INTERNAL_SECRET=<your-secret> modal run modal_serve.py
```

**Specs:**
- Model: `google/gemma-4-31B-it` (Apache 2.0, no HF token needed)
- Context: 65,536 tokens (configurable via `--max-model-len`)
- Concurrency: 32 simultaneous requests (`@modal.concurrent(max_inputs=32)`)
- Scale-to-zero: after 3 minutes idle
- Cold start: ~3-5 minutes (weights cached in Modal Volume after first deploy)

**Cost:** ~$0.39/minute for 2× H100 SXM. A 4-minute video pipeline takes ~5 minutes → **~$1.97 per video**.

---

## Performance

| Video Length | Scenes | Provider | Total Time | Cost |
|---|---|---|---|---|
| 2 min | 5 | Gemma (Modal) | ~3 min | ~$0.80 |
| 4 min | 10 | Gemma (Modal) | ~5 min | ~$1.50 |
| 4 min | 10 | Standard (Claude) | ~15 min | ~$1.20 |

*Times measured on 12-core Windows machine with 6 render workers.*

---

## Config Reference (`config.py`)

```python
# Parallel worker counts
tts_workers:         int   = 8    # parallel TTS API calls
coder_workers:       int   = 8    # parallel ManimCoder LLM calls
image_workers:       int   = 6    # parallel ImageGen API calls
render_workers:      int   = 6    # parallel Manim subprocesses

# Rendering
manim_timeout_secs:  int   = 420  # 7 min per scene
enable_critic_loop:  bool  = True # VLMCritic quality gate

# Script
scenes_per_minute:   float = 2.5  # ~24s avg per scene
subscene_split_threshold_secs: float = 40.0
```

---

## Project Structure

```
procex/
├── main.py                     ← CLI entry point
├── orchestrator.py             ← pipeline coordinator + checkpointing
├── parallel_runner.py          ← Stage B parallel execution engine
├── state.py                    ← ProcExState dataclass (checkpointable)
├── config.py                   ← all configuration
├── modal_serve.py              ← Modal H100 Gemma deployment
│
├── agents/
│   ├── domain_router.py        ← domain classification
│   ├── script_writer.py        ← narration scene generation
│   ├── deep_research.py        ← agentic web research + script writing
│   ├── tts_agent.py            ← Gemini TTS audio generation
│   ├── visual_director.py      ← per-scene visual strategy
│   ├── manim_coder.py          ← Manim Python code generation
│   ├── image_gen_agent.py      ← Gemini Imagen + ken-burns
│   ├── video_gen_agent.py      ← Novita Seedance I2V
│   ├── renderer_agent.py       ← Manim CLI render
│   ├── vlm_critic.py           ← vision quality gate (5-point rubric)
│   ├── assembler_agent.py      ← FFmpeg final assembly
│   └── documentary_parser.py  ← PDF → tagged scenes
│
├── utils/
│   ├── gemma_client.py         ← TogetherAI Gemma client (backup)
│   ├── modal_gemma_client.py   ← Modal vLLM Gemma client (active)
│   ├── llm_client.py           ← multi-provider LLM client
│   ├── context_injection.py    ← wrap_with_context + strip_context_tags
│   ├── slop_refiner.py         ← 8-pattern narration quality pass
│   ├── music_mixer.py          ← background music overlay
│   └── documentary_parser.py  ← paragraph type tagging utilities
│
└── skills/
    ├── domains/
    │   ├── ml_math.yaml
    │   ├── medical.yaml
    │   └── cs.yaml
    ├── styles/
    │   ├── youtube-tutorial.yaml
    │   └── tiktok-thriller.yaml
    └── refine/
        └── anti_slop.yaml
```

---

## Output Cleanup

```bash
# Preview what would be deleted
python flush_output.py --dry-run

# Clean all outputs
python flush_output.py --yes
```

---

## Known Limitations

- Manim requires Python 3.12 — not compatible with 3.13
- Portrait mode (`1080p_v`) in Gemma mode may have tighter VLMCritic margins — reroutes are more frequent
- Very long context strings (>500 chars) in `--context` may push ManimCoder prompts close to the 65K context window limit — keep context concise
- Documentary mode (PDF input) currently only tested in standard provider mode

---

## License

Apache 2.0 — see `LICENSE` file.

Built by [Swipesoft](https://github.com/Swipesoft) · Submitted to Gemma 4 Good Hackathon 2026