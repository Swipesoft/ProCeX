# ProcEx — Procedural Cinematic Explainer Pipeline

Generate **cinematic, perfectly-synced educational videos** from PDF papers or textbooks.
Up to 10 minutes. Fully deterministic, editable, reproducible.

---

## Architecture

```
PDF Input
   ↓
[DomainRouter]          Classifies domain (ML_MATH / MEDICAL / CS / NCLEX)
                        Loads domain skill pack (skills/*.yaml)
   ↓
[ScriptWriter]          LLM → structured narration scenes JSON
                        ~2.5 scenes/minute, cinematic documentary style
   ↓
[TTSAgent]              ElevenLabs convert_with_timestamps
                        audio.mp3 + character→word timestamp extraction
   ↓
[VisualDirector]        ← THE KEY AGENT
                        Per-scene: MANIM | IMAGE_GEN | HYBRID | TEXT_ANIMATION
                        Medical pathophysiology → MANIM (flowchart)
                        Medical anatomy → IMAGE_GEN (NanoBanana Pro)
   ↓
[ManimCoder]            Generates timestamp-synced Manim Python per scene
[ImageGenAgent]         NanoBanana Pro/Fast for anatomy scenes + label overlay
   ↓
[RendererAgent]         Manim CLI render / Ken Burns FFmpeg / hybrid composite
   ↓
[AssemblerAgent]        FFmpeg concat + audio overlay + SRT subtitles
   ↓
output/videos/<topic>.mp4
```

---

## Quick Start

### 1. System dependencies
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg texlive-latex-recommended texlive-fonts-extra

# macOS
brew install ffmpeg
```

### 2. Python dependencies
```bash
pip install -r requirements.txt
pip install manim   # separately — large dependency
```

### 3. API Keys
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...
```

### 4. Run
```bash
# 5-minute ML paper explainer
python main.py --input attention_is_all_you_need.pdf --topic "Attention Mechanism" --minutes 5

# 10-minute medical video
python main.py --input diabetes_chapter.pdf --topic "Pathophysiology of Diabetes" --minutes 10 --resolution 4K

# Resume a failed run
python main.py --resume output/checkpoints/attention_mechanism_checkpoint.json
```

---

## Visual Strategy Decision Logic

The **VisualDirector** is the core intelligence. It reads the domain skill pack
and decides per-scene which renderer to use:

| Scene Content | Strategy | Why |
|---|---|---|
| Gradient descent, backprop, SVD | `MANIM` | Mathematical — Manim is exact |
| Pathophysiology of diabetes | `MANIM` | Mechanism flowchart — no anatomy needed |
| Pharmacology of opioids | `MANIM` | Receptor diagram in Manim |
| Anatomy of the nephron | `IMAGE_GEN` | Viewer needs to *see* the structure spatially |
| Cross-section of the heart | `IMAGE_GEN` | Real anatomical spatial layout |
| Nephron anatomy + filtration mechanism | `IMAGE_MANIM_HYBRID` | Image bg + Manim overlay arrows |
| Opening hook / section transition | `TEXT_ANIMATION` | Cinematic title card in Manim |

**Rule: When in doubt → MANIM. IMAGE_GEN only for spatial anatomy.**

---

## Skill Packs

Each domain has a `skills/*.yaml` file that injects:
- `script_instructions` → ScriptWriter personality
- `image_gen_triggers` → words that hint at IMAGE_GEN
- `manim_style` → color palette, animation preferences
- `manim_elements` → available Manim primitives for this domain

Customize these to tune the pipeline for your content.

---

## Output Files

```
output/
├── videos/
│   ├── attention_mechanism.mp4       ← final video (with audio)
│   ├── attention_mechanism.srt       ← subtitles
│   └── attention_mechanism_cc.mp4    ← captioned version (optional)
├── audio/
│   └── attention_mechanism.mp3       ← TTS audio
├── scenes/
│   ├── scene_01.mp4                  ← individual rendered clips
│   └── scene_02.mp4
├── manim/
│   ├── scene_01.py                   ← generated Manim code (editable!)
│   └── scene_02.py
├── images/
│   ├── scene_05_raw.png              ← NanoBanana raw output
│   └── scene_05_labeled.png          ← with PIL label overlay
└── checkpoints/
    └── attention_mechanism_checkpoint.json  ← resume from here if interrupted
```

---

## Editing & Re-rendering

Because everything is code, you can:
1. Edit `output/manim/scene_03.py` to fix an animation
2. Re-run `manim -qh scene_03.py Scene03` to re-render just that clip
3. Replace `output/scenes/scene_03.mp4` with the new render
4. Re-run the AssemblerAgent only (use checkpoint)

This is the core advantage over black-box tools.

---

## LLM Fallback Chain

```
Claude (claude-sonnet-4-6)  →  Gemini (gemini-2.0-flash)  →  OpenAI (gpt-4o)
```

If any provider fails or has no API key, the next is tried automatically.
NanoBanana (image generation) uses Gemini only — no fallback for images.

---

## Cost Estimate (per 10-minute video)

| Component | Approx Cost |
|---|---|
| ScriptWriter (LLM) | ~$0.10 |
| VisualDirector (LLM) | ~$0.15 |
| ManimCoder 15 scenes (LLM) | ~$0.40 |
| ElevenLabs TTS (~1500 words) | ~$0.15 |
| NanoBanana Pro (3-5 anatomy images) | ~$0.20 |
| **Total** | **~$1.00-$1.50** |

Manim rendering is free (local CPU/GPU).
