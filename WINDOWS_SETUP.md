# ProcEx — Windows Setup & Usage Guide

Complete guide for running ProcEx on Windows 10/11.
No WSL required — everything runs natively.

---

## What You Are Installing

| Tool | Purpose |
|---|---|
| Python 3.11+ | Runs the pipeline |
| FFmpeg | Video assembly, Ken Burns effect, audio sync |
| MiKTeX | LaTeX renderer — required by Manim for equations |
| Manim Community | Renders animated scenes |
| Python packages | LLM SDKs, TTS, PDF parsing, image processing |

---

## Step 1 — Python 3.11+

Download from https://python.org/downloads

During install:
- Tick **"Add Python to PATH"**
- Tick **"Install for all users"** (recommended)

Verify in a new terminal:
```
python --version
pip --version
```

---

## Step 2 — FFmpeg

**Option A — winget (Windows 11, easiest):**
```
winget install Gyan.FFmpeg
```

**Option B — Chocolatey:**
```
choco install ffmpeg
```

**Option C — Manual:**
1. Download `ffmpeg-release-essentials.zip` from https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to PATH:
   - Start -> Search "Environment Variables"
   - Under System Variables -> select Path -> Edit -> New -> type `C:\ffmpeg\bin`
   - Click OK on all dialogs
4. Restart your terminal

Verify:
```
ffmpeg -version
```

---

## Step 3 — MiKTeX (LaTeX for Manim equations)

Download from https://miktex.org/download — choose the Windows installer.

During setup:
- Install for: All users (recommended)
- Install missing packages on-the-fly: Yes  <-- critical

After install, open MiKTeX Console from Start Menu and click Check for updates -> Update now.

Verify:
```
latex --version
```

First run note: When Manim renders a MathTex equation for the first time,
MiKTeX will pop up a dialog asking to install missing packages. Click Install.
This only happens once.

---

## Step 4 — Manim Community

```
pip install manim
```

Verify:
```
manim --version
```

If you get a Cairo or Pango error:
```
pip install manim[cairo]
```

---

## Step 5 — Python Packages

Navigate to the procex\ folder in your terminal, then:

```
pip install -r requirements.txt
```

This installs: anthropic, google-genai, openai, elevenlabs, pymupdf,
Pillow, PyYAML, numpy, and other utilities.

---

## Step 6 — API Keys

You need at least one LLM key and one ElevenLabs key to run the pipeline.

**Option A — .env file (recommended):**

Create a file called `.env` inside the procex\ folder:
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
```

Then install python-dotenv:
```
pip install python-dotenv
```

Add these two lines at the very top of main.py (before all other imports):
```python
from dotenv import load_dotenv
load_dotenv()
```

**Option B — PowerShell session variables:**
```powershell
$env:ANTHROPIC_API_KEY  = "sk-ant-..."
$env:GEMINI_API_KEY     = "AIza..."
$env:ELEVENLABS_API_KEY = "..."
```

**Option C — Permanent system environment variables:**
Start -> Search "Environment Variables" -> System Variables -> New
Add each key as a new variable.

Where to get keys:
- Anthropic:  https://console.anthropic.com
- Gemini:     https://aistudio.google.com/apikey
- OpenAI:     https://platform.openai.com/api-keys
- ElevenLabs: https://elevenlabs.io (Dashboard -> API Keys)

---

## Step 7 — Enable Long Path Support

Manim and FFmpeg create deeply nested paths that can exceed Windows' 260-character limit.

Run this in PowerShell as Administrator:
```powershell
New-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" `
  -Value 1 `
  -PropertyType DWORD `
  -Force
```

Then restart your terminal.

---

## Step 8 — Enable UTF-8 Console Output

Windows CMD/PowerShell defaults to cp1252 encoding. ProcEx uses Unicode.

Add this to your PowerShell profile (run `notepad $PROFILE` to open it):
```
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
```

Or just prefix every run with:
```
set PYTHONUTF8=1 && python main.py ...
```

---

## Running ProcEx

Open PowerShell or Windows Terminal in the procex\ folder.

### Basic — 5-minute video:
```powershell
python main.py --input paper.pdf --minutes 5
```

### Full example — 7-minute medical video:
```powershell
python main.py `
    --input "C:\Users\YourName\Downloads\diabetes_chapter.pdf" `
    --topic "Pathophysiology of Type 2 Diabetes" `
    --minutes 7 `
    --resolution 1080p
```

### 10-minute 4K anatomy video:
```powershell
python main.py `
    --input renal_anatomy.pdf `
    --topic "Anatomy of the Nephron" `
    --minutes 10 `
    --resolution 4K
```

### Resume a failed run:
```powershell
python main.py --resume output\checkpoints\attention_mechanism_checkpoint.json
```

### Fast test run — 720p, 3 minutes (quickest):
```powershell
python main.py --input paper.pdf --minutes 3 --resolution 720p
```

---

## Output Files

```
procex\
└── output\
    ├── videos\
    │   ├── your_topic.mp4          <- FINAL VIDEO
    │   ├── your_topic.srt          <- subtitle file
    │   └── your_topic_cc.mp4       <- captioned version
    ├── audio\
    │   └── your_topic.mp3          <- TTS narration
    ├── scenes\
    │   ├── scene_01.mp4            <- individual rendered clips
    │   └── scene_02.mp4
    ├── manim\
    │   ├── scene_01.py             <- generated Manim code (editable!)
    │   └── scene_02.py
    ├── images\
    │   ├── scene_05_raw.png        <- NanoBanana anatomy image
    │   └── scene_05_labeled.png    <- with callout labels
    └── checkpoints\
        └── your_topic_checkpoint.json   <- resume here if pipeline crashes
```

Open the final .mp4 in VLC or Windows Media Player.

---

## Re-rendering a Single Scene

Because all Manim code is saved to output\manim\, you can edit and re-render individual scenes:

```powershell
# Edit the scene in any text editor
notepad output\manim\scene_03.py

# Re-render just that one scene
cd output\manim
manim -qh scene_03.py Scene03

# Copy the new render over the old clip
copy media\videos\scene_03\1080p60\Scene03.mp4 ..\scenes\scene_03.mp4
```

---

## Common Issues and Fixes

### "manim is not recognized"
Python Scripts folder is not in PATH. Find it:
```
python -m site --user-scripts
```
Add that path to System PATH (same process as FFmpeg Step 2).

### MiKTeX popup on first render
Expected behaviour. Click Install and wait. Only happens once per package.

### "xelatex not found"
MiKTeX is not in PATH. Open MiKTeX Console, check the install path,
and add the bin\x64\ folder to System PATH.
Usually: C:\Users\YourName\AppData\Local\Programs\MiKTeX\miktex\bin\x64\

### FFmpeg not recognized
C:\ffmpeg\bin is not in PATH. Recheck Step 2 and restart the terminal.

### UnicodeEncodeError or garbled output
Run with:
```
set PYTHONUTF8=1 && python main.py --input paper.pdf
```

### 4K renders are slow
Normal. Manim 4K is CPU-heavy. Use --resolution 720p for testing,
switch to 1080p or 4K for final renders.

### Avoid spaces in folder path
Use C:\projects\procex\ not C:\My Projects\procex\

---

## Full PowerShell Example

```powershell
# Navigate to procex
cd C:\projects\procex

# Set API keys for this session
$env:ANTHROPIC_API_KEY  = "sk-ant-..."
$env:GEMINI_API_KEY     = "AIza..."
$env:ELEVENLABS_API_KEY = "..."

# Run — 7-minute ML paper
python main.py `
    --input "C:\Users\Emmanuel\Downloads\attention_is_all_you_need.pdf" `
    --topic "The Attention Mechanism" `
    --minutes 7 `
    --resolution 1080p

# Output: output\videos\the_attention_mechanism.mp4
```

---

## Cost Estimate Per Video

| Component | Cost |
|---|---|
| ScriptWriter (LLM) | ~$0.10 |
| VisualDirector (LLM) | ~$0.15 |
| ManimCoder 15 scenes | ~$0.40 |
| ElevenLabs TTS | ~$0.15 |
| NanoBanana Pro (anatomy images) | ~$0.20 |
| Total per 10-min video | ~$1.00-$1.50 |

Manim rendering is free (local CPU). No GPU required.
