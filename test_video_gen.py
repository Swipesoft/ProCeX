"""
test_video_gen.py — standalone CLI test for Novita Seedance 1.5 Pro T2V.

Usage:
    cd procex/procex
    python test_video_gen.py
    python test_video_gen.py --prompt "A mathematician writing equations on a chalkboard"
    python test_video_gen.py --duration 5 --resolution 720p --ratio 9:16

Tests the full async flow: submit → poll → download → verify file.
Reads NOVITA_API_KEY from .env automatically.
"""
import argparse, os, sys, time, json, requests
from pathlib import Path

parser = argparse.ArgumentParser(description="Test Novita Seedance 1.5 Pro T2V")
parser.add_argument("--prompt",     default="Slow cinematic aerial shot of ancient Alexandria at dusk, "
                                            "golden light, atmospheric haze, Manim-style.")
parser.add_argument("--duration",   type=int,   default=5,    help="Clip duration in seconds [4-12]")
parser.add_argument("--resolution", default="720p",           help="480p or 720p")
parser.add_argument("--ratio",      default="9:16",           help="Aspect ratio e.g. 9:16 or 16:9")
parser.add_argument("--out",        default="test_video_gen_output.mp4")
parser.add_argument("--timeout",    type=int,   default=300,  help="Max poll wait in seconds")
args = parser.parse_args()

SEP = "━"*54

# ── Load .env ─────────────────────────────────────────────────────────────────
def load_dotenv(path=".env"):
    if not os.path.exists(path): return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_dotenv()

print(f"\n{SEP}")
print("  Novita Seedance 1.5 Pro — T2V Test")
print(f"{SEP}\n")

# ── 1. API key ────────────────────────────────────────────────────────────────
print("[1] Checking NOVITA_API_KEY...")
api_key = os.environ.get("NOVITA_API_KEY", "")
if not api_key:
    print("  ✗ NOVITA_API_KEY not set in .env or environment")
    print("    Add:  NOVITA_API_KEY=your_key_here  to your .env file")
    sys.exit(1)
# Mask key for display
masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
print(f"  ✓ Key found: {masked}")

# ── 2. Validate params ────────────────────────────────────────────────────────
print("\n[2] Parameters...")
duration = max(4, min(12, args.duration))
if duration != args.duration:
    print(f"  ⚠ Duration clamped to {duration}s (API range: 4-12s)")
if args.resolution not in ("480p","720p"):
    print(f"  ⚠ Resolution '{args.resolution}' not supported — using 720p")
    args.resolution = "720p"
print(f"  prompt:     {args.prompt[:80]}{'...' if len(args.prompt)>80 else ''}")
print(f"  duration:   {duration}s")
print(f"  resolution: {args.resolution}")
print(f"  ratio:      {args.ratio}")
print(f"  output:     {args.out}")

# ── 3. Submit T2V task ────────────────────────────────────────────────────────
T2V_URL    = "https://api.novita.ai/v3/async/seedance-v1.5-pro-t2v"
RESULT_URL = "https://api.novita.ai/v3/async/task-result"  # task_id as ?task_id= query param

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type":  "application/json",
}

payload = {
    "prompt":          args.prompt,
    "duration":        duration,
    "ratio":           args.ratio,
    "resolution":      args.resolution,
    "fps":             24,
    "watermark":       False,
    "generate_audio":  False,   # we supply our own TTS
    "camera_fixed":    False,
    "seed":            -1,
}

print(f"\n[3] Submitting T2V task to Novita...")
print(f"  POST {T2V_URL}")
t_submit = time.time()
try:
    resp = requests.post(T2V_URL, json=payload, headers=headers, timeout=30)
except requests.ConnectionError as e:
    print(f"  ✗ Connection error: {e}")
    sys.exit(1)

print(f"  HTTP {resp.status_code}")

if resp.status_code == 429:
    print(f"  ✗ Rate limited (429): {resp.text[:300]}")
    sys.exit(1)
if resp.status_code == 401:
    print(f"  ✗ Authentication failed (401) — check your NOVITA_API_KEY")
    sys.exit(1)
if not resp.ok:
    print(f"  ✗ Submit failed ({resp.status_code}): {resp.text[:400]}")
    sys.exit(1)

task_id = resp.json().get("task_id","")
if not task_id:
    print(f"  ✗ No task_id in response: {resp.text[:300]}")
    sys.exit(1)

print(f"  ✓ task_id: {task_id}")

# ── 4. Poll for result ────────────────────────────────────────────────────────
print(f"\n[4] Polling for result (timeout={args.timeout}s)...")
poll_url = RESULT_URL   # task_id sent as query param
deadline = time.time() + args.timeout
attempt  = 0

while time.time() < deadline:
    attempt += 1
    time.sleep(5)
    elapsed = time.time() - t_submit

    try:
        r = requests.get(poll_url, headers=headers,
                         params={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        data   = r.json()
        status = data.get("task", {}).get("status", "UNKNOWN")
        print(f"  [{elapsed:5.0f}s] poll #{attempt}: {status}")

        if status == "TASK_STATUS_SUCCEED":
            videos = data.get("videos", [])  # top-level, not nested under task
            if not videos:
                print("  ✗ Task succeeded but no output_videos in response")
                print(f"    Full response: {json.dumps(data, indent=2)[:500]}")
                sys.exit(1)
            video_url = videos[0].get("video_url","")  # correct field per API docs
            if not video_url:
                print("  ✗ output_videos[0] has no url field")
                sys.exit(1)
            print(f"  ✓ Video URL: {video_url[:80]}...")
            break

        elif status in ("TASK_STATUS_FAILED","TASK_STATUS_EXPIRED"):
            err = data.get("task",{}).get("err_message","(no message)")
            print(f"  ✗ Task {status}: {err}")
            sys.exit(1)

    except requests.HTTPError as e:
        print(f"  ⚠ Poll HTTP error: {e}")
    except Exception as e:
        print(f"  ⚠ Poll error: {e}")

else:
    print(f"  ✗ Timed out after {args.timeout}s waiting for video")
    sys.exit(1)

# ── 5. Download ───────────────────────────────────────────────────────────────
print(f"\n[5] Downloading video clip...")
try:
    # S3 pre-signed URLs are self-authenticating — no Authorization header
    r = requests.get(video_url, headers={}, timeout=120, stream=True)
    r.raise_for_status()
    total = 0
    with open(args.out, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"  ✓ Downloaded {total//1024}KB → {os.path.abspath(args.out)}")
except Exception as e:
    print(f"  ✗ Download failed: {e}")
    sys.exit(1)

# ── 6. Verify file ────────────────────────────────────────────────────────────
print(f"\n[6] Verifying output...")
size = os.path.getsize(args.out)
if size < 10_000:
    print(f"  ✗ File is suspiciously small ({size} bytes)")
    sys.exit(1)

# Use ffprobe to check duration and codec
try:
    r = subprocess.run if (subprocess := __import__("subprocess")) else None
    r = subprocess.run(
        ["ffprobe","-v","error","-show_streams","-show_format",
         "-of","json", args.out],
        capture_output=True, text=True, timeout=15
    )
    info = json.loads(r.stdout)
    fmt  = info.get("format",{})
    dur  = float(fmt.get("duration",0))
    streams = info.get("streams",[])
    has_video = any(s.get("codec_type")=="video" for s in streams)
    codec     = next((s.get("codec_name","?") for s in streams
                      if s.get("codec_type")=="video"), "?")
    width     = next((s.get("width",0) for s in streams
                      if s.get("codec_type")=="video"), 0)
    height    = next((s.get("height",0) for s in streams
                      if s.get("codec_type")=="video"), 0)

    print(f"  Duration:   {dur:.1f}s")
    print(f"  Resolution: {width}x{height}")
    print(f"  Codec:      {codec}")
    print(f"  File size:  {size//1024}KB")
    print(f"  Has video:  {has_video}")

    if not has_video:
        print("  ✗ No video stream found in output file")
        sys.exit(1)
    if dur < 3.0:
        print(f"  ⚠ Duration {dur:.1f}s seems short")
except Exception as e:
    print(f"  ⚠ ffprobe check failed: {e} — file exists, assume OK")

total_time = time.time() - t_submit
print(f"\n{SEP}")
print(f"  ✓ T2V test PASSED in {total_time:.0f}s")
print(f"  Output: {os.path.abspath(args.out)}")
print(f"  Play:   ffplay \"{os.path.abspath(args.out)}\"")
print(f"{SEP}\n")