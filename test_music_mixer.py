"""
test_music_mixer.py — standalone CLI test for ProcEx background music mixing.

Usage:
    cd procex/procex
    python test_music_mixer.py                  # auto-finds TTS in output/audio/
    python test_music_mixer.py --tts output/audio/collatz_conjectureproblem.mp3
    python test_music_mixer.py --play           # open result in media player
"""
import argparse, os, subprocess, sys, random, shutil, tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--tts",      default=None)
parser.add_argument("--songs",    default=None)
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--play",     action="store_true")
args = parser.parse_args()

def probe(path):
    try:
        r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                            "-of","default=noprint_wrappers=1:nokey=1",path],
                           capture_output=True,text=True,timeout=15)
        return float(r.stdout.strip())
    except: return 0.0

SEP = "━"*54
print(f"\n{SEP}\n  ProcEx Music Mixer Test\n{SEP}\n")

# 1. ffmpeg
print("[1] Checking ffmpeg/ffprobe...")
for tool in ["ffmpeg","ffprobe"]:
    r = subprocess.run([tool,"-version"],capture_output=True,timeout=10)
    if r.returncode!=0: print(f"  ✗ {tool} not found"); sys.exit(1)
    print(f"  ✓ {tool}: {(r.stdout or r.stderr or b'').decode().split(chr(10))[0][:70]}")

# 2. songs/
print("\n[2] Locating songs/...")
script_dir = os.path.dirname(os.path.abspath(__file__))
songs_root = args.songs or next((c for c in [
    os.path.join(script_dir,"songs"),
    os.path.join(os.path.dirname(script_dir),"songs"),
] if os.path.isdir(c)), None)
if not songs_root: print("  ✗ songs/ not found. Pass --songs"); sys.exit(1)
print(f"  ✓ {songs_root}")

acapella_dir = os.path.join(songs_root,"acapella_tracks")
action_dir   = os.path.join(songs_root,"action_tracks")
acapella_tracks, action_tracks = [], []
for label,d,store in [("acapella_tracks",acapella_dir,acapella_tracks),
                      ("action_tracks",  action_dir,  action_tracks)]:
    if not os.path.isdir(d): print(f"  ✗ {label}/ missing"); sys.exit(1)
    files = sorted(f for f in os.listdir(d) if f.endswith((".mp3",".wav")))
    store.extend(os.path.join(d,f) for f in files)
    print(f"  ✓ {label}/: {len(files)} tracks")

# 3. TTS audio — auto-discover real files
print("\n[3] Locating TTS audio...")
tts_path = args.tts
using_real = False
if not tts_path:
    # IMPORTANT: combined output files (e.g. collatz_topic.mp3) are OVERWRITTEN
    # by music_mixer with music already baked in. Using them adds a second music
    # layer. Prefer per-scene chunks (*_scene_XX.mp3) which are always unmixed.
    audio_dir = os.path.join(script_dir,"output","audio")
    if os.path.isdir(audio_dir):
        all_mp3 = sorted(
            [os.path.join(audio_dir,f) for f in os.listdir(audio_dir)
             if f.endswith(".mp3") and not f.startswith("_")],
            key=os.path.getmtime, reverse=True)
        scene_files = [f for f in all_mp3 if "_scene_" in os.path.basename(f)]
        combined    = [f for f in all_mp3 if os.path.getsize(f) > 500_000]
        # Prefer per-scene (unmixed); fall back to combined with warning
        tts_path = scene_files[0] if scene_files else (combined[0] if combined else None)

tmp_tone = None
if tts_path and os.path.exists(tts_path):
    tts_dur = probe(tts_path)
    using_real = True
    is_scene = "_scene_" in os.path.basename(tts_path)
    print(f"  ✓ Real TTS: {os.path.basename(tts_path)}  {tts_dur:.1f}s  {os.path.getsize(tts_path)//1024}KB")
    if is_scene:
        print(f"  ✓ Per-scene file — guaranteed unmixed raw TTS")
    elif os.path.getsize(tts_path) > 500_000:
        print(f"  ⚠ WARNING: this combined file may already have music mixed in!")
        print(f"    Using it here will add a second simultaneous music layer.")
        print(f"    Pass --tts output/audio/*_scene_01.mp3 for clean test.")
else:
    tmp_tone = tempfile.NamedTemporaryFile(suffix=".mp3",delete=False)
    tmp_tone.close(); tts_path = tmp_tone.name
    subprocess.run(["ffmpeg","-y","-f","lavfi",
                   "-i",f"sine=frequency=440:duration={args.duration}",
                   "-c:a","libmp3lame","-q:a","5",tts_path],
                  capture_output=True,timeout=30)
    tts_dur = probe(tts_path)
    print(f"  ⚠ No real TTS — using {tts_dur:.1f}s test tone")
    print(f"    Run with real audio: python test_music_mixer.py "
          f"--tts output/audio/<topic>.mp3")

# 4. Mix
print("\n[4] Testing amix for each category...")
MIX_DUR = min(tts_dur, 110.0)
results = []

with tempfile.TemporaryDirectory() as tmpdir:
    for category, tracks in [("acapella",acapella_tracks),("action",action_tracks)]:
        if not tracks: print(f"  ✗ No {category} tracks"); continue
        track    = random.choice(tracks)
        t_dur    = probe(track)
        out_path = os.path.join(tmpdir,f"mixed_{category}.mp3")
        print(f"\n  [{category}] {os.path.basename(track)} ({t_dur:.1f}s)  mix={MIX_DUR:.1f}s")

        # slice TTS
        tts_seg = os.path.join(tmpdir,f"seg_{category}.mp3")
        r = subprocess.run(["ffmpeg","-y","-ss","0","-i",tts_path,
                            "-t",f"{MIX_DUR:.3f}",
                            "-c:a","libmp3lame","-q:a","2",tts_seg],
                           capture_output=True,text=True,timeout=60)
        if r.returncode!=0:
            print(f"  ✗ TTS slice failed:\n{r.stderr[-200:]}"); continue

        usable     = max(t_dur - 10.0, 1.0)
        trim_start = max(5.0, 5.0 + usable/2 - MIX_DUR/2) if usable>MIX_DUR else 5.0

        r = subprocess.run([
            "ffmpeg","-y",
            "-i", tts_seg,
            "-ss", f"{trim_start:.2f}", "-i", track,
            "-filter_complex",
            f"[0:a][1:a]amix=inputs=2:weights=0.85 0.15:normalize=0:duration=first:dropout_transition=2[out]",
            "-map","[out]","-c:a","libmp3lame","-q:a","2",
            "-t",f"{MIX_DUR:.3f}", out_path,
        ], capture_output=True, text=True, timeout=120)

        if r.returncode!=0:
            print(f"  ✗ amix FAILED:")
            for line in r.stderr.strip().split("\n")[-10:]: print(f"    {line}")
            results.append((category,False,None)); continue

        out_dur  = probe(out_path)
        out_size = os.path.getsize(out_path)
        print(f"  ✓ Output: {out_dur:.1f}s  {out_size//1024}KB")

        # volume check
        r2 = subprocess.run(["ffmpeg","-i",out_path,"-af","volumedetect",
                             "-vn","-sn","-dn","-f","null","-"],
                            capture_output=True,text=True,timeout=30)
        for line in r2.stderr.split("\n"):
            if "mean_volume" in line or "max_volume" in line:
                print(f"    {line.strip()}")
                if "mean_volume" in line:
                    try:
                        db = float(line.split(":")[1].replace("dBFS","").strip())
                        if db < -60: print(f"  ⚠ WARNING: mean={db:.1f}dBFS — very quiet!")
                        else:        print(f"  ✓ mean={db:.1f}dBFS — audible")
                    except: pass

        results.append((category, out_size>1024, out_path))

    # save best result
    print("\n[5] Saving output...")
    for category,ok,out_path in results:
        if ok and out_path and os.path.exists(out_path):
            dest = os.path.join(script_dir,"test_music_mixed.mp3")
            shutil.copy2(out_path, dest)
            abs_dest = os.path.abspath(dest)
            print(f"  ✓ Saved {category} mix → {abs_dest}")
            if args.play:
                if sys.platform=="win32": os.startfile(abs_dest)
                elif sys.platform=="darwin": subprocess.run(["open",abs_dest])
                else: subprocess.run(["xdg-open",abs_dest])
            else:
                print(f"\n  Listen:  ffplay \"{abs_dest}\"")
            break

passed = sum(1 for _,ok,_ in results if ok)
print(f"\n{SEP}")
print(f"  {passed}/{len(results)} categories passed")
print(f"  TTS source: {'real file — '+os.path.basename(tts_path) if using_real else 'test tone'}")
print(f"{SEP}\n")

if tmp_tone and os.path.exists(tts_path):
    try: os.unlink(tts_path)
    except: pass