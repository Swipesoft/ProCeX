"""
utils/music_mixer.py

Mixes background instrumental music into the assembled TTS audio at a
95% (voice) / 5% (music) volume ratio.

Track selection rules:
  - NARRATOR / STORY scenes  → acapella_tracks (solemn, no percussion)
  - TECHNICAL / VOICE scenes → action_tracks   (allegro, with percussion)
  - Mixed groups             → action_tracks   (default to energetic)

Track structure:
  procex/songs/acapella_tracks/track_0.mp3 ... track_N.mp3
  procex/songs/action_tracks/track_0.mp3  ... track_N.mp3

Each track: 120s ≤ len ≤ 180s. Intro and outro have 4-5s silence.
We trim 5s from both ends before use to avoid dead air at transitions.

Algorithm:
  1. Group consecutive scenes into sub-theme groups where:
       - Total group duration < 120s (fits in one track)
       - All scenes share the same music category (acapella vs action)
  2. For each group, pick a random track from the appropriate folder
  3. Trim 5s from both start and end of the selected track
  4. If track longer than group: cut from the centre (preserves the
     main body of the music, avoids both the intro and the outro fade)
  5. Mix per-group audio: ffmpeg amix at 0.95/0.05 volume split
  6. Concatenate all mixed group segments back to a single final audio

Wire-in point: called from AssemblerAgent after per-scene audio is
assembled but before the humanizer runs.
"""
from __future__ import annotations

import os
import random
import subprocess
import tempfile
from typing import Optional

# Volume ratio: TTS voice dominates, music is subtle background texture
# These are RELATIVE weights passed to amix — treated as a ratio (not gain).
# With normalize=1, amix auto-levels the output to prevent clipping while
# keeping the ratio. 85:15 gives clearly audible background music.
VOICE_VOL  = 0.95
MUSIC_VOL  = 0.05

# Trim from each end of every track (seconds) to avoid intro/outro silence
TRACK_TRIM = 5.0

# Max group duration before we start a new music segment
MAX_GROUP_DUR = 115.0   # slightly under 120s so we never exceed a track

# Paragraph types that map to each music category
# Paragraph types → music category mapping
# NARRATOR and empty (non-documentary) → acapella by default for SHORT scenes
# All others → action (energetic, allegro)
# For non-documentary runs where all paragraph_types are empty, we use
# narration_text heuristics to decide — history/drama/science = action,
# very short bridge scenes = acapella
ACAPELLA_TYPES = {"NARRATOR"}      # only pure narrator bridges get acapella
ACTION_TYPES   = {"TECHNICAL", "STORY", "VOICE"}


def _songs_root(project_root: str) -> str:
    """
    Return the songs directory.
    Checks two locations in priority order:
      1. procex/songs/  (sibling of output_root — works for any output_root name)
      2. songs/         (relative to cwd — fallback)

    output_root is typically "output" (relative) or an absolute path.
    We resolve it to absolute first so dirname() works correctly.
    """
    abs_root = os.path.abspath(project_root)
    # Candidate 1: sibling of output_root
    sibling = os.path.join(os.path.dirname(abs_root), "songs")
    if os.path.isdir(sibling):
        return sibling
    # Candidate 2: relative to cwd
    cwd_songs = os.path.join(os.getcwd(), "songs")
    if os.path.isdir(cwd_songs):
        return cwd_songs
    # Return the sibling path regardless (caller checks isdir)
    return sibling


def _pick_track(songs_root: str, category: str, exclude: set) -> Optional[str]:
    """
    Randomly pick a track from the appropriate folder.
    exclude: set of previously used track paths to avoid immediate repeats.
    Returns None if folder is empty or missing.
    """
    folder = os.path.join(songs_root, f"{category}_tracks")
    if not os.path.isdir(folder):
        return None
    tracks = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".mp3") or f.endswith(".wav")
    )
    if not tracks:
        return None
    available = [t for t in tracks if t not in exclude] or tracks
    return random.choice(available)


def _probe_duration(path: str) -> float:
    """ffprobe audio duration in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _music_category(scenes: list) -> str:
    """
    Determine music category for a group of scenes.

    Documentary scenes:
      TECHNICAL / STORY / VOICE → action_tracks
      NARRATOR only             → acapella_tracks

    Non-documentary scenes (paragraph_type=""):
      Default to action_tracks — most topics (history, science, drama)
      benefit from energetic backing.
      Exception: groups of very short scenes (<= 15s total) get acapella
      (these are typically bridge/intro moments).
    """
    has_action_type = False
    all_empty       = True
    total_dur       = 0.0

    for s in scenes:
        pt = getattr(s, "paragraph_type", "") or ""
        if pt:
            all_empty = False
        if pt in ACTION_TYPES:
            has_action_type = True
        total_dur += getattr(s, "tts_duration", 0.0) or getattr(s, "duration_seconds", 0.0)

    if has_action_type:
        return "action"

    # Non-documentary: default action unless very short bridge group
    if all_empty:
        return "acapella" if total_dur <= 15.0 else "action"

    return "acapella"


def _mix_group(
    tts_segment_path: str,
    track_path:       str,
    group_duration:   float,
    out_path:         str,
) -> bool:
    """
    Mix one TTS segment with a music track using ffmpeg amix.
    Trims 5s from each end of the track, then cuts to group_duration
    from the centre to avoid intro/outro silence and abrupt endings.

    Returns True on success.
    """
    track_dur = _probe_duration(track_path)
    if track_dur <= 0:
        return False

    usable_dur    = max(track_dur - TRACK_TRIM * 2, 1.0)
    need_dur      = group_duration
    trim_start    = TRACK_TRIM

    # If music is longer than group, cut from the centre
    if usable_dur > need_dur:
        centre      = TRACK_TRIM + usable_dur / 2
        trim_start  = max(TRACK_TRIM, centre - need_dur / 2)

    # ffmpeg: mix TTS (0.95) + music slice (0.05)
    # -stream_loop -1 on music input so short tracks loop if needed
    cmd = [
        "ffmpeg", "-y",
        "-i", tts_segment_path,
        "-ss", f"{trim_start:.2f}", "-i", track_path,
        "-filter_complex",
        # normalize=1 makes amix treat weights as relative ratios and auto-levels
        # the output, preventing the combined signal from clipping while still
        # honouring the weight ratio. weights="17 3" = 85%/15% relative split.
        f"[0:a][1:a]amix=inputs=2:weights={int(VOICE_VOL*100)} {int(MUSIC_VOL*100)}:normalize=1:duration=first:dropout_transition=2[out]",
        "-map", "[out]",
        "-c:a", "libmp3lame", "-q:a", "2",   # mp3 codec for mp3 container
        "-t", f"{group_duration:.3f}",
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0 and result.stderr:
            # Surface ffmpeg error so caller can log it
            raise RuntimeError(result.stderr[-400:])
        return result.returncode == 0
    except RuntimeError:
        raise   # propagate so mix_music_into_audio can log it
    except Exception:
        return False


def _concat_segments(segment_paths: list, out_path: str) -> bool:
    """Concatenate audio segments using ffmpeg concat demuxer."""
    if len(segment_paths) == 1:
        import shutil
        shutil.copy2(segment_paths[0], out_path)
        return True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", out_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        os.unlink(list_path)
        return result.returncode == 0
    except Exception:
        try: os.unlink(list_path)
        except Exception: pass
        return False


def mix_music_into_audio(
    final_audio_path: str,
    scenes:           list,
    output_root:      str,
    log_fn=None,
) -> str:
    """
    Main entry point. Takes the assembled TTS audio and mixes background
    music into it based on scene paragraph_type groupings.

    Returns path to the music-mixed audio file (overwrites final_audio_path).
    Falls back gracefully — if anything fails, returns original path unchanged.
    """
    def log(msg):
        if log_fn:
            log_fn(f"[MusicMixer] {msg}")

    songs_root = _songs_root(output_root)

    # Check songs directory exists
    has_acapella = os.path.isdir(os.path.join(songs_root, "acapella_tracks"))
    has_action   = os.path.isdir(os.path.join(songs_root, "action_tracks"))

    if not has_acapella and not has_action:
        log("songs/ directory not found — skipping music mixing")
        return final_audio_path

    log(f"Starting music mix: {len(scenes)} scenes, songs root={songs_root}")

    # ── Group scenes into music segments ─────────────────────────────────────
    groups   = []
    current  = []
    curr_dur = 0.0

    for scene in scenes:
        scene_dur = getattr(scene, "tts_duration", 0.0) or scene.duration_seconds
        cat       = _music_category([scene])

        # Start new group if adding this scene would exceed max OR
        # category changes (acapella/action transition = new group = new track)
        if current:
            curr_cat = _music_category(current)
            if curr_cat != cat or curr_dur + scene_dur > MAX_GROUP_DUR:
                groups.append((current, curr_dur, curr_cat))
                current  = []
                curr_dur = 0.0

        current.append(scene)
        curr_dur += scene_dur

    if current:
        groups.append((current, curr_dur, _music_category(current)))

    log(f"Formed {len(groups)} music groups")

    # ── Extract per-group audio from the assembled TTS file ───────────────────
    # The TTS file is one big concatenated mp3. We slice it by scene offsets.
    audio_dir      = os.path.join(output_root, "audio")
    mixed_segments = []
    used_tracks    = set()
    group_offset   = 0.0

    for g_idx, (group_scenes, group_dur, category) in enumerate(groups):
        # Slice this group's audio from the combined TTS file
        group_raw = os.path.join(audio_dir, f"_group_{g_idx:03d}_raw.mp3")
        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{group_offset:.3f}",
            "-i", final_audio_path,
            "-t",  f"{group_dur:.3f}",
            "-c:a", "libmp3lame", "-q:a", "2",   # re-encode: MP3 copy-slice unreliable
            group_raw,
        ]
        r = subprocess.run(slice_cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            log(f"Group {g_idx}: audio slice failed: {r.stderr[-300:]}")
            mixed_segments.append(None)
            group_offset += group_dur
            continue

        # Pick track
        track = _pick_track(songs_root, category, used_tracks)
        if not track:
            log(f"Group {g_idx}: no {category} tracks found — keeping TTS only")
            mixed_segments.append(group_raw)
            group_offset += group_dur
            continue

        used_tracks.add(track)
        log(f"Group {g_idx}: {category} track={os.path.basename(track)}, dur={group_dur:.1f}s")

        group_mixed = os.path.join(audio_dir, f"_group_{g_idx:03d}_mixed.mp3")
        try:
            ok = _mix_group(group_raw, track, group_dur, group_mixed)
        except RuntimeError as ffmpeg_err:
            log(f"Group {g_idx}: ffmpeg mix error: {str(ffmpeg_err)[:300]}")
            ok = False
        if ok:
            mixed_segments.append(group_mixed)
        else:
            log(f"Group {g_idx}: mix failed — keeping TTS only")
            mixed_segments.append(group_raw)

        group_offset += group_dur

    # ── Concatenate all mixed segments ────────────────────────────────────────
    valid_segments = [s for s in mixed_segments if s and os.path.exists(s)]
    if not valid_segments:
        log("No mixed segments produced — original audio unchanged")
        return final_audio_path

    mixed_final = final_audio_path.replace(".mp3", "_music_mixed.mp3")
    ok = _concat_segments(valid_segments, mixed_final)

    if ok and os.path.exists(mixed_final):
        import shutil
        shutil.move(mixed_final, final_audio_path)
        log(f"Music mix complete → {final_audio_path}")
    else:
        log("Final concat failed — original audio unchanged")

    # Clean up ALL temp group files (raw slices + mixed) regardless of outcome
    # This prevents 0 KB debris files accumulating in the audio directory
    import glob
    audio_dir_cleanup = os.path.join(output_root, "audio")
    for pattern in ["_group_*_raw.mp3", "_group_*_mixed.mp3"]:
        for f in glob.glob(os.path.join(audio_dir_cleanup, pattern)):
            try:
                os.remove(f)
            except Exception:
                pass

    return final_audio_path