"""
utils/audio_humanizer.py

Post-processes TTS-generated MP3 audio to sound more natural.
Applied after TTSAgent writes the combined audio file, before AssemblerAgent reads it.

Techniques applied (in order):
  1. Warmth EQ     — gentle +2dB boost at 250Hz to fill out thin TTS low-mids
  2. De-essing     — gentle attenuation at 7kHz to soften synthetic sibilance
  3. Micro pitch   — very subtle pitch drift (±0.12 semitones) via resampling
  4. Volume breath — slow random-walk volume modulation (not a sine wave)
  5. Short reverb  — small room impulse to place the voice in a space
  6. Normalise     — peak-normalise to -1dBFS before export

Intentionally NOT included:
  - Room-tone noise (Aoede already has character; noise makes it sound like bad phone)
  - Aggressive pitch shifts (>0.15 semitones sounds wobbly and draws attention)

Dependencies: pydub, numpy, soundfile (all lightweight, no librosa needed)
ffmpeg must be installed system-wide for pydub MP3 decode.
"""
from __future__ import annotations
import numpy as np
import os
import tempfile


def humanize(
    input_path:  str,
    output_path: str | None = None,
    *,
    warmth_db:       float = 1.0,     # low-mid boost in dB — 1dB is felt not heard
    deness_db:       float = 0.8,     # high-freq cut in dB — subtle sibilance taming
    pitch_range:     float = 0.06,    # max semitone drift — halved; 0.06 is truly inaudible
    volume_depth:    float = 0.012,   # max ±volume modulation — reduced to ±1.2%
    reverb_mix:      float = 0.03,    # wet/dry reverb mix — 3% barely adds air
    verbose:         bool  = False,
) -> str:
    """
    Apply humanisation effects to a TTS audio file.

    Parameters
    ----------
    input_path  : Path to the source MP3 or WAV file.
    output_path : Where to write the result. Defaults to overwriting input_path.
    warmth_db   : Low-mid shelf boost. 1.0dB is felt not heard — adds body without
                  sounding processed. 2.0dB was too obvious when stacked with other effects.
    deness_db   : High-freq shelf cut. 0.8dB tames Aoede sibilance imperceptibly.
    pitch_range : Max pitch drift in semitones. 0.06 is genuinely inaudible as pitch
                  shift but removes the robotic constant-pitch feel.
                  0.12 with short segments created audible flutter — halved.
    volume_depth: Amplitude of the slow volume breathing. 0.012 = ±1.2%.
    reverb_mix  : Wet fraction for short room reverb. 0.03 = 3% — adds air only.
                  0.06 compounded with EQ and pitch to sound obviously processed.
    verbose     : Log processing steps.

    Returns
    -------
    Path to the processed file.
    """
    from pydub import AudioSegment
    import soundfile as sf

    def log(msg):
        if verbose:
            print(f"[AudioHumanizer] {msg}")

    if output_path is None:
        output_path = input_path

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log(f"Loading: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".mp3":
        seg = AudioSegment.from_mp3(input_path)
    else:
        seg = AudioSegment.from_file(input_path)

    sr       = seg.frame_rate
    channels = seg.channels

    # Work in float32 mono internally, re-stereo at end if needed
    raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if channels == 2:
        raw = raw.reshape(-1, 2)
        mono = raw.mean(axis=1)
    else:
        mono = raw

    # Normalise to [-1, +1] working range
    peak = np.max(np.abs(mono)) or 1.0
    mono = mono / peak
    n    = len(mono)

    # ── 2. Warmth EQ (low-mid shelf boost at ~250Hz) ──────────────────────────
    if warmth_db > 0:
        log(f"Warmth EQ +{warmth_db}dB @ 250Hz")
        mono = _shelf_filter(mono, sr, cutoff=250, gain_db=warmth_db, shelf="low")

    # ── 3. De-essing (high shelf cut at ~7kHz) ────────────────────────────────
    if deness_db > 0:
        log(f"De-essing -{deness_db}dB @ 7kHz")
        mono = _shelf_filter(mono, sr, cutoff=7000, gain_db=-deness_db, shelf="high")

    # ── 4. Micro pitch drift ──────────────────────────────────────────────────
    if pitch_range > 0:
        log(f"Pitch drift ±{pitch_range} semitones")
        mono = _micro_pitch(mono, sr, max_semitones=pitch_range)

    # ── 5. Volume breathing (slow random walk, NOT a sine wave) ───────────────
    if volume_depth > 0:
        log(f"Volume breath ±{volume_depth*100:.1f}%")
        mono = _volume_breath(mono, volume_depth=volume_depth)

    # ── 6. Short room reverb ──────────────────────────────────────────────────
    if reverb_mix > 0:
        log(f"Room reverb {reverb_mix*100:.0f}% wet")
        mono = _simple_reverb(mono, sr, mix=reverb_mix)

    # ── 7. Peak normalise to -1 dBFS ─────────────────────────────────────────
    target_peak = 10 ** (-1.0 / 20)   # -1 dBFS
    current_peak = np.max(np.abs(mono)) or 1.0
    mono = mono * (target_peak / current_peak)
    log(f"Normalised to -1 dBFS (gain={target_peak/current_peak:.3f})")

    # ── 8. Export ─────────────────────────────────────────────────────────────
    # Write to a temp WAV first, then convert to MP3 via pydub if needed
    out_ext = os.path.splitext(output_path)[1].lower()

    if out_ext in (".mp3",):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, mono, sr, subtype="PCM_16")
        out_seg = AudioSegment.from_wav(tmp_path)
        out_seg.export(output_path, format="mp3", bitrate="192k")
        os.unlink(tmp_path)
    else:
        sf.write(output_path, mono, sr, subtype="PCM_16")

    log(f"Written: {output_path}")
    return output_path


# ── DSP helpers ───────────────────────────────────────────────────────────────

def _shelf_filter(
    x: np.ndarray,
    sr: int,
    cutoff: float,
    gain_db: float,
    shelf: str,     # "low" | "high"
) -> np.ndarray:
    """
    Simple first-order shelf filter via frequency-domain multiplication.
    Gentle enough that ±3dB is inaudible as EQ but effective.
    """
    n    = len(x)
    X    = np.fft.rfft(x)
    freq = np.fft.rfftfreq(n, d=1.0/sr)

    gain_linear = 10 ** (gain_db / 20)
    envelope    = np.ones(len(freq))

    if shelf == "low":
        # Ramp from 1.0 → gain_linear below cutoff
        envelope = np.where(
            freq < cutoff,
            gain_linear,
            np.where(
                freq < cutoff * 2,
                1.0 + (gain_linear - 1.0) * (1.0 - (freq - cutoff) / cutoff),
                1.0
            )
        )
    else:  # high
        envelope = np.where(
            freq > cutoff,
            gain_linear,
            np.where(
                freq > cutoff / 2,
                1.0 + (gain_linear - 1.0) * ((freq - cutoff/2) / (cutoff/2)),
                1.0
            )
        )

    return np.fft.irfft(X * envelope, n=n).astype(np.float32)


def _micro_pitch(
    x: np.ndarray,
    sr: int,
    max_semitones: float = 0.12,
    segment_ms: int = 1500,  # longer segments = smoother pitch drift, no flutter
) -> np.ndarray:
    """
    Divide audio into overlapping segments, resample each by a slightly
    different ratio to create a slow, inaudible pitch drift.
    Uses linear resampling (no librosa needed) — at ±0.12 semitones
    the quality loss from linear interp is completely inaudible.
    """
    seg_len   = int(sr * segment_ms / 1000)
    hop_len   = seg_len // 2
    out       = np.zeros(len(x) + seg_len, dtype=np.float32)
    window    = np.hanning(seg_len).astype(np.float32)
    rng       = np.random.default_rng(seed=42)

    # Smooth random pitch curve (random walk)
    n_segs = (len(x) // hop_len) + 2
    walk   = np.cumsum(rng.uniform(-0.3, 0.3, n_segs))
    # Scale walk to ±max_semitones
    walk   = walk - walk.mean()
    mx     = np.max(np.abs(walk)) or 1.0
    walk   = walk / mx * max_semitones

    pos = 0
    for i in range(n_segs):
        start = i * hop_len
        if start >= len(x):
            break
        chunk = x[start:start + seg_len]
        if len(chunk) < 4:
            break
        chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        # Resample ratio for this semitone offset
        semitones  = walk[i]
        ratio      = 2 ** (semitones / 12)
        new_len    = int(len(chunk) * ratio)
        if new_len < 2:
            continue

        old_idx = np.linspace(0, len(chunk) - 1, new_len)
        resampled = np.interp(old_idx, np.arange(len(chunk)), chunk)

        # Overlap-add back to seg_len
        if len(resampled) >= seg_len:
            resampled = resampled[:seg_len]
        else:
            resampled = np.pad(resampled, (0, seg_len - len(resampled)))

        resampled *= window
        out[start:start + seg_len] += resampled

    result = out[:len(x)]

    # RMS normalisation — overlap-add with Hanning window at longer segment
    # sizes causes energy loss (window doesn't sum to 1.0 perfectly at edges).
    # Restore output RMS to match input RMS so pitch drift doesn't change volume.
    rms_in  = float(np.sqrt(np.mean(x ** 2))) if len(x) > 0 else 1.0
    rms_out = float(np.sqrt(np.mean(result ** 2)))
    if rms_out > 1e-8:
        result = result * (rms_in / rms_out)

    return result.astype(np.float32)


def _volume_breath(
    x: np.ndarray,
    volume_depth: float = 0.018,
    breath_hz: float = 0.07,    # ~1 breath cycle per 14 seconds
) -> np.ndarray:
    """
    Slow random-walk volume modulation that mimics natural breathing rhythm.
    Uses Perlin-like smooth noise rather than a sine wave.
    """
    n   = len(x)
    rng = np.random.default_rng(seed=7)

    # Generate low-frequency random walk then smooth it heavily
    n_control = max(int(n * breath_hz / 100), 10)
    control   = np.cumsum(rng.uniform(-1, 1, n_control))
    control   = control - control.mean()
    mx        = np.max(np.abs(control)) or 1.0
    control   = control / mx * volume_depth

    # Upsample control points to audio length via linear interp
    ctrl_idx  = np.linspace(0, n - 1, n_control)
    audio_idx = np.arange(n)
    envelope  = np.interp(audio_idx, ctrl_idx, control)

    return (x * (1.0 + envelope)).astype(np.float32)


def _simple_reverb(
    x: np.ndarray,
    sr: int,
    mix: float = 0.06,
    pre_delay_ms: float = 4.0,
    decay: float = 0.12,   # was 0.28 — metallic tail reduced
    n_taps: int = 3,        # was 6 — fewer taps, cleaner air
) -> np.ndarray:
    """
    Lightweight comb-filter reverb. Simulates a small room (broadcast booth)
    with a short pre-delay and fast decay. No convolution IR needed.
    """
    pre_samples = int(sr * pre_delay_ms / 1000)
    wet         = np.zeros(len(x) + pre_samples * n_taps, dtype=np.float32)

    # Comb filter: multiple delayed, decaying copies
    rng = np.random.default_rng(seed=13)
    tap_offsets = sorted(rng.integers(pre_samples, pre_samples * n_taps, n_taps))

    for i, offset in enumerate(tap_offsets):
        gain = decay ** (i + 1)
        if offset < len(wet) - len(x):
            wet[offset:offset + len(x)] += x * gain

    wet = wet[:len(x)]

    # Mix dry + wet
    result = x * (1.0 - mix) + wet * mix
    return result.astype(np.float32)