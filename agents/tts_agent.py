"""
agents/tts_agent.py

TTS with provider routing — Gemini (default) or OpenAI (fallback).

Gemini TTS (gemini-2.5-flash-preview-tts):
  - High-quality, expressive narration via google-genai SDK
  - Returns raw PCM L16 audio — converted to WAV then MP3 via ffmpeg
  - Voice: Aoede (default). Others: Charon, Fenrir, Kore, Orus, Puck...
  - Uses GEMINI_API_KEY (same key used for VisualDirector / VLMCritic)

OpenAI TTS (tts-1 / tts-1-hd):
  - Kept fully intact as fallback
  - tts-1    = $15/1M chars, tts-1-hd = $30/1M chars
  - Voices: onyx, nova, alloy, echo, fable, shimmer

Neither provider returns word-level timestamps. We estimate them by:
  1. Probing actual audio duration with ffprobe
  2. Distributing word timestamps proportionally by character position
This gives ~+-0.3s accuracy, sufficient for Manim animation sync.

Set tts_provider = "gemini" | "openai" in config.py.
"""
from __future__ import annotations
import os
import re
import struct
import wave
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
import shutil
from state import ProcExState, WordTimestamp
from config import ProcExConfig
from utils.llm_client import LLMClient
from agents.base_agent import BaseAgent


class TTSAgent(BaseAgent):
    name = "TTSAgent"

    def run(self, state: ProcExState) -> ProcExState:
        provider = (self.cfg.tts_provider or "gemini").lower()
        self._log(f"Generating TTS audio via {provider.upper()}...")

        if provider == "gemini":
            return self._run_gemini(state)
        else:
            return self._run_openai(state)

    # ── Gemini TTS ────────────────────────────────────────────────────────────

    def _run_gemini(self, state: ProcExState) -> ProcExState:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai SDK required: pip install google-genai")

        if not self.cfg.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file."
            )

        client    = genai.Client(api_key=self.cfg.gemini_api_key)
        audio_dir = self.cfg.dirs["audio"]
        os.makedirs(audio_dir, exist_ok=True)

        def _make_gemini_config(voice_name: str) -> types.GenerateContentConfig:
            """
            Build a Gemini TTS config for the given voice name.
            Called per-scene so documentary multi-voice scenes each get their
            own config (Aoede for narrator/story/technical, Charon or other
            male voice for [VOICE: X] character paragraphs).
            """
            return types.GenerateContentConfig(
                temperature          = 1.0,
                response_modalities  = ["AUDIO"],
                speech_config        = types.SpeechConfig(
                    voice_config = types.VoiceConfig(
                        prebuilt_voice_config = types.PrebuiltVoiceConfig(
                            voice_name = voice_name,
                        )
                    )
                ),
            )

        def _process_scene_gemini(scene):
            """Call Gemini TTS for one scene, save as MP3. Thread-safe."""
            from utils.context_injection import strip_context_tags as _strip_ctx
            raw_text = _strip_ctx(scene.narration_text.strip())
            if not raw_text:
                return scene, None, 0.0

            # Strip documentary paragraph type tags before TTS so they are
            # never read aloud. Tags look like: [NARRATOR], [STORY],
            # [TECHNICAL], [VOICE: Einstein]. We strip the tag line entirely.
            raw_text = self._strip_paragraph_tags(raw_text)
            if not raw_text:
                return scene, None, 0.0

            # Resolve voice: use scene.tts_voice override if set (documentary
            # multi-voice). "male" is a sentinel that resolves to
            # cfg.gemini_tts_voice_male (Fenrir by default) — allows
            # documentary_parser to set a voice without hardcoding a name.
            raw_voice  = getattr(scene, "tts_voice", "") or ""
            if raw_voice.lower() in ("male", "fenrir", "charon", "orus", "puck"):
                voice_name = getattr(self.cfg, "gemini_tts_voice_male", raw_voice) or raw_voice
            elif raw_voice:
                voice_name = raw_voice
            else:
                voice_name = self.cfg.gemini_tts_voice
            generate_config = _make_gemini_config(voice_name)

            text       = self._clean_narration_for_tts(raw_text)
            chunk_path = os.path.join(
                audio_dir, f"{state.topic_slug}_scene_{scene.id:02d}.mp3"
            )
            self._log(
                f"Scene {scene.id}: {len(text)} chars -> Gemini TTS "                f"({voice_name}" +
                (" [VOICE override]" if voice_name != self.cfg.gemini_tts_voice else "") +
                ")..."
            )

            # ── Retry loop with rate-limit-aware backoff ─────────────────
            MAX_TTS_RETRIES = 5
            last_error      = None

            for attempt in range(1, MAX_TTS_RETRIES + 1):
                try:
                    # Collect all PCM chunks from streaming response
                    pcm_data = b""
                    mime_type = "audio/L16;rate=24000"
                    for chunk in client.models.generate_content_stream(
                        model    = self.cfg.gemini_tts_model,
                        contents = text,
                        config   = generate_config,
                    ):
                        if not chunk.parts:
                            continue
                        part = chunk.parts[0]
                        if part.inline_data and part.inline_data.data:
                            pcm_data  += part.inline_data.data
                            mime_type  = part.inline_data.mime_type or mime_type

                    if not pcm_data:
                        raise RuntimeError("Gemini TTS returned no audio data")

                    # Convert PCM L16 → WAV → MP3 via ffmpeg
                    wav_bytes = self._pcm_to_wav(pcm_data, mime_type)
                    self._wav_to_mp3(wav_bytes, chunk_path)
                    last_error = None
                    break   # success — exit retry loop

                except Exception as e:
                    last_error = e
                    err_str    = str(e)

                    # Parse retry delay from 429 response if present
                    # Gemini embeds: "Please retry in 19.206856505s."
                    import re as _re
                    delay_match = _re.search(r'retry in\s+([\d.]+)s', err_str, _re.IGNORECASE)
                    if delay_match:
                        sleep_secs = float(delay_match.group(1)) + 2.0  # +2s buffer
                    else:
                        sleep_secs = min(5.0 * attempt, 60.0)  # exponential cap at 60s

                    if attempt < MAX_TTS_RETRIES:
                        self._log(
                            f"Scene {scene.id} Gemini TTS attempt {attempt}/{MAX_TTS_RETRIES} "
                            f"failed: {err_str[:120]} — retrying in {sleep_secs:.1f}s"
                        )
                        import time as _time
                        _time.sleep(sleep_secs)
                    else:
                        self._log(
                            f"Scene {scene.id} Gemini TTS FAILED after {MAX_TTS_RETRIES} "
                            f"attempts: {err_str[:120]} — marking scene for purge"
                        )

            if last_error is not None:
                # All retries exhausted — flag scene for purge instead of writing silence
                scene.tts_failed = True
                return scene, None, 0.0

            chunk_duration = self._probe_duration(chunk_path)
            if chunk_duration <= 0:
                chunk_duration = len(text.split()) / 2.8

            scene._tts_relative_ts = self._build_timestamps(text, chunk_duration)
            return scene, chunk_path, chunk_duration

        return self._run_parallel_and_assemble(
            state, _process_scene_gemini, provider="gemini"
        )

    # ── OpenAI TTS ────────────────────────────────────────────────────────────

    def _run_openai(self, state: ProcExState) -> ProcExState:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai SDK required: pip install openai")

        if not self.cfg.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file.\n"
                "Get a key at: https://platform.openai.com/api-keys"
            )

        client    = OpenAI(api_key=self.cfg.openai_api_key)
        audio_dir = self.cfg.dirs["audio"]
        os.makedirs(audio_dir, exist_ok=True)

        def _process_scene_openai(scene):
            """Call OpenAI TTS for one scene, save as MP3. Thread-safe."""
            from utils.context_injection import strip_context_tags as _strip_ctx
            raw_text = _strip_ctx(scene.narration_text.strip())
            if not raw_text:
                return scene, None, 0.0

            text       = self._clean_narration_for_tts(raw_text)
            chunk_path = os.path.join(
                audio_dir, f"{state.topic_slug}_scene_{scene.id:02d}.mp3"
            )
            self._log(f"Scene {scene.id}: {len(text)} chars -> OpenAI TTS ({self.cfg.openai_tts_voice})...")

            try:
                response = client.audio.speech.create(
                    model           = self.cfg.openai_tts_model,
                    voice           = self.cfg.openai_tts_voice,
                    input           = text,
                    response_format = "mp3",
                )
                response.stream_to_file(chunk_path)
            except Exception as e:
                self._log(f"Scene {scene.id} OpenAI TTS failed: {e}")
                chunk_path = self._write_silence(scene.duration_seconds, chunk_path)
                return scene, chunk_path, scene.duration_seconds

            chunk_duration = self._probe_duration(chunk_path)
            if chunk_duration <= 0:
                chunk_duration = len(text.split()) / 2.8

            scene._tts_relative_ts = self._build_timestamps(text, chunk_duration)
            return scene, chunk_path, chunk_duration

        return self._run_parallel_and_assemble(
            state, _process_scene_openai, provider="openai"
        )

    # ── Shared parallel runner + assembler ───────────────────────────────────

    def _run_parallel_and_assemble(
        self, state: ProcExState, process_fn, provider: str
    ) -> ProcExState:
        audio_dir         = self.cfg.dirs["audio"]
        scene_audio_paths = []
        all_timestamps    = []
        time_offset       = 0.0

        max_tts_workers = min(len(state.scenes), self.cfg.tts_workers)
        scene_results   = {}  # scene.id -> (chunk_path, duration)

        with ThreadPoolExecutor(max_workers=max_tts_workers) as pool:
            futures = {pool.submit(process_fn, s): s for s in state.scenes}
            for fut in as_completed(futures):
                s, chunk_path, dur = fut.result()
                if chunk_path:
                    scene_results[s.id] = (chunk_path, dur)
                    self._log(f"Scene {s.id}: {dur:.1f}s TTS done")

        # ── Purge TTS-failed scenes ──────────────────────────────────────────
        # Scenes where all TTS retries exhausted are removed from state.scenes
        # so they never produce silence gaps or misaligned audio in the video.
        failed_ids = {s.id for s in state.scenes if getattr(s, "tts_failed", False)}
        if failed_ids:
            self._log(
                f"Purging {len(failed_ids)} scene(s) with failed TTS: {sorted(failed_ids)}"
            )
            state.scenes = [s for s in state.scenes if s.id not in failed_ids]

        for scene in state.scenes:
            if scene.id not in scene_results:
                continue
            chunk_path, chunk_duration = scene_results[scene.id]

            scene.duration_seconds = chunk_duration
            scene.tts_audio_path   = chunk_path
            scene.tts_duration     = chunk_duration
            # Record the absolute start offset of this scene within the combined
            # audio file. This is critical: when music_mixer redirects all
            # scene.tts_audio_path pointers to the combined music-mixed file,
            # the assembler uses tts_audio_start to seek to the right position.
            # Without this, every scene would play from offset 0 (scene 1 audio).
            scene.tts_audio_start  = round(time_offset, 4)
            scene_audio_paths.append(chunk_path)

            relative_ts = getattr(scene, "_tts_relative_ts", [])
            scene.timestamps = [
                WordTimestamp(
                    word  = t.word,
                    start = round(t.start + time_offset, 4),
                    end   = round(t.end   + time_offset, 4),
                )
                for t in relative_ts
            ]
            all_timestamps.extend(scene.timestamps)
            time_offset += chunk_duration
            self._log(f"Scene {scene.id}: offset applied, cumulative={time_offset:.1f}s")

        final_audio = os.path.join(audio_dir, f"{state.topic_slug}.mp3")

        if not scene_audio_paths:
            raise RuntimeError(
                f"No audio generated via {provider}. "
                f"Check your API key and account status."
            )
        elif len(scene_audio_paths) == 1:
            shutil.copy2(scene_audio_paths[0], final_audio)
        else:
            self._concat_audio(scene_audio_paths, final_audio)

        state.audio_path           = final_audio
        state.all_timestamps       = all_timestamps
        state.total_audio_duration = time_offset

        total_chars = sum(len(s.narration_text) for s in state.scenes)
        if provider == "openai":
            cost_est = total_chars / 1_000_000 * (
                30.0 if "hd" in self.cfg.openai_tts_model else 15.0
            )
            cost_str = f"est. cost ${cost_est:.4f}"
        else:
            cost_str = "Gemini TTS (see Google AI pricing)"

        # ── Audio humanisation ────────────────────────────────────────────────
        # Apply subtle post-processing to make Gemini Aoede sound more natural:
        # warmth EQ, de-essing, micro pitch drift, volume breathing, short reverb.
        # Skipped gracefully if pydub/soundfile are not installed.
        try:
            from utils.audio_humanizer import humanize
            self._log("Applying audio humanisation (warmth EQ, pitch drift, reverb)...")
            humanize(final_audio, final_audio, verbose=False)
            self._log("Audio humanisation done")
        except ImportError:
            self._log("Audio humanisation skipped (pydub/soundfile not installed)")
        except Exception as e:
            self._log(f"Audio humanisation failed (non-critical): {e}")

        # ── Background music mixing ───────────────────────────────────────────
        # Mix acapella/action tracks under the TTS narration.
        # After mixing, update scene.tts_audio_path to point to the combined
        # music-mixed file so AssemblerAgent uses it for final video sync.
        # Without this, assembler reads per-scene files written BEFORE mixing,
        # so music never enters the final video.
        try:
            from utils.music_mixer import mix_music_into_audio
            mix_music_into_audio(
                final_audio_path = final_audio,
                scenes           = state.scenes,
                output_root      = self.cfg.output_root,
                log_fn           = self._log,
            )
            # Redirect all scene audio paths to the music-mixed combined file.
            # tts_audio_start carries the correct seek offset for each scene.
            if os.path.exists(final_audio):
                for scene in state.scenes:
                    scene.tts_audio_path = final_audio
                self._log(
                    f"Scene audio paths updated to music-mixed file: {final_audio}"
                )
        except Exception as e:
            self._log(f"Music mixing skipped (non-critical): {e}")

        self._log(
            f"Audio ready -> {final_audio}\n"
            f"  {time_offset:.1f}s ({time_offset/60:.1f} min) | "
            f"{len(all_timestamps)} word timestamps | "
            f"{cost_str}"
        )
        return state

    # ── Gemini PCM helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_mime_audio_params(mime_type: str) -> dict:
        """Extract bits_per_sample and rate from audio/L16;rate=24000."""
        bits_per_sample = 16
        rate            = 24000
        for part in mime_type.split(";"):
            part = part.strip()
            if part.lower().startswith("rate="):
                try:
                    rate = int(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif part.startswith("audio/L"):
                try:
                    bits_per_sample = int(part.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def _pcm_to_wav(self, pcm_data: bytes, mime_type: str) -> bytes:
        """Wrap raw PCM L16 bytes in a proper WAV container."""
        params          = self._parse_mime_audio_params(mime_type)
        bits_per_sample = params["bits_per_sample"]
        sample_rate     = params["rate"]
        num_channels    = 1
        data_size       = len(pcm_data)
        bytes_per_sample = bits_per_sample // 8
        block_align     = num_channels * bytes_per_sample
        byte_rate       = sample_rate * block_align
        chunk_size      = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE",
            b"fmt ", 16, 1, num_channels,
            sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size,
        )
        return header + pcm_data

    def _wav_to_mp3(self, wav_bytes: bytes, output_path: str) -> None:
        """
        Convert WAV bytes to MP3 file via ffmpeg.
        Tries pipe:0 first (fastest). Falls back to a temp file on Windows
        where pipe:0 can fail with certain ffmpeg builds.
        """
        import tempfile

        def _try_ffmpeg(extra_input_flags: list, pipe_input=None) -> bool:
            cmd = ["ffmpeg", "-y", *extra_input_flags, "-c:a", "libmp3lame", "-q:a", "2", output_path]
            result = subprocess.run(cmd, input=pipe_input, capture_output=True, timeout=120)
            return result.returncode == 0

        # Try 1: pipe:0 (Unix / modern Windows ffmpeg)
        if _try_ffmpeg(["-f", "wav", "-i", "pipe:0"], pipe_input=wav_bytes):
            return

        # Try 2: temp WAV file (Windows pipe fallback)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            if _try_ffmpeg(["-i", tmp_path]):
                return
            raise RuntimeError(
                "WAV to MP3 conversion failed via both pipe and temp file. "
                "Check ffmpeg installation."
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    # ── Shared timestamp builder ──────────────────────────────────────────────

    @staticmethod
    def _build_timestamps(text: str, duration: float) -> list:
        """Proportional word timestamps from character positions."""
        words       = text.split()
        total_chars = len(text)
        char_cursor = 0
        ts          = []
        for word in words:
            sr = char_cursor / max(total_chars, 1)
            er = (char_cursor + len(word)) / max(total_chars, 1)
            ts.append(WordTimestamp(
                word  = word,
                start = round(sr * duration, 4),
                end   = round(er * duration, 4),
            ))
            char_cursor += len(word) + 1
        return ts

    # ── Narration preprocessor ────────────────────────────────────────────────

    @staticmethod
    def _strip_paragraph_tags(text: str) -> str:
        """
        Remove documentary paragraph type tags from narration text so they
        are never read aloud by the TTS engine.

        Strips lines that are purely a tag like:
          [NARRATOR]
          [STORY]
          [TECHNICAL]
          [VOICE: Einstein]
          ▸ NARRATOR
          ▸ VOICE: EINSTEIN

        Also strips inline occurrences at the start of the text.
        Preserves all other content unchanged — non-documentary scenes
        pass through this function with zero modification.
        """
        import re
        # Strip full lines that are just a tag
        text = re.sub(
            r'(?m)^\s*(?:\[(?:NARRATOR|STORY|TECHNICAL|VOICE[^\]]*)]|'
            r'▸\s*(?:NARRATOR|STORY|TECHNICAL|VOICE[^\n]*))\s*$',
            "",
            text,
        )
        # Strip leading inline tag if it starts the text block
        text = re.sub(
            r'^\s*\[(?:NARRATOR|STORY|TECHNICAL|VOICE[^\]]*)]\s*',
            "",
            text,
        )
        # Collapse multiple blank lines left by stripping
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _clean_narration_for_tts(text: str) -> str:
        """
        Convert math/LaTeX notation into natural spoken language before TTS.

        Without this, the voice reads "X transpose T dot W sub Q alpha plus
        y times one over two raised to power one over n" — a wall of symbols.
        With it, equations are either converted to readable phrases or
        collapsed to "this formula" / "this equation".
        """
        # LaTeX display math blocks: $$...$$ -> "the equation"
        text = re.sub(r'\$\$[^$]+\$\$', 'the equation', text)

        # LaTeX block math: \[...\] -> "the equation"
        text = re.sub(r'\\\[[^\]]+\\\]', 'the equation', text)

        # LaTeX inline math: $...$ -> "the expression"
        text = re.sub(r'\$[^$]{1,80}\$', 'the expression', text)

        # LaTeX fractions: \frac{a}{b} -> "a over b"
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 over \2', text)

        # LaTeX sqrt: \sqrt{x} -> "root of x"
        text = re.sub(r'\\sqrt\{([^}]+)\}', r'root of \1', text)

        # Common LaTeX Greek letters and operators
        latex_map = [
            (r'\\alpha',    'alpha'),
            (r'\\beta',     'beta'),
            (r'\\gamma',    'gamma'),
            (r'\\delta',    'delta'),
            (r'\\epsilon',  'epsilon'),
            (r'\\theta',    'theta'),
            (r'\\lambda',   'lambda'),
            (r'\\mu',       'mu'),
            (r'\\sigma',    'sigma'),
            (r'\\tau',      'tau'),
            (r'\\phi',      'phi'),
            (r'\\psi',      'psi'),
            (r'\\omega',    'omega'),
            (r'\\nabla',    'the gradient'),
            (r'\\partial',  'partial'),
            (r'\\sum',      'the sum of'),
            (r'\\prod',     'the product of'),
            (r'\\int',      'the integral of'),
            (r'\\infty',    'infinity'),
            (r'\\cdot',     'times'),
            (r'\\times',    'times'),
            (r'\\leq',      'less than or equal to'),
            (r'\\geq',      'greater than or equal to'),
            (r'\\neq',      'not equal to'),
            (r'\\approx',   'approximately'),
            (r'\\in',       'in'),
            (r'\\mathbb\{R\}', 'the reals'),
            (r'\\text\{([^}]+)\}', r'\1'),
            (r'\\[a-zA-Z]+', ''),          # strip any remaining LaTeX commands
        ]
        for pattern, replacement in latex_map:
            text = re.sub(pattern, replacement, text)

        # Superscripts: x^T -> "x transpose", x^2 -> "x squared", x^n -> "x to the n"
        text = re.sub(r'(\w)\^T\b',             r'\1 transpose',           text)
        text = re.sub(r'(\w)\^2\b',             r'\1 squared',             text)
        text = re.sub(r'(\w)\^3\b',             r'\1 cubed',               text)
        text = re.sub(r'(\w)\^\{([^}]+)\}',     r'\1 to the power of \2',  text)
        text = re.sub(r'(\w)\^(\w+)',            r'\1 to the power of \2',  text)

        # Subscripts: W_Q -> "W sub Q", d_k -> "d sub k"
        text = re.sub(r'([A-Z])_([A-Z])\b',     r'\1 sub \2',  text)
        text = re.sub(r'([a-z])_([a-z])\b',     r'\1 sub \2',  text)
        text = re.sub(r'(\w)_\{([^}]+)\}',      r'\1 sub \2',  text)

        # sqrt(...) without LaTeX
        text = re.sub(r'sqrt\(([^)]+)\)',        r'root of \1', text)

        # Long runs of math-looking tokens (3+ capital variables with operators)
        # e.g. "Q K^T / d_k V" -> "this formula"
        math_chain = re.compile(
            r'(?<![a-z])'
            r'(?:[A-Z]\w*\s*[+\-*/=]\s*){2,}'
            r'[A-Z]\w*'
        )
        text = math_chain.sub('this formula', text)

        # Common fractions
        frac_map = [
            (r'\b1/2\b', 'one half'),
            (r'\b1/3\b', 'one third'),
            (r'\b1/4\b', 'one quarter'),
            (r'\b1/n\b', 'one over n'),
        ]
        for pattern, replacement in frac_map:
            text = re.sub(pattern, replacement, text)

        # Generic fractions: 3/4 -> "3 over 4"
        text = re.sub(r'\b(\d+)/(\d+)\b', r'\1 over \2', text)

        # Clean up leftover braces and excess whitespace
        text = text.replace('{', '').replace('}', '')
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _probe_duration(self, path: str) -> float:
        """Use ffprobe to get exact audio duration in seconds."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _write_silence(self, duration: float, output_path: str) -> str:
        """Generate a silent MP3 placeholder for a failed TTS chunk."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(max(1.0, duration)),
            "-c:a", "libmp3lame", "-q:a", "4",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=30)
        return output_path

    def _concat_audio(self, chunks: list[str], output: str) -> None:
        """Concatenate MP3 chunks with FFmpeg concat demuxer."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            list_path = f.name
            for c in chunks:
                safe = os.path.abspath(c).replace("\\", "/")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300
        )
        os.unlink(list_path)

        if result.returncode != 0:
            raise RuntimeError(f"Audio concat failed:\n{result.stderr[-1000:]}")