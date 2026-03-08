"""
agents/tts_agent.py

TTS via OpenAI Audio API (pay-as-you-go, no subscription).
  tts-1    = $15/1M chars  (~$0.15 per 10-min video)
  tts-1-hd = $30/1M chars  (~$0.30 per 10-min video, better quality)
  Voices: onyx (deep/authoritative), nova, alloy, echo, fable, shimmer

OpenAI TTS does not return word-level timestamps. We estimate them by:
  1. Probing actual audio duration with ffprobe
  2. Distributing word timestamps proportionally across that duration
     (proportional to char position within the text)
This gives ~+-0.3s accuracy, sufficient for Manim animation sync.

Narration text is pre-processed by _clean_narration_for_tts() before sending
to the API — this converts raw math/LaTeX notation into spoken equivalents so
the voice doesn't spell out "X transpose T dot W alpha" letter by letter.
"""
from __future__ import annotations
import os
import re
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
        self._log("Generating TTS audio via OpenAI (pay-as-you-go)...")

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

        scene_audio_paths: list[str]           = []
        all_timestamps:    list[WordTimestamp]  = []
        time_offset = 0.0

        def _process_scene(scene):
            """Process one scene: TTS call + duration probe + timestamps. Thread-safe."""
            raw_text = scene.narration_text.strip()
            if not raw_text:
                return scene, None, 0.0

            text       = self._clean_narration_for_tts(raw_text)
            chunk_path = os.path.join(
                audio_dir, f"{state.topic_slug}_scene_{scene.id:02d}.mp3"
            )
            self._log(f"Scene {scene.id}: {len(text)} chars -> OpenAI TTS...")

            try:
                response = client.audio.speech.create(
                    model           = self.cfg.openai_tts_model,
                    voice           = self.cfg.openai_tts_voice,
                    input           = text,
                    response_format = "mp3",
                )
                response.stream_to_file(chunk_path)
            except Exception as e:
                self._log(f"Scene {scene.id} TTS failed: {e}")
                chunk_path = self._write_silence(scene.duration_seconds, chunk_path)
                return scene, chunk_path, scene.duration_seconds

            chunk_duration = self._probe_duration(chunk_path)
            if chunk_duration <= 0:
                chunk_duration = len(text.split()) / 2.8

            # Build per-scene word timestamps (relative, offset applied later)
            words       = text.split()
            total_chars = len(text)
            char_cursor = 0
            scene_ts    = []
            for word in words:
                sr = char_cursor / max(total_chars, 1)
                er = (char_cursor + len(word)) / max(total_chars, 1)
                scene_ts.append(WordTimestamp(
                    word  = word,
                    start = round(sr * chunk_duration, 4),
                    end   = round(er * chunk_duration, 4),
                ))
                char_cursor += len(word) + 1
            scene._tts_relative_ts = scene_ts   # stash for offset application below
            return scene, chunk_path, chunk_duration

        # Run TTS calls in parallel
        max_tts_workers = min(len(state.scenes), self.cfg.tts_workers)
        scene_results   = {}  # scene.id -> (chunk_path, duration)

        with ThreadPoolExecutor(max_workers=max_tts_workers) as pool:
            futures = {pool.submit(_process_scene, s): s for s in state.scenes}
            for fut in as_completed(futures):
                s, chunk_path, dur = fut.result()
                if chunk_path:
                    scene_results[s.id] = (chunk_path, dur)
                    self._log(f"Scene {s.id}: {dur:.1f}s TTS done")

        # Apply global time offsets in scene order (must be sequential)
        for scene in state.scenes:
            if scene.id not in scene_results:
                continue
            chunk_path, chunk_duration = scene_results[scene.id]

            scene.duration_seconds = chunk_duration
            scene.tts_audio_path   = chunk_path
            scene.tts_duration     = chunk_duration
            scene_audio_paths.append(chunk_path)

            # Apply cumulative offset to relative timestamps
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

        # Concatenate all chunks into final audio file
        final_audio = os.path.join(audio_dir, f"{state.topic_slug}.mp3")

        if not scene_audio_paths:
            raise RuntimeError(
                "No audio generated. Check OPENAI_API_KEY and account credits at "
                "https://platform.openai.com/account/usage"
            )
        elif len(scene_audio_paths) == 1:
            shutil.copy2(scene_audio_paths[0], final_audio)
        else:
            self._concat_audio(scene_audio_paths, final_audio)

        state.audio_path           = final_audio
        state.all_timestamps       = all_timestamps
        state.total_audio_duration = time_offset

        total_chars = sum(len(s.narration_text) for s in state.scenes)
        cost_est    = total_chars / 1_000_000 * (
            30.0 if "hd" in self.cfg.openai_tts_model else 15.0
        )
        self._log(
            f"Audio ready -> {final_audio}\n"
            f"  {time_offset:.1f}s ({time_offset/60:.1f} min) | "
            f"{len(all_timestamps)} word timestamps | "
            f"est. cost ${cost_est:.4f}"
        )

        return state

    # ── Narration preprocessor ────────────────────────────────────────────────

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