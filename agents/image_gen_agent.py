"""
agents/image_gen_agent.py
Generates images for IMAGE_GEN and IMAGE_MANIM_HYBRID scenes using NanoBanana.
Selects model tier (Pro vs Fast) based on scene complexity.
Applies PIL label overlay when scene.needs_labels = True.
"""
from __future__ import annotations
import base64
import os
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig, RESOLUTIONS
from utils.llm_client import LLMClient
from utils.label_overlay import add_labels
from agents.base_agent import BaseAgent


class ImageGenAgent(BaseAgent):
    name = "ImageGenAgent"

    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        super().__init__(cfg, llm)
        self._genai = None
        self._init_genai()

    def _init_genai(self):
        if not self.cfg.gemini_api_key:
            self._log("WARNING: GEMINI_API_KEY not set — IMAGE_GEN scenes will be skipped")
            return
        try:
            from google import genai
            self._genai = genai.Client(api_key=self.cfg.gemini_api_key)
        except ImportError:
            self._log("WARNING: google-genai SDK not installed — pip install google-genai")

    def run(self, state: ProcExState) -> ProcExState:
        targets = [
            s for s in state.scenes
            if s.visual_strategy == VisualStrategy.IMAGE_GEN
        ]

        if not targets:
            self._log("No IMAGE_GEN scenes — skipping")
            return state

        self._log(f"Generating images for {len(targets)} scenes...")
        images_dir = self.cfg.dirs["images"]
        os.makedirs(images_dir, exist_ok=True)

        for scene in targets:
            try:
                self._generate_for_scene(scene, state.resolution, images_dir)
            except Exception as e:
                self._err(state, f"Scene {scene.id} image gen failed: {e}")
                self._log(f"Scene {scene.id}: will render with fallback text animation")
                scene.visual_strategy = VisualStrategy.TEXT_ANIMATION

        return state

    def _generate_for_scene(self, scene: Scene, resolution: str, images_dir: str) -> None:
        if self._genai is None:
            raise RuntimeError("Gemini client not initialized")

        res = RESOLUTIONS[resolution]

        # ── Choose model tier ────────────────────────────────────────────
        # Pro: labeled diagrams, complex anatomy, search grounding needed
        # Fast: backgrounds for hybrid scenes, simpler imagery
        use_pro = (
            scene.needs_labels
            or scene.visual_strategy == VisualStrategy.IMAGE_GEN
            or len(scene.label_list) > 0
        )
        model = self.cfg.nano_pro_model if use_pro else self.cfg.nano_fast_model
        self._log(f"Scene {scene.id}: using {'Pro' if use_pro else 'Fast'} model ({model})")

        # ── Build enriched prompt ────────────────────────────────────────
        prompt = self._enrich_prompt(scene, res.nano_res, res.aspect_ratio)

        # ── Call NanoBanana ──────────────────────────────────────────────
        from google.genai import types

        response = self._genai.models.generate_content(
            model    = model,
            contents = prompt,
            config   = types.GenerateContentConfig(
                response_modalities=["image", "text"],
            ),
        )

        # ── Extract and save image ───────────────────────────────────────
        image_path = os.path.join(images_dir, f"scene_{scene.id:02d}_raw.png")

        saved = False
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                img_bytes = part.inline_data.data
                if isinstance(img_bytes, str):
                    img_bytes = base64.b64decode(img_bytes)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                saved = True
                break

        if not saved:
            # Try as_image() API
            try:
                from PIL import Image
                for part in response.parts:
                    if hasattr(part, "as_image"):
                        img = part.as_image()
                        img.save(image_path)
                        saved = True
                        break
            except Exception:
                pass

        if not saved:
            raise RuntimeError(f"NanoBanana returned no image data for scene {scene.id}")

        self._log(f"Scene {scene.id}: image saved → {image_path}")

        # ── Apply label overlay ──────────────────────────────────────────
        final_path = image_path
        if scene.needs_labels and scene.label_list:
            labeled_path = os.path.join(images_dir, f"scene_{scene.id:02d}_labeled.png")
            try:
                final_path = add_labels(image_path, scene.label_list, labeled_path)
                self._log(f"Scene {scene.id}: labels applied → {final_path}")
            except Exception as e:
                self._log(f"Scene {scene.id}: label overlay failed ({e}) — using raw image")
                final_path = image_path

        scene.image_paths = [final_path]



    @staticmethod
    def _enrich_prompt(scene: Scene, nano_res: str, aspect_ratio: str = "16:9") -> str:
        """Enrich the VisualDirector's prompt for NanoBanana."""
        base = scene.visual_prompt.strip()

        additions = []
        if nano_res not in base:
            additions.append(f"{nano_res} resolution")
        if aspect_ratio not in base:
            additions.append(f"{aspect_ratio} aspect ratio")
        if scene.needs_labels and "label" not in base.lower():
            label_str = ", ".join(scene.label_list[:8])
            additions.append(f"clearly labeled callouts for: {label_str}")

        if additions:
            base += ". " + ". ".join(additions) + "."

        return base