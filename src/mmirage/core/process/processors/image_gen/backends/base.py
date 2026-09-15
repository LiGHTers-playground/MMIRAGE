"""Image generation backend protocol for MMIRAGE."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from PIL.Image import Image as PILImage


@runtime_checkable
class ImageGenerationBackend(Protocol):
    """Protocol for pluggable image generation backends.

    All backends receive pre-rendered prompts and pre-computed per-sample seeds
    from the processor.  The processor handles all Jinja template rendering,
    filename generation, and result bookkeeping; the backend is responsible
    only for turning prompts + params into PIL images.
    """

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[Optional[str]]],
        params: Dict[str, Any],
        seeds: List[Optional[int]],
    ) -> List[PILImage]:
        """Generate one image per prompt.

        Args:
            prompts: Positive prompt strings, one per sample.
            negative_prompts: Optional list of negative prompts aligned with
                ``prompts``.  ``None`` means no negative prompts at all;
                individual ``None`` elements mean no negative prompt for that
                sample.
            params: Shared generation kwargs (width, height,
                num_inference_steps, guidance_scale, …).
            seeds: Per-sample integer seeds for deterministic generation, or
                ``None`` elements for unseeded samples.  The list is always
                the same length as ``prompts``.

        Returns:
            List of ``PIL.Image`` objects, one per prompt, in the same order.
        """
        ...

    def generate_one(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> PILImage:
        """Generate a single image for one prompt."""
        ...

    def shutdown(self) -> None:
        """Release any resources held by the backend."""
        ...
