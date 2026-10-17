"""SGLang Diffusion image generation backend.

This backend talks to an SGLang Diffusion HTTP server using the
OpenAI-compatible image generation endpoint:

    POST /v1/images/generations

The caller owns the server lifecycle and passes its base URL.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None

if TYPE_CHECKING:
    # `PILImage` above is the PIL.Image *module*, so it cannot annotate a
    # single image. The class is PIL.Image.Image.
    from PIL.Image import Image as PILImageType

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]


class SGLangImageBackend:
    """Image backend for SGLang Diffusion's OpenAI-compatible image API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: int = 900,
        request_model: Optional[str] = None,
        validate_server: bool = True,
        max_concurrent_requests: int = 1,
    ) -> None:
        if PILImage is None:  # pragma: no cover
            raise RuntimeError(
                "SGLangImageBackend requires Pillow. Install it with `pip install Pillow`."
            )
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be non-empty")

        self._server_root_url, self._api_base_url = self._normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout_seconds = int(timeout_seconds)
        self._request_model = request_model
        self._max_concurrent_requests = int(max_concurrent_requests)
        if self._max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be a positive integer, "
                f"got {max_concurrent_requests!r}."
            )

        if validate_server:
            self.validate_server()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def validate_server(self) -> None:
        """Raise if the configured SGLang server cannot be reached."""
        candidate_urls = (
            f"{self._server_root_url}/models",
            f"{self._api_base_url}/models",
            f"{self._server_root_url}/health",
        )

        errors: List[str] = []
        for url in candidate_urls:
            try:
                self._read_json(url, timeout_seconds=10)
                logger.info("SGLang server is reachable at %s", url)
                return
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "Cannot reach SGLang server. Tried:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]] = None,
        params: Optional[Mapping[str, Any]] = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> List[PILImageType]:
        """Generate one image per prompt and return PIL Images."""
        params = params or {}
        prompts = list(prompts)

        if negative_prompts is not None and len(negative_prompts) != len(prompts):
            raise ValueError(
                f"Expected {len(prompts)} negative prompts, got {len(negative_prompts)}"
            )
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} seeds, got {len(seeds)}")

        def generate_one(index: int) -> PILImageType:
            negative_prompt = (
                negative_prompts[index] if negative_prompts is not None else None
            )
            seed = seeds[index] if seeds is not None else None
            return self.generate_one(
                prompt=prompts[index],
                negative_prompt=negative_prompt,
                params=params,
                seed=seed,
            )

        if self._max_concurrent_requests == 1 or len(prompts) <= 1:
            return [generate_one(i) for i in range(len(prompts))]

        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool:
            futures = [pool.submit(generate_one, i) for i in range(len(prompts))]
            return [future.result() for future in futures]

    def generate_one(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> PILImageType:
        """Generate a single image and return a PIL Image."""
        payload = self._build_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            params=params or {},
            seed=seed,
        )
        result = self._read_json(
            f"{self._api_base_url}/images/generations",
            payload=payload,
            timeout_seconds=self._timeout_seconds,
        )
        return self._decode_image_response(result, prompt)

    def shutdown(self) -> None:
        """Release HTTP backend resources."""

    def __enter__(self) -> "SGLangImageBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()

    # ---------------------------------------------------------------------
    # Payload and response handling
    # ---------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        params: Mapping[str, Any],
        seed: Optional[int],
    ) -> JsonDict:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        payload: JsonDict = {
            "prompt": prompt,
            "n": int(params.get("n", 1)),
            "response_format": params.get("response_format", "b64_json"),
        }

        if payload["n"] != 1:
            raise ValueError(
                "This backend currently expects n=1 because it decodes only one image "
                "per request. Use multiple prompts for batch generation."
            )

        if self._request_model:
            payload["model"] = self._request_model
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = int(seed)

        size = self._extract_size(params)
        if size:
            payload["size"] = size

        # Common diffusion knobs. Keep names exactly as the server expects.
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "output-quality",
            "output-compression",
        ):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        # Friendly aliases for callers that prefer Python identifiers.
        aliases = {
            "output_quality": "output-quality",
            "output_compression": "output-compression",
        }
        for source_key, target_key in aliases.items():
            if source_key in params and params[source_key] is not None:
                payload[target_key] = params[source_key]

        reserved = {
            "prompt",
            "negative_prompt",
            "model",
            "n",
            "response_format",
            "size",
            "width",
            "height",
            "seed",
            "generator",
            "num_inference_steps",
            "guidance_scale",
            "output-quality",
            "output-compression",
            "output_quality",
            "output_compression",
        }

        for key, value in params.items():
            if key not in reserved and value is not None:
                payload[key] = value

        return payload

    @staticmethod
    def _extract_size(params: Mapping[str, Any]) -> Optional[str]:
        if params.get("size"):
            return str(params["size"])

        width = params.get("width")
        height = params.get("height")
        if width is None or height is None:
            return None

        return f"{int(width)}x{int(height)}"

    @staticmethod
    def _decode_image_response(result: Dict[str, Any], prompt: str) -> PILImageType:
        try:
            data = result["data"]
            if not isinstance(data, list) or not data:
                raise TypeError("response field `data` must be a non-empty list")

            first = data[0]
            if not isinstance(first, Mapping):
                raise TypeError("response field `data[0]` must be an object")

            b64_data = first["b64_json"]
            if not isinstance(b64_data, str):
                raise TypeError("response field `data[0].b64_json` must be a string")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected SGLang image response for prompt "
                f"{_prompt_preview(prompt)!r}: {result!r}"
            ) from exc

        try:
            image_bytes = base64.b64decode(b64_data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                "SGLang returned invalid base64 image data for prompt "
                f"{_prompt_preview(prompt)!r}"
            ) from exc

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as image:  # type: ignore[union-attr]
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                "Could not decode SGLang image response for prompt "
                f"{_prompt_preview(prompt)!r}"
            ) from exc

    # ---------------------------------------------------------------------
    # HTTP helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> Tuple[str, str]:
        normalized = base_url.strip().rstrip("/")

        if normalized.endswith("/v1"):
            api_base_url = normalized
            server_root_url = normalized[: -len("/v1")].rstrip("/")
        else:
            server_root_url = normalized
            api_base_url = f"{normalized}/v1"

        return server_root_url, api_base_url

    def _read_json(
        self,
        url: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> JsonDict:
        return self._read_json_static(
            url=url,
            api_key=self._api_key,
            payload=payload,
            timeout_seconds=timeout_seconds or self._timeout_seconds,
        )

    @staticmethod
    def _read_json_static(
        *,
        url: str,
        api_key: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
        timeout_seconds: int,
    ) -> JsonDict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if body is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang server returned HTTP {exc.code} for {url}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach SGLang server at {url}: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"SGLang server returned non-JSON response from {url}: {raw[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"SGLang server returned unexpected JSON from {url}: {parsed!r}"
            )

        return parsed


def _prompt_preview(prompt: str, limit: int = 80) -> str:
    compact = " ".join(prompt.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."
