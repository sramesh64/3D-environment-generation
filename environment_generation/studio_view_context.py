from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


STUDIO_VIEW_CONTEXT_FILENAME = "studio_view_context.json"
STUDIO_VIEW_CONTEXT_SCHEMA_VERSION = "2.0"
MAX_SUBMITTED_VIEW_IMAGE_BYTES = 4 * 1024 * 1024
MIN_SUBMITTED_VIEW_EDGE = 96
MAX_SUBMITTED_VIEW_EDGE = 2048
SCREEN_REGIONS = {
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
}

_REGION_PATTERNS = (
    ("bottom_left", re.compile(r"\b(?:bottom|lower)[\s-]+left\b", re.IGNORECASE)),
    ("bottom_right", re.compile(r"\b(?:bottom|lower)[\s-]+right\b", re.IGNORECASE)),
    ("top_left", re.compile(r"\b(?:top|upper)[\s-]+left\b", re.IGNORECASE)),
    ("top_right", re.compile(r"\b(?:top|upper)[\s-]+right\b", re.IGNORECASE)),
    ("bottom_center", re.compile(r"\b(?:bottom|lower)[\s-]+(?:center|middle)\b", re.IGNORECASE)),
    ("top_center", re.compile(r"\b(?:top|upper)[\s-]+(?:center|middle)\b", re.IGNORECASE)),
    ("center_left", re.compile(r"\b(?:center|middle)[\s-]+left\b", re.IGNORECASE)),
    ("center_right", re.compile(r"\b(?:center|middle)[\s-]+right\b", re.IGNORECASE)),
)


def detect_screen_region_requirements(prompt: str, view_context: dict[str, Any]) -> list[dict[str, Any]]:
    screen_space = view_context.get("screen_space") if isinstance(view_context, dict) else None
    regions = screen_space.get("regions") if isinstance(screen_space, dict) else None
    camera = screen_space.get("camera") if isinstance(screen_space, dict) else None
    if not isinstance(regions, dict) or not isinstance(camera, dict):
        return []
    requirements = []
    for region, pattern in _REGION_PATTERNS:
        if pattern.search(prompt) and _valid_region_payload(regions.get(region)):
            requirements.append(
                {
                    "type": "screen_region",
                    "region": region,
                    "source": "submitted_user_view",
                    "description": f"The edited object must appear in the {region.replace('_', '-')} region of the submitted view.",
                }
            )
    return requirements


def persist_studio_view_context(
    *,
    scene_dir: Path,
    review_id: str,
    prompt: str,
    view_context: dict[str, Any],
    image_data_url: str = "",
) -> dict[str, Any]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    image = _persist_submitted_image(scene_dir, review_id, image_data_url) if image_data_url else None
    requirements = detect_screen_region_requirements(prompt, view_context)
    payload = {
        "schema_version": STUDIO_VIEW_CONTEXT_SCHEMA_VERSION,
        "review_id": review_id,
        "prompt": prompt,
        "capture_kind": str(view_context.get("capture_kind") or "structured_before_edit"),
        "captured_at": str(view_context.get("captured_at") or ""),
        "view_context": view_context,
        "requirements": requirements,
        "submitted_image": image,
    }
    _atomic_write_json(scene_dir / STUDIO_VIEW_CONTEXT_FILENAME, payload)
    review_dir = scene_dir / "visual_reviews" / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(review_dir / STUDIO_VIEW_CONTEXT_FILENAME, payload)
    return payload


def load_studio_view_context(scene_dir: Path) -> dict[str, Any] | None:
    path = scene_dir / STUDIO_VIEW_CONTEXT_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def submitted_view_image_path(scene_dir: Path, payload: dict[str, Any] | None) -> Path | None:
    image = payload.get("submitted_image") if isinstance(payload, dict) else None
    relative = image.get("path") if isinstance(image, dict) else None
    if not isinstance(relative, str) or not relative:
        return None
    path = (scene_dir / relative).resolve()
    try:
        path.relative_to(scene_dir.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def active_screen_requirements(payload: dict[str, Any] | None, *, prompt: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or str(payload.get("prompt") or "").strip() != prompt.strip():
        return []
    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        return []
    return [item for item in requirements if isinstance(item, dict) and item.get("region") in SCREEN_REGIONS]


def verification_projection(payload: dict[str, Any] | None, region: str) -> dict[str, Any] | None:
    if region not in SCREEN_REGIONS or not isinstance(payload, dict):
        return None
    view_context = payload.get("view_context")
    screen_space = view_context.get("screen_space") if isinstance(view_context, dict) else None
    camera = screen_space.get("camera") if isinstance(screen_space, dict) else None
    regions = screen_space.get("regions") if isinstance(screen_space, dict) else None
    region_payload = regions.get(region) if isinstance(regions, dict) else None
    if not _valid_camera(camera) or not _valid_region_payload(region_payload):
        return None
    anchor = region_payload.get("anchor") if isinstance(region_payload, dict) else None
    return {
        "context_review_id": str(payload.get("review_id") or ""),
        "camera": camera,
        "region_bounds_uv": region_payload["bounds_uv"],
        "anchor_world_position": anchor.get("world_position") if isinstance(anchor, dict) else None,
    }


def verification_camera(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    view_context = payload.get("view_context")
    screen_space = view_context.get("screen_space") if isinstance(view_context, dict) else None
    camera = screen_space.get("camera") if isinstance(screen_space, dict) else None
    if not _valid_camera(camera):
        return None
    return {
        "context_review_id": str(payload.get("review_id") or ""),
        "camera": camera,
    }


def _persist_submitted_image(scene_dir: Path, review_id: str, data_url: str) -> dict[str, Any]:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("submitted_view_image must be a PNG data URL")
    try:
        data = base64.b64decode(data_url[len(prefix) :], validate=True)
    except ValueError as exc:
        raise ValueError("submitted_view_image contains invalid base64") from exc
    if not data or len(data) > MAX_SUBMITTED_VIEW_IMAGE_BYTES:
        raise ValueError("submitted_view_image exceeds the image size limit")
    width, height = _validated_png_dimensions(data)
    if not (
        MIN_SUBMITTED_VIEW_EDGE <= width <= MAX_SUBMITTED_VIEW_EDGE
        and MIN_SUBMITTED_VIEW_EDGE <= height <= MAX_SUBMITTED_VIEW_EDGE
    ):
        raise ValueError("submitted_view_image dimensions are outside the allowed range")
    relative = Path("visual_reviews") / review_id / "submitted_view.png"
    path = scene_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative.as_posix(),
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validated_png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("submitted_view_image is not a valid PNG")
    header_dimensions = struct.unpack(">II", data[16:24])
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("submitted_view_image is not a PNG")
            decoded_dimensions = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("submitted_view_image is not a decodable PNG") from exc
    if decoded_dimensions != header_dimensions:
        raise ValueError("submitted_view_image has inconsistent PNG dimensions")
    return decoded_dimensions


def _valid_camera(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    position = value.get("position")
    target = value.get("target")
    return (
        _finite_vector(position, 3)
        and _finite_vector(target, 3)
        and _finite_number(value.get("fov_y_degrees"))
        and _finite_number(value.get("aspect"))
        and float(value["aspect"]) > 0
    )


def _valid_region_payload(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("bounds_uv"), dict):
        return False
    bounds = value["bounds_uv"]
    if not all(_finite_number(bounds.get(key)) for key in ("left", "top", "right", "bottom")):
        return False
    return float(bounds["left"]) < float(bounds["right"]) and float(bounds["top"]) < float(bounds["bottom"])


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_finite_number(item) for item in value)


def _finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
