from __future__ import annotations

from pathlib import Path

from .runtime_config import configure_mujoco_gl


class PreviewError(RuntimeError):
    """Raised when MuJoCo preview rendering fails."""


def render_preview(
    xml_path: Path,
    output_path: Path,
    *,
    camera: str = "overview",
    width: int = 1280,
    height: int = 820,
    settle_steps: int = 8,
) -> Path:
    try:
        import mujoco
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local install
        raise PreviewError(f"MuJoCo/Pillow preview dependencies are unavailable: {exc}") from exc

    configure_mujoco_gl()

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        for _ in range(max(0, settle_steps)):
            mujoco.mj_step(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            renderer.update_scene(data, camera=camera_id if camera_id >= 0 else None)
            pixels = renderer.render()
        finally:
            renderer.close()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output_path)
    except Exception as exc:
        raise PreviewError(str(exc)) from exc
    return output_path


def render_orbit_previews(
    xml_path: Path,
    output_paths: list[Path],
    *,
    width: int = 1280,
    height: int = 820,
    settle_steps: int = 8,
) -> list[Path]:
    if not output_paths:
        return []
    try:
        import mujoco
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local install
        raise PreviewError(f"MuJoCo/Pillow preview dependencies are unavailable: {exc}") from exc

    configure_mujoco_gl()

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        for _ in range(max(0, settle_steps)):
            mujoco.mj_step(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat = [float(model.stat.center[0]), float(model.stat.center[1]), 0.75]
            camera.distance = max(8.0, float(model.stat.extent) * 0.65)
            camera.elevation = -28.0
            for index, output_path in enumerate(output_paths):
                camera.azimuth = -90.0 + (360.0 * index / len(output_paths))
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(output_path)
        finally:
            renderer.close()
    except Exception as exc:
        raise PreviewError(str(exc)) from exc
    return output_paths


def image_nonblank(path: Path) -> bool:
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - depends on local install
        return False
    image = Image.open(path).convert("RGB")
    extrema = image.getextrema()
    return any(low != high for low, high in extrema)
