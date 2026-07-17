from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest


def _run_node(root: Path, script: str) -> dict[str, object]:
    completed = subprocess.run(
        [shutil.which("node") or "node", "--input-type=module", "-e", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_asset_normalization_centers_off_origin_geometry_after_scaling() -> None:
    root = Path(__file__).parents[1]
    renderer_url = (root / "environment_generation" / "studio_web" / "visual_renderer.js").as_uri()
    three_url = (root / "environment_generation" / "studio_web" / "vendor" / "three.module.js").as_uri()
    script = f"""
      import * as THREE from {json.dumps(three_url)};
      import {{ normalizeAsset }} from {json.dumps(renderer_url)};

      const source = new THREE.Group();
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 4, 1));
      mesh.position.set(3, 2, -5);
      source.add(mesh);
      const target = new THREE.Vector3(6, 2, 0.3);

      normalizeAsset(source, target, false);
      source.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(source);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      console.log(JSON.stringify({{ center: center.toArray(), size: size.toArray() }}));
    """

    result = _run_node(root, script)

    assert result["center"] == pytest.approx([0, 0, 0], abs=1e-9)
    assert result["size"] == pytest.approx([6, 2, 0.3], abs=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_fence_asset_is_centered_inside_its_declared_wall_bounds() -> None:
    root = Path(__file__).parents[1]
    web_root = root / "environment_generation" / "studio_web"
    renderer_url = (web_root / "visual_renderer.js").as_uri()
    three_url = (web_root / "vendor" / "three.module.js").as_uri()
    loader_url = (web_root / "vendor" / "GLTFLoader.js").as_uri()
    fence_path = web_root / "assets" / "courtyard" / "kenney_nature" / "fence_simple.glb"
    script = f"""
      import fs from "node:fs";
      import * as THREE from {json.dumps(three_url)};
      import {{ GLTFLoader }} from {json.dumps(loader_url)};
      import {{ normalizeAsset }} from {json.dumps(renderer_url)};

      const bytes = fs.readFileSync({json.dumps(str(fence_path))});
      const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      const gltf = await new Promise((resolve, reject) => new GLTFLoader().parse(buffer, "", resolve, reject));
      const target = new THREE.Vector3(1.35, 1, 0.35);

      normalizeAsset(gltf.scene, target, false);
      gltf.scene.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      console.log(JSON.stringify({{
        center: box.getCenter(new THREE.Vector3()).toArray(),
        size: box.getSize(new THREE.Vector3()).toArray(),
      }}));
    """

    result = _run_node(root, script)

    assert result["center"] == pytest.approx([0, 0, 0], abs=1e-9)
    assert result["size"] == pytest.approx([1.35, 1, 0.35], abs=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_threejs_ramp_geometry_matches_authored_walkable_endpoints() -> None:
    root = Path(__file__).parents[1]
    geometry_url = (root / "environment_generation" / "studio_web" / "ramp_geometry.js").as_uri()
    script = f"""
      import {{ rampRenderGeometry }} from {json.dumps(geometry_url)};
      const geometry = rampRenderGeometry({{
        position: [999, 999, 999],
        size: [3, 2, 0.25],
        yaw: 0,
        metadata: {{ geometry_version: 2, rise: 1, low_end: [0, 1, 0] }},
      }});
      console.log(JSON.stringify(geometry));
    """

    result = _run_node(root, script)

    assert result["lowEnd"] == pytest.approx([0, 1, 0])
    assert result["highEnd"] == pytest.approx([3, 1, 1])
    assert result["size"] == pytest.approx([math.sqrt(10), 2, 0.25])
    assert result["angle"] == pytest.approx(math.atan2(1, 3))
