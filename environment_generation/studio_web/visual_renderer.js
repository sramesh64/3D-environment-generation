import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { COURTYARD_ASSETS, courtyardAssetKey } from "./courtyard_assets.js";
import { rampRenderGeometry } from "./ramp_geometry.js";

const MIN_SIZE = 0.02;

export class VisualPreviewRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.fixedSize = Array.isArray(options.fixedSize) ? options.fixedSize.map(Number) : null;
    this.captureMode = Boolean(options.captureMode);
    this.clampView = options.clampView !== false;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: this.captureMode,
    });
    this.renderer.setPixelRatio(this.captureMode ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.className = "visual-canvas";
    this.container.appendChild(this.renderer.domElement);
    this.environment = new THREE.Group();
    this.scene.add(this.environment);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.physicsBindings = new Map();
    this.visualBindings = new Map();
    this.gameBindings = new Map();
    this.followSourceId = "";
    this.followInitialized = false;
    this.view = { target: [0, 0, 0.5], distance: 14, azimuth: -42, elevation: 38, panX: 0, panY: 0 };
    this.cameraPose = null;
    this.defaultFov = 45;
    this.materials = {};
    this.resizeObserver = this.fixedSize ? null : new ResizeObserver(() => this.resize());
    this.resizeObserver?.observe(this.container);
    this.resize();
    this.sceneRevision = 0;
    this.assetsReady = Promise.resolve();
  }

  async setScene(visualScene) {
    const revision = ++this.sceneRevision;
    this.visualScene = visualScene;
    this.palette = visualScene?.palette || {};
    this.materials = makeMaterials(this.palette);
    clearGroup(this.root);
    this.physicsBindings.clear();
    this.visualBindings.clear();
    this.gameBindings.clear();
    this.followSourceId = "";
    this.followInitialized = false;
    this.configureEnvironment();
    const camera = visualScene?.camera || {};
    this.view.target = camera.target || [0, 0, 0.5];
    this.view.distance = Number(camera.distance || this.view.distance);
    this.view.azimuth = Number(camera.azimuth ?? this.view.azimuth);
    this.view.elevation = Number(camera.elevation ?? this.view.elevation);
    this.view.panX = 0;
    this.view.panY = 0;
    this.render();
    this.assetsReady = courtyardAssetLibrary.preload(visualScene?.objects || []);
    await this.assetsReady;
    if (revision !== this.sceneRevision) return;
    for (const object of visualScene?.objects || []) {
      const node = this.createObject(object);
      if (!node) continue;
      this.root.add(node);
      if (object.source_id) this.visualBindings.set(object.source_id, node);
      if (object.semantic_type === "floor_switch" && object.source_id) {
        this.gameBindings.set(object.source_id, node);
      }
      if (object.physics_backed && ["dynamic", "mechanism"].includes(object.body_type) && object.source_id) {
        const initialRotation = node.quaternion.clone();
        const localOffset = node.position
          .clone()
          .sub(toThree(object.position))
          .applyQuaternion(initialRotation.clone().invert());
        this.physicsBindings.set(object.source_id, {
          node,
          localOffset,
          followHeight: Number(object.size?.[2] || 1) * 0.24,
        });
      }
    }
    this.render();
  }

  beginAgentFollow(sourceId) {
    this.followSourceId = sourceId || "";
    this.followInitialized = false;
  }

  applyPhysicsState(objects) {
    let followPosition = null;
    for (const transform of objects || []) {
      const binding = this.physicsBindings.get(transform.id);
      if (!binding || !Array.isArray(transform.position) || !Array.isArray(transform.rotation_matrix)) continue;
      const rotation = physicsRotationToThree(transform.rotation_matrix);
      const position = toThree(transform.position);
      binding.node.quaternion.copy(rotation);
      binding.node.position.copy(position).add(binding.localOffset.clone().applyQuaternion(rotation));
      if (transform.id === this.followSourceId) {
        followPosition = transform.position.map(Number);
        followPosition[2] += binding.followHeight;
      }
    }
    if (followPosition) {
      if (!this.followInitialized) {
        this.view.target = followPosition;
        this.followInitialized = true;
      } else {
        this.view.target = this.view.target.map((value, index) => THREE.MathUtils.lerp(Number(value), followPosition[index], 0.52));
      }
    }
    this.render();
  }

  applyGameState(game) {
    const activeTriggers = new Set(
      (game?.mechanisms || []).filter((item) => item.active).map((item) => item.trigger_id),
    );
    for (const [sourceId, node] of this.gameBindings) {
      const active = activeTriggers.has(sourceId);
      node.traverse((child) => {
        if (!child.isMesh || !child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) {
          material.emissive?.set(active ? 0x54f29b : 0x153b45);
          material.emissiveIntensity = active ? 0.75 : 0.12;
        }
      });
    }
    this.render();
  }

  setObjectVisibility(sourceId, visible) {
    const node = this.visualBindings.get(String(sourceId || ""));
    if (!node) return false;
    node.visible = Boolean(visible);
    this.render();
    return true;
  }

  setCameraPose(pose) {
    const position = finiteVector(pose?.position, 3, "camera position");
    const target = finiteVector(pose?.target, 3, "camera target");
    const fov = Number(pose?.fov_y_degrees ?? 64);
    if (!Number.isFinite(fov) || fov < 30 || fov > 100) {
      throw new Error("camera fov must be between 30 and 100 degrees");
    }
    this.cameraPose = { position, target, fov };
    this.render();
  }

  clearCameraPose() {
    this.cameraPose = null;
    this.camera.fov = this.defaultFov;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  setView(nextView) {
    this.cameraPose = null;
    this.camera.fov = this.defaultFov;
    this.camera.updateProjectionMatrix();
    this.view = { ...this.view, ...nextView };
    const camera = this.visualScene?.camera || {};
    if (this.clampView) {
      const minDistance = Number(camera.min_distance || 4);
      const maxDistance = Number(camera.max_distance || 40);
      this.view.distance = clamp(this.view.distance, minDistance, maxDistance);
    }
    this.view.elevation = clamp(this.view.elevation, 18, 72);
    this.render();
  }

  configureEnvironment() {
    clearGroup(this.environment);
    this.scene.background = gradientTexture(this.palette.sky_top || "#9DDDF7", this.palette.sky_bottom || "#DDF2FF");
    this.scene.fog = new THREE.Fog(this.palette.sky_bottom || "#DDF2FF", 38, 92);
    const terrain = primaryTerrainObject(this.visualScene);
    this.environment.add(createSkyDome(this.palette, this.visualScene?.world_size || [12, 12, 4]));
    this.environment.add(createSun(this.palette, this.visualScene?.world_size || [12, 12, 4]));
    if (terrain) this.environment.add(createSurroundingTerrain(terrain, this.materials));
    const hemi = new THREE.HemisphereLight(0xf8fff0, 0x6f8d62, 2.5);
    this.environment.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff1c7, 3.4);
    sun.position.set(-8, 12, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 60;
    sun.shadow.camera.left = -24;
    sun.shadow.camera.right = 24;
    sun.shadow.camera.top = 24;
    sun.shadow.camera.bottom = -24;
    this.environment.add(sun);
    const fill = new THREE.DirectionalLight(0x9ecbff, 0.85);
    fill.position.set(8, 6, -8);
    this.environment.add(fill);
  }

  createObject(object) {
    if (object.visual_type === "wood_ramp" || object.semantic_type === "ramp") {
      return this.createRamp(object);
    }
    if (object.semantic_type === "gate") {
      return this.createGate(object);
    }
    const assetObject = this.createAssetObject(object);
    if (assetObject) return assetObject;
    switch (object.visual_type) {
      case "terrain_grass":
        return this.createTerrain(object);
      case "path_tile":
        return this.createPathTile(object);
      case "fence_wall":
        return this.createFence(object);
      case "hedge_wall":
        return this.createHedge(object);
      case "stone_wall":
        return this.createStoneWall(object);
      case "plain_wall":
        return this.createPlainWall(object);
      case "wood_platform":
        return this.createPlatform(object);
      case "wood_ramp":
        return this.createRamp(object);
      case "crate":
        return this.createCrate(object);
      case "barrel":
        return this.createBarrel(object);
      case "boulder":
        return this.createBoulder(object);
      case "agent_hero":
        return this.createAgent(object);
      case "goal_portal":
        return this.createGoal(object);
      case "hazard_spikes":
        return this.createHazard(object);
      case "courtyard_hazard":
        return this.createCourtyardHazard(object);
      case "goal_pad":
        return this.createGoalPad(object);
      case "target_region":
        return this.createTargetRegion(object);
      case "floor_switch":
        return this.createFloorSwitch(object);
      case "sliding_gate":
        return this.createGate(object);
      case "tree":
        return this.createTree(object);
      case "shrub":
        return this.createShrub(object);
      case "stone":
        return this.createStone(object);
      case "sign":
        return this.createSign(object);
      default:
        return this.createFallback(object);
    }
  }

  createAssetObject(object) {
    const key = courtyardAssetKey(object.appearance);
    if (!key) return null;
    const entry = COURTYARD_ASSETS[key];
    if (!entry?.semantics?.includes(String(object.semantic_type || ""))) return null;
    const source = courtyardAssetLibrary.clone(key);
    if (!source || !entry?.url) return null;
    const wrapper = object.physics_backed ? orientedGroup(object) : baseGroup(object);
    const target = new THREE.Vector3(
      Math.max(MIN_SIZE, Number(object.size?.[0] || 1)),
      Math.max(MIN_SIZE, Number(object.size?.[2] || 1)),
      Math.max(MIN_SIZE, Number(object.size?.[1] || 1)),
    );
    if (object.semantic_type === "wall") {
      const alongX = target.x >= target.z;
      const length = alongX ? target.x : target.z;
      const count = Math.max(1, Math.ceil(length / 1.45));
      const segmentLength = length / count;
      for (let index = 0; index < count; index += 1) {
        const segment = index === 0 ? source : courtyardAssetLibrary.clone(key);
        const segmentTarget = alongX
          ? new THREE.Vector3(segmentLength * 0.98, target.y, target.z)
          : new THREE.Vector3(target.x, target.y, segmentLength * 0.98);
        normalizeAsset(segment, segmentTarget, false);
        const offset = (index + 0.5) * segmentLength - length * 0.5;
        segment.position[alongX ? "x" : "z"] += offset;
        wrapper.add(segment);
      }
    } else {
      normalizeAsset(source, target, !object.physics_backed && entry.anchor === "bottom");
      wrapper.add(source);
    }
    return wrapper;
  }

  createTerrain(object) {
    const group = new THREE.Group();
    const [width, depth, height] = object.size;
    const visualThickness = 0.035;
    const topY = height / 2;
    const base = new THREE.Mesh(new THREE.BoxGeometry(width, visualThickness, depth), this.materials.clearing);
    base.position.y = topY - visualThickness / 2;
    base.receiveShadow = true;
    group.add(base);

    const borderWidth = Math.max(0.12, Math.min(width, depth) * 0.018);
    const borders = [
      { x: 0, z: depth / 2 - borderWidth / 2, w: width, d: borderWidth },
      { x: 0, z: -depth / 2 + borderWidth / 2, w: width, d: borderWidth },
      { x: width / 2 - borderWidth / 2, z: 0, w: borderWidth, d: depth },
      { x: -width / 2 + borderWidth / 2, z: 0, w: borderWidth, d: depth },
    ];
    for (const strip of borders) {
      const border = new THREE.Mesh(new THREE.BoxGeometry(strip.w, 0.018, strip.d), this.materials.clearingEdge);
      border.position.set(strip.x, topY + 0.003, strip.z);
      border.receiveShadow = true;
      group.add(border);
    }

    const patchSize = 1.3;
    for (let x = -width / 2 + patchSize / 2; x < width / 2; x += patchSize) {
      for (let z = -depth / 2 + patchSize / 2; z < depth / 2; z += patchSize) {
        const ix = Math.round((x + width / 2) / patchSize);
        const iz = Math.round((z + depth / 2) / patchSize);
        if ((ix * 7 + iz * 11) % 5 > 1) continue;
        const scale = 0.58 + ((ix * 13 + iz * 17) % 5) * 0.06;
        const patch = new THREE.Mesh(new THREE.CircleGeometry(patchSize * 0.5 * scale, 10), this.materials.clearingPatch);
        patch.position.set(x, topY + 0.012, z);
        patch.rotation.x = -Math.PI / 2;
        patch.rotation.z = ((ix * 19 + iz * 23) % 7 - 3) * 0.18;
        patch.scale.set(1.18, 0.72 + ((ix + iz) % 3) * 0.12, 1);
        patch.receiveShadow = true;
        group.add(patch);
      }
    }
    setWorldTransform(group, object.position, object.yaw);
    return group;
  }

  createPathTile(object) {
    const mesh = new THREE.Mesh(boxGeometry(object.size), this.materials.path);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    setWorldTransform(mesh, object.position, object.yaw);
    return mesh;
  }

  createFence(object) {
    const group = orientedGroup(object);
    const [width, depth, height] = object.size;
    const alongX = width >= depth;
    const length = alongX ? width : depth;
    const postCount = Math.max(3, Math.ceil(length / 1.6) + 1);
    for (let index = 0; index < postCount; index += 1) {
      const t = index / (postCount - 1) - 0.5;
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.16, height * 1.08, 0.16), this.materials.woodDark);
      post.position.set(alongX ? t * length : 0, 0, alongX ? 0 : t * length);
      post.castShadow = true;
      group.add(post);
    }
    for (const y of [-height * 0.18, height * 0.22]) {
      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(alongX ? length : 0.14, 0.12, alongX ? 0.14 : length),
        this.materials.wood,
      );
      rail.position.y = y;
      rail.castShadow = true;
      group.add(rail);
    }
    return group;
  }

  createHedge(object) {
    const group = orientedGroup(object);
    const [width, depth, height] = object.size;
    const base = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.75, depth), this.materials.leaf);
    base.castShadow = true;
    base.receiveShadow = true;
    group.add(base);
    const alongX = width >= depth;
    const length = alongX ? width : depth;
    const count = Math.max(3, Math.ceil(length / 1.2));
    for (let index = 0; index < count; index += 1) {
      const t = index / Math.max(1, count - 1) - 0.5;
      const puff = new THREE.Mesh(new THREE.SphereGeometry(0.28, 10, 8), this.materials.leafLight);
      puff.position.set(alongX ? t * length : 0, height * 0.32, alongX ? 0 : t * length);
      puff.scale.set(1.2, 0.8, 1.0);
      puff.castShadow = true;
      group.add(puff);
    }
    return group;
  }

  createStoneWall(object) {
    const group = orientedGroup(object);
    const [width, depth, height] = object.size;
    const alongX = width >= depth;
    const length = alongX ? width : depth;
    const count = Math.max(2, Math.ceil(length / 1.1));
    for (let index = 0; index < count; index += 1) {
      const t = index / Math.max(1, count - 1) - 0.5;
      const block = new THREE.Mesh(
        new THREE.BoxGeometry(alongX ? length / count * 0.92 : width, height * 0.78, alongX ? depth : length / count * 0.92),
        index % 2 ? this.materials.stone : this.materials.stoneDark,
      );
      block.position.set(alongX ? t * length : 0, 0, alongX ? 0 : t * length);
      block.castShadow = true;
      block.receiveShadow = true;
      group.add(block);
    }
    return group;
  }

  createPlainWall(object) {
    const group = orientedGroup(object);
    const wall = new THREE.Mesh(boxGeometry(object.size), this.materials.stone);
    wall.castShadow = true;
    wall.receiveShadow = true;
    group.add(wall);
    return group;
  }

  createPlatform(object) {
    const group = orientedGroup(object);
    const top = new THREE.Mesh(
      boxGeometry(object.size),
      this.materials.wood,
    );
    top.castShadow = true;
    top.receiveShadow = true;
    group.add(top);
    const [width, depth, height] = object.size;
    const rim = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.18, 0.08, depth + 0.18),
      this.materials.woodDark,
    );
    rim.position.y = height * 0.5 + 0.035;
    group.add(rim);
    return group;
  }

  createRamp(object) {
    const geometry = rampRenderGeometry(object);
    const group = new THREE.Group();
    setWorldTransform(group, geometry.position, geometry.yaw);
    const mesh = new THREE.Mesh(
      boxGeometry(geometry.size),
      this.materials.woodDark,
    );
    mesh.rotation.z = geometry.angle;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const count = Math.max(3, Math.ceil(geometry.size[0] / 0.48));
    const spacing = geometry.size[0] / count;
    for (let index = 0; index < count; index += 1) {
      const board = new THREE.Mesh(
        new THREE.BoxGeometry(spacing * 0.9, 0.045, geometry.size[1] * 1.02),
        this.materials.wood,
      );
      board.position.set((index + 0.5) * spacing - geometry.size[0] * 0.5, geometry.size[2] * 0.5 + 0.018, 0);
      board.castShadow = true;
      board.receiveShadow = true;
      mesh.add(board);
    }
    group.add(mesh);
    return group;
  }

  createCrate(object) {
    const group = orientedGroup(object);
    const [width, depth, height] = object.size;
    const body = new THREE.Mesh(boxGeometry(object.size), this.materials.wood);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
    for (const x of [-width * 0.34, width * 0.34]) {
      const beam = new THREE.Mesh(new THREE.BoxGeometry(0.12, height * 1.06, depth * 1.05), this.materials.woodDark);
      beam.position.x = x;
      group.add(beam);
    }
    for (const z of [-depth * 0.34, depth * 0.34]) {
      const beam = new THREE.Mesh(new THREE.BoxGeometry(width * 1.05, height * 1.06, 0.12), this.materials.woodDark);
      beam.position.z = z;
      group.add(beam);
    }
    const cross = new THREE.Mesh(new THREE.BoxGeometry(width * 1.25, 0.1, 0.12), this.materials.woodDark);
    cross.rotation.y = Math.PI / 4;
    cross.position.y = height * 0.12;
    group.add(cross);
    return group;
  }

  createBarrel(object) {
    const group = orientedGroup(object);
    const radius = object.size[0] / 2;
    const height = object.size[2];
    const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.92, height, 18), this.materials.wood);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
    for (const y of [-height * 0.32, height * 0.32]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(radius * 1.02, 0.035, 8, 18), this.materials.woodDark);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = y;
      group.add(ring);
    }
    return group;
  }

  createBoulder(object) {
    const radius = Math.max(object.size[0], object.size[1], object.size[2]) / 2;
    const mesh = new THREE.Mesh(new THREE.DodecahedronGeometry(radius, 0), this.materials.stone);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    setWorldTransform(mesh, object.position, object.yaw);
    return mesh;
  }

  createAgent(object) {
    const group = baseGroup(object);
    const height = object.size[2];
    const radius = object.size[0] / 2;
    const legs = new THREE.Mesh(new THREE.BoxGeometry(radius * 1.15, height * 0.18, radius * 0.55), this.materials.agentDark);
    legs.position.y = height * 0.1;
    group.add(legs);
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(radius * 0.62, height * 0.38, 4, 10), this.materials.agent);
    body.position.y = height * 0.43;
    body.castShadow = true;
    group.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.66, 14, 10), this.materials.agent);
    head.position.y = height * 0.78;
    head.castShadow = true;
    group.add(head);
    const eyeGeometry = new THREE.SphereGeometry(radius * 0.13, 10, 8);
    for (const x of [-radius * 0.2, radius * 0.2]) {
      const eye = new THREE.Mesh(eyeGeometry, this.materials.agentFace);
      eye.position.set(x, height * 0.8, -radius * 0.66);
      eye.scale.set(0.9, 1.08, 0.42);
      group.add(eye);
    }
    const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, height * 0.22, 8), this.materials.agentDark);
    antenna.position.y = height * 1.03;
    group.add(antenna);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), this.materials.goal);
    dot.position.y = height * 1.16;
    group.add(dot);
    return group;
  }

  createGoal(object) {
    const group = baseGroup(object);
    const radius = Math.max(object.size[0], object.size[1]) * 0.36;
    const height = object.size[2];
    const base = new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.15, radius * 1.15, 0.06, 32), this.materials.goalZone);
    base.position.y = 0.06;
    group.add(base);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.055, 12, 36), this.materials.goal);
    ring.position.y = height * 0.62;
    ring.castShadow = true;
    group.add(ring);
    const core = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.45, 16, 10), this.materials.goalCore);
    core.position.y = height * 0.62;
    group.add(core);
    return group;
  }

  createGoalPad(object) {
    const group = baseGroup(object);
    const radius = Math.max(object.size[0], object.size[1]) * 0.42;
    const beaconHeight = Math.max(0.9, Number(object.size[2] || 1.2) * 0.82);
    const plinth = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.08, 0.1, 32), this.materials.goalPad);
    plinth.position.y = 0.05;
    plinth.receiveShadow = true;
    group.add(plinth);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(radius * 0.84, 0.045, 10, 36), this.materials.goal);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.12;
    group.add(rim);
    const inset = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.68, radius * 0.68, 0.025, 32), this.materials.goalSurface);
    inset.position.y = 0.115;
    group.add(inset);
    for (let index = 0; index < 3; index += 1) {
      const marker = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.035, radius * 0.32), this.materials.goalCore);
      marker.position.set((index - 1) * radius * 0.42, 0.145, 0);
      group.add(marker);
    }
    const beam = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.34, radius * 0.5, beaconHeight, 24, 1, true),
      this.materials.goalBeam,
    );
    beam.position.y = beaconHeight * 0.5 + 0.12;
    group.add(beam);
    for (const yaw of [0, Math.PI / 2]) {
      const halo = new THREE.Mesh(new THREE.TorusGeometry(radius * 0.62, 0.035, 8, 32), this.materials.goal);
      halo.position.y = beaconHeight * 0.72 + 0.12;
      halo.rotation.y = yaw;
      group.add(halo);
    }
    const beaconCore = new THREE.Mesh(new THREE.OctahedronGeometry(radius * 0.22, 0), this.materials.goalCore);
    beaconCore.position.y = beaconHeight + radius * 0.18;
    beaconCore.rotation.y = Math.PI / 4;
    group.add(beaconCore);
    return group;
  }

  createFloorSwitch(object) {
    const group = baseGroup(object);
    const [width, depth] = object.size;
    const base = new THREE.Mesh(new THREE.BoxGeometry(width, 0.08, depth), this.materials.switchBase);
    base.position.y = 0.04;
    base.receiveShadow = true;
    group.add(base);
    const top = new THREE.Mesh(new THREE.BoxGeometry(width * 0.74, 0.04, depth * 0.74), this.materials.switch);
    top.position.y = 0.1;
    group.add(top);
    return group;
  }

  createTargetRegion(object) {
    const group = baseGroup(object);
    const [width, depth] = object.size;
    const rail = Math.max(0.045, Math.min(width, depth) * 0.035);
    const edgeY = 0.045;
    const fill = new THREE.Mesh(new THREE.BoxGeometry(width * 0.94, 0.025, depth * 0.94), this.materials.targetFill);
    fill.position.y = 0.02;
    group.add(fill);
    for (const [x, z, sx, sz] of [
      [0, -depth * 0.5, width, rail],
      [0, depth * 0.5, width, rail],
      [-width * 0.5, 0, rail, depth],
      [width * 0.5, 0, rail, depth],
    ]) {
      const edge = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.055, sz), this.materials.target);
      edge.position.set(x, edgeY, z);
      group.add(edge);
    }
    const iconSize = Math.min(width, depth) * 0.28;
    const packageMark = new THREE.Mesh(
      new THREE.BoxGeometry(iconSize, 0.035, iconSize),
      this.materials.targetFill,
    );
    packageMark.position.y = 0.055;
    packageMark.rotation.y = Math.PI / 4;
    group.add(packageMark);
    for (const yaw of [Math.PI / 4, -Math.PI / 4]) {
      const slash = new THREE.Mesh(
        new THREE.BoxGeometry(iconSize * 1.15, 0.045, rail * 0.72),
        this.materials.target,
      );
      slash.position.y = 0.078;
      slash.rotation.y = yaw;
      group.add(slash);
    }
    return group;
  }

  createGate(object) {
    const group = orientedGroup(object);
    const [width, depth, height] = object.size;
    const alongX = width >= depth;
    const length = alongX ? width : depth;
    const postCount = Math.max(3, Math.ceil(length / 0.55));
    for (let index = 0; index < postCount; index += 1) {
      const t = index / Math.max(1, postCount - 1) - 0.5;
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.1, height, 0.1), this.materials.gate);
      post.position.set(alongX ? t * length : 0, 0, alongX ? 0 : t * length);
      post.castShadow = true;
      group.add(post);
    }
    for (const y of [-height * 0.35, 0, height * 0.35]) {
      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(alongX ? length : 0.12, 0.1, alongX ? 0.12 : length),
        this.materials.gateDark,
      );
      rail.position.y = y;
      group.add(rail);
    }
    return group;
  }

  createHazard(object) {
    return this.createDangerZone(object, "spikes");
  }

  createCourtyardHazard(object) {
    return this.createDangerZone(object, object.appearance?.variant || "puddle");
  }

  createDangerZone(object, variant) {
    const group = baseGroup(object);
    const [width, depth, height] = object.size;
    const shortSide = Math.min(width, depth);
    const frameWidth = Math.max(0.04, Math.min(0.14, shortSide * 0.13));
    const innerWidth = Math.max(0.08, width - frameWidth * 2);
    const innerDepth = Math.max(0.08, depth - frameWidth * 2);
    const base = new THREE.Mesh(new THREE.BoxGeometry(width, 0.045, depth), this.materials.hazardDark);
    base.position.y = Math.max(0.018, height * 0.16);
    base.receiveShadow = true;
    group.add(base);

    const dangerSurface = new THREE.Mesh(
      new THREE.BoxGeometry(innerWidth, 0.036, innerDepth),
      variant === "flowerbed" ? this.materials.soil : variant === "broken_paving" ? this.materials.hazardVoid : this.materials.hazardLiquid,
    );
    dangerSurface.position.y = Math.max(0.045, height * 0.22);
    dangerSurface.receiveShadow = true;
    group.add(dangerSurface);

    this.addCautionFrame(group, width, depth, frameWidth);

    if (variant === "flowerbed") {
      const count = Math.max(5, Math.round(width * depth * 1.8));
      for (let index = 0; index < count; index += 1) {
        const thorn = new THREE.Mesh(
          new THREE.ConeGeometry(Math.min(0.09, shortSide * 0.06), Math.min(0.34, shortSide * 0.22), 5),
          index % 3 ? this.materials.hazard : this.materials.hazardWarning,
        );
        thorn.position.set(
          ((index * 37) % 97) / 96 * innerWidth - innerWidth / 2,
          0.13,
          ((index * 53) % 89) / 88 * innerDepth - innerDepth / 2,
        );
        thorn.rotation.z = (index % 2 ? 1 : -1) * 0.18;
        thorn.castShadow = true;
        group.add(thorn);
      }
    } else if (variant === "broken_paving") {
      for (let index = 0; index < 6; index += 1) {
        const crack = new THREE.Mesh(new THREE.BoxGeometry(innerWidth * 0.3, 0.025, 0.04), this.materials.hazard);
        crack.position.set((index - 2.5) * innerWidth * 0.1, 0.078, (index % 2 ? 1 : -1) * innerDepth * 0.2);
        crack.rotation.y = (index - 2) * 0.31;
        group.add(crack);
      }
    } else if (variant === "spikes") {
      const countX = Math.max(1, Math.floor(innerWidth / 0.45));
      const countZ = Math.max(1, Math.floor(innerDepth / 0.45));
      for (let ix = 0; ix < countX; ix += 1) {
        for (let iz = 0; iz < countZ; iz += 1) {
          const spike = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.38, 4), this.materials.hazardMetal);
          spike.position.set((ix + 0.5) / countX * innerWidth - innerWidth / 2, 0.24, (iz + 0.5) / countZ * innerDepth - innerDepth / 2);
          spike.rotation.y = Math.PI / 4;
          spike.castShadow = true;
          group.add(spike);
        }
      }
    } else {
      const puddleParts = [
        [-0.23, -0.08, 0.42, 0.34],
        [0.2, 0.12, 0.38, 0.3],
        [0.05, -0.2, 0.3, 0.24],
      ];
      for (const [x, z, scaleX, scaleZ] of puddleParts) {
        const puddle = new THREE.Mesh(new THREE.CircleGeometry(0.5, 18), this.materials.puddle);
        puddle.rotation.x = -Math.PI / 2;
        puddle.position.set(x * innerWidth, 0.075, z * innerDepth);
        puddle.scale.set(innerWidth * scaleX, innerDepth * scaleZ, 1);
        group.add(puddle);
      }
    }

    if (shortSide >= 0.62) this.addHazardWarningMark(group, width, depth);
    this.addHazardBeacons(group, width, depth, frameWidth);
    return group;
  }

  addCautionFrame(group, width, depth, frameWidth) {
    const addStrip = (length, alongX, offset) => {
      const count = Math.max(2, Math.ceil(length / 0.32));
      const segmentLength = length / count;
      for (let index = 0; index < count; index += 1) {
        const segment = new THREE.Mesh(
          new THREE.BoxGeometry(alongX ? segmentLength + 0.004 : frameWidth, 0.04, alongX ? frameWidth : segmentLength + 0.004),
          index % 2 ? this.materials.hazardWarning : this.materials.hazardDark,
        );
        const along = -length / 2 + segmentLength * (index + 0.5);
        segment.position.set(alongX ? along : offset, 0.085, alongX ? offset : along);
        segment.receiveShadow = true;
        group.add(segment);
      }
    };
    addStrip(width, true, depth / 2 - frameWidth / 2);
    addStrip(width, true, -depth / 2 + frameWidth / 2);
    addStrip(depth - frameWidth * 2, false, width / 2 - frameWidth / 2);
    addStrip(depth - frameWidth * 2, false, -width / 2 + frameWidth / 2);
  }

  addHazardWarningMark(group, width, depth) {
    const radius = Math.min(width, depth) * 0.2;
    const triangle = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.032, 3), this.materials.hazardWarning);
    triangle.position.y = 0.105;
    triangle.rotation.y = Math.PI;
    group.add(triangle);
    const inset = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.72, radius * 0.72, 0.036, 3), this.materials.hazardDark);
    inset.position.y = 0.124;
    inset.rotation.y = Math.PI;
    group.add(inset);
    const stem = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.12, 0.04, radius * 0.64), this.materials.hazardWarning);
    stem.position.set(0, 0.15, radius * 0.05);
    group.add(stem);
    const dot = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.14, 0.04, radius * 0.14), this.materials.hazardWarning);
    dot.position.set(0, 0.15, -radius * 0.38);
    group.add(dot);
  }

  addHazardBeacons(group, width, depth, frameWidth) {
    if (Math.min(width, depth) < 0.75) return;
    const inset = frameWidth * 0.7;
    for (const [x, z] of [
      [-width / 2 + inset, -depth / 2 + inset],
      [width / 2 - inset, depth / 2 - inset],
    ]) {
      const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.075, 8), this.materials.hazardBeacon);
      beacon.position.set(x, 0.14, z);
      group.add(beacon);
    }
  }

  createTree(object) {
    const group = baseGroup(object);
    const height = object.size[2];
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.13, height * 0.45, 8), this.materials.woodDark);
    trunk.position.y = height * 0.22;
    trunk.castShadow = true;
    group.add(trunk);
    const variant = object.metadata?.variant || "round";
    if (variant === "pine") {
      for (let index = 0; index < 3; index += 1) {
        const cone = new THREE.Mesh(new THREE.ConeGeometry(object.size[0] * (0.55 - index * 0.08), height * 0.36, 9), this.materials.leaf);
        cone.position.y = height * (0.48 + index * 0.16);
        cone.castShadow = true;
        group.add(cone);
      }
    } else {
      const crown = new THREE.Mesh(new THREE.SphereGeometry(object.size[0] * 0.45, 13, 9), this.materials.leaf);
      crown.position.y = height * 0.68;
      crown.scale.set(1.05, variant === "two_tier" ? 0.85 : 1, 1.0);
      crown.castShadow = true;
      group.add(crown);
      if (variant === "two_tier") {
        const top = new THREE.Mesh(new THREE.SphereGeometry(object.size[0] * 0.32, 12, 8), this.materials.leafLight);
        top.position.y = height * 0.9;
        top.castShadow = true;
        group.add(top);
      }
    }
    return group;
  }

  createShrub(object) {
    const group = baseGroup(object);
    for (let index = 0; index < 4; index += 1) {
      const puff = new THREE.Mesh(new THREE.SphereGeometry(0.22 + index * 0.02, 10, 7), index % 2 ? this.materials.leaf : this.materials.leafLight);
      puff.position.set((index - 1.5) * 0.16, 0.2 + (index % 2) * 0.05, (index % 3 - 1) * 0.1);
      puff.scale.set(object.size[0], object.size[2], object.size[1]);
      puff.castShadow = true;
      group.add(puff);
    }
    return group;
  }

  createStone(object) {
    const mesh = new THREE.Mesh(new THREE.DodecahedronGeometry(0.5, 0), this.materials.stone);
    mesh.scale.set(object.size[0], object.size[2], object.size[1]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    setWorldTransform(mesh, [object.position[0], object.position[1], object.size[2] * 0.35], object.yaw);
    return mesh;
  }

  createSign(object) {
    const group = baseGroup(object);
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.08, object.size[2] * 0.78, 0.08), this.materials.woodDark);
    post.position.y = object.size[2] * 0.38;
    group.add(post);
    const board = new THREE.Mesh(new THREE.BoxGeometry(object.size[0], object.size[2] * 0.34, 0.08), this.materials.wood);
    board.position.y = object.size[2] * 0.78;
    board.castShadow = true;
    group.add(board);
    const mark = new THREE.Mesh(new THREE.BoxGeometry(object.size[0] * 0.45, 0.045, 0.09), this.materials.goal);
    mark.position.y = object.size[2] * 0.8;
    mark.position.z = -0.045;
    group.add(mark);
    return group;
  }

  createFallback(object) {
    const mesh = new THREE.Mesh(boxGeometry(object.size), this.materials.stone);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    setWorldTransform(mesh, object.position, object.yaw);
    return mesh;
  }

  resize() {
    const rect = this.fixedSize ? null : this.container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(this.fixedSize?.[0] || rect?.width || 1));
    const height = Math.max(1, Math.floor(this.fixedSize?.[1] || rect?.height || 1));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this.cameraPose) {
      this.camera.fov = this.cameraPose.fov;
      this.camera.updateProjectionMatrix();
      this.camera.position.copy(toThree(this.cameraPose.position));
      this.camera.lookAt(toThree(this.cameraPose.target));
      this.renderer.render(this.scene, this.camera);
      return;
    }
    const target = toThree(this.view.target);
    target.x += this.view.panX || 0;
    target.z += this.view.panY || 0;
    const azimuth = THREE.MathUtils.degToRad(this.view.azimuth);
    const elevation = THREE.MathUtils.degToRad(this.view.elevation);
    const distance = this.view.distance;
    const horizontal = Math.cos(elevation) * distance;
    this.camera.position.set(
      target.x + Math.sin(azimuth) * horizontal,
      target.y + Math.sin(elevation) * distance,
      target.z + Math.cos(azimuth) * horizontal,
    );
    this.camera.lookAt(target);
    this.renderer.render(this.scene, this.camera);
  }

  describeScreenSpace(physicsObjects = []) {
    this.render();
    this.camera.updateMatrixWorld(true);
    const surfaces = (this.visualScene?.objects || [])
      .filter((object) => isPlacementSurface(object))
      .sort((left, right) => placementSurfaceScore(right) - placementSurfaceScore(left));
    const primarySurface = surfaces[0] || null;
    const playableArea = primarySurface ? describePlayableArea(primarySurface, this.camera) : null;
    const regions = playableArea ? buildScreenRegions(playableArea, primarySurface, this.camera) : {};
    const objects = (physicsObjects || [])
      .filter((object) => object && Array.isArray(object.position) && Array.isArray(object.size))
      .map((object) => describeProjectedObject(object, this.camera));
    const target = toThree(this.view.target);
    target.x += this.view.panX || 0;
    target.z += this.view.panY || 0;
    return {
      camera: {
        projection: "perspective",
        fov_y_degrees: roundNumber(this.camera.fov),
        aspect: roundNumber(this.camera.aspect),
        near: roundNumber(this.camera.near),
        far: roundNumber(this.camera.far),
        position: roundVector(fromThree(this.camera.position)),
        target: roundVector(fromThree(target)),
        azimuth_degrees: roundNumber(this.view.azimuth),
        elevation_degrees: roundNumber(this.view.elevation),
        distance: roundNumber(this.view.distance),
      },
      playable_area: playableArea,
      regions,
      projected_objects: objects,
    };
  }

  async capturePng() {
    if (!this.captureMode) throw new Error("PNG capture requires capture mode");
    await this.assetsReady;
    this.render();
    const stats = this.capturePixelStats();
    if (stats.colorRange < 8 || stats.visiblePixels < 100) {
      throw new Error("Captured visual review frame was blank");
    }
    const blob = await new Promise((resolve, reject) => {
      this.renderer.domElement.toBlob(
        (value) => value ? resolve(value) : reject(new Error("Could not encode visual review frame")),
        "image/png",
      );
    });
    return { blob, stats };
  }

  capturePixelStats() {
    const width = this.renderer.domElement.width;
    const height = this.renderer.domElement.height;
    const context = this.renderer.getContext();
    const pixels = new Uint8Array(width * height * 4);
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let min = 255;
    let max = 0;
    let visiblePixels = 0;
    const stride = Math.max(4, Math.floor(pixels.length / 32000 / 4) * 4);
    for (let index = 0; index < pixels.length; index += stride) {
      if (pixels[index + 3] > 0) visiblePixels += 1;
      min = Math.min(min, pixels[index], pixels[index + 1], pixels[index + 2]);
      max = Math.max(max, pixels[index], pixels[index + 1], pixels[index + 2]);
    }
    return { width, height, colorRange: max - min, visiblePixels };
  }

  dispose() {
    this.sceneRevision += 1;
    this.resizeObserver?.disconnect();
    clearGroup(this.root);
    clearGroup(this.environment);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

class CourtyardAssetLibrary {
  constructor() {
    this.loader = new GLTFLoader();
    this.promises = new Map();
    this.scenes = new Map();
    this.errors = new Map();
  }

  async preload(objects) {
    const keys = [...new Set((objects || []).map((object) => courtyardAssetKey(object.appearance)).filter(Boolean))];
    await Promise.all(keys.map((key) => this.load(key)));
  }

  load(key) {
    const entry = COURTYARD_ASSETS[key];
    if (!entry?.url) return Promise.resolve(null);
    if (!this.promises.has(key)) {
      this.promises.set(key, new Promise((resolve) => {
        this.loader.load(
          entry.url,
          (gltf) => {
            this.scenes.set(key, gltf.scene);
            resolve(gltf.scene);
          },
          undefined,
          (error) => {
            this.errors.set(key, String(error?.message || error || "asset load failed"));
            resolve(null);
          },
        );
      }));
    }
    return this.promises.get(key);
  }

  clone(key) {
    return this.scenes.get(key)?.clone(true) || null;
  }
}

const courtyardAssetLibrary = new CourtyardAssetLibrary();

export function normalizeAsset(source, target, bottomAnchored) {
  source.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(source);
  let naturalSize = box.getSize(new THREE.Vector3());
  const rotateFootprint = (naturalSize.x >= naturalSize.z) !== (target.x >= target.z);
  if (rotateFootprint) {
    source.rotation.y += Math.PI / 2;
    source.updateMatrixWorld(true);
    box = new THREE.Box3().setFromObject(source);
    naturalSize = box.getSize(new THREE.Vector3());
  }
  source.scale.multiply(new THREE.Vector3(
    target.x / Math.max(MIN_SIZE, naturalSize.x),
    target.y / Math.max(MIN_SIZE, naturalSize.y),
    target.z / Math.max(MIN_SIZE, naturalSize.z),
  ));

  // glTF roots are not guaranteed to be centered on their geometry. Center
  // after scaling and rotation because an Object3D's own position is not
  // affected by its scale; doing this beforehand shifts the visible model
  // away from the physics collider.
  source.updateMatrixWorld(true);
  box = new THREE.Box3().setFromObject(source);
  const center = box.getCenter(new THREE.Vector3());
  source.position.x -= center.x;
  source.position.z -= center.z;
  source.position.y += bottomAnchored ? -box.min.y : -center.y;
  source.updateMatrixWorld(true);
  source.traverse((node) => {
    if (!node.isMesh) return;
    node.castShadow = true;
    node.receiveShadow = true;
    if (!node.material) return;
    const hasMaterialArray = Array.isArray(node.material);
    const materials = hasMaterialArray ? node.material : [node.material];
    const clones = materials.map((material) => {
      const clone = material.clone();
      clone.roughness = Math.max(0.62, Number(clone.roughness ?? 0.7));
      return clone;
    });
    node.material = hasMaterialArray ? clones : clones[0];
  });
}

function makeMaterials(palette) {
  const mat = (color, options = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.78, metalness: 0.02, flatShading: true, ...options });
  return {
    grass: mat(palette.grass || "#78C85C"),
    grassDark: mat(palette.grass_dark || "#4C9B48"),
    grassLight: mat(palette.grass_light || "#86DA68"),
    meadow: mat(palette.meadow || "#67B957"),
    meadowDark: mat(palette.meadow_dark || "#4E9A4F"),
    clearing: mat(palette.clearing || "#86D86C"),
    clearingEdge: mat(palette.grass_dark || "#3F8E4D", { transparent: true, opacity: 0.32 }),
    clearingPatch: mat(palette.clearing_alt || "#B4E987", { transparent: true, opacity: 0.22 }),
    path: mat(palette.path || "#D8B46A"),
    wood: mat(palette.wood || "#A96D3B"),
    woodDark: mat(palette.wood_dark || "#5D3525"),
    stone: mat(palette.stone || "#A9A8A2"),
    stoneDark: mat(palette.stone_dark || "#686D70"),
    leaf: mat(palette.leaf || "#4AA35A"),
    leafLight: mat(palette.leaf_light || "#7ED16A"),
    agent: mat(palette.agent || "#3A7BFF"),
    agentDark: mat("#1B356D"),
    agentFace: mat("#DDEFFF"),
    goal: mat(palette.goal || "#FFD166", { emissive: palette.goal || "#FFD166", emissiveIntensity: 0.35 }),
    goalCore: mat(palette.goal_core || "#66E3FF", { transparent: true, opacity: 0.78, emissive: palette.goal_core || "#66E3FF", emissiveIntensity: 0.6 }),
    goalBeam: new THREE.MeshBasicMaterial({ color: palette.goal_core || "#66E3FF", transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide }),
    goalZone: mat("#67F4B3", { transparent: true, opacity: 0.32 }),
    goalPad: mat("#59636F", { metalness: 0.12, roughness: 0.48 }),
    goalSurface: mat(palette.goal || "#FFD166", { emissive: palette.goal || "#FFD166", emissiveIntensity: 0.18, roughness: 0.48 }),
    target: mat(palette.target || "#70D6E8", { emissive: palette.target || "#70D6E8", emissiveIntensity: 0.28 }),
    targetFill: mat(palette.target || "#70D6E8", { transparent: true, opacity: 0.2, depthWrite: false }),
    switch: mat(palette.switch || "#4AC7E8", { emissive: palette.switch || "#4AC7E8", emissiveIntensity: 0.32 }),
    switchBase: mat("#4B5862", { metalness: 0.08, roughness: 0.58 }),
    gate: mat("#718274"),
    gateDark: mat("#40534A"),
    hazard: mat(palette.hazard || "#E24B5C"),
    hazardDark: mat("#20252A", { roughness: 0.92 }),
    hazardVoid: mat("#161B20", { roughness: 0.96 }),
    hazardLiquid: mat("#9E2636", { emissive: "#5C101B", emissiveIntensity: 0.28, roughness: 0.3 }),
    hazardWarning: mat("#FFD43B", { emissive: "#6B4C00", emissiveIntensity: 0.2, roughness: 0.58 }),
    hazardBeacon: mat("#FF3D4F", { emissive: "#FF172E", emissiveIntensity: 1.15, roughness: 0.3 }),
    hazardMetal: mat("#737D83", { metalness: 0.55, roughness: 0.35 }),
    puddle: mat("#4EA9C7", { transparent: true, opacity: 0.72, roughness: 0.28 }),
    soil: mat("#725239"),
  };
}

function orientedGroup(object) {
  const group = new THREE.Group();
  setWorldTransform(group, object.position, object.yaw);
  return group;
}

function baseGroup(object) {
  const group = new THREE.Group();
  const baseZ = object.physics_backed ? object.position[2] - object.size[2] / 2 : object.position[2];
  setWorldTransform(group, [object.position[0], object.position[1], baseZ], object.yaw);
  return group;
}

function setWorldTransform(node, position, yaw = 0) {
  const pos = toThree(position);
  node.position.copy(pos);
  node.rotation.y = -Number(yaw || 0);
}

function toThree(position) {
  return new THREE.Vector3(Number(position[0] || 0), Number(position[2] || 0), Number(position[1] || 0));
}

function finiteVector(value, length, label) {
  if (!Array.isArray(value) || value.length !== length) {
    throw new Error(`${label} must contain ${length} values`);
  }
  const result = value.map(Number);
  if (!result.every(Number.isFinite)) throw new Error(`${label} must be finite`);
  return result;
}

function fromThree(position) {
  return [Number(position.x || 0), Number(position.z || 0), Number(position.y || 0)];
}

function isPlacementSurface(object) {
  if (!object?.physics_backed || !Array.isArray(object.position) || !Array.isArray(object.size)) return false;
  const semantic = String(object.semantic_type || "").toLowerCase();
  const tags = new Set((object.tags || []).map((tag) => String(tag).toLowerCase()));
  return semantic === "ground" || semantic === "platform" || tags.has("walkable");
}

function placementSurfaceScore(object) {
  const semanticBonus = String(object.semantic_type || "").toLowerCase() === "ground" ? 1e6 : 0;
  return semanticBonus + Math.abs(Number(object.size?.[0] || 0) * Number(object.size?.[1] || 0));
}

function describePlayableArea(surface, camera) {
  const worldCorners = surfaceTopCorners(surface);
  const projected = worldCorners.map((point) => projectWorldPoint(point, camera));
  const visible = projected.filter((point) => point.in_front);
  if (visible.length < 3) return null;
  const polygon = visible.map((point) => point.uv.map((value) => roundNumber(clamp(value, 0, 1))));
  const bounds = uvBounds(polygon);
  if (!bounds || bounds.right - bounds.left < 0.05 || bounds.bottom - bounds.top < 0.05) return null;
  return {
    surface_id: String(surface.source_id || surface.id || ""),
    semantic_type: String(surface.semantic_type || ""),
    top_z: roundNumber(Number(surface.position[2] || 0) + Math.abs(Number(surface.size[2] || 0)) * 0.5),
    world_corners: worldCorners.map(roundVector),
    screen_polygon_uv: polygon,
    bounds_uv: roundBounds(bounds),
  };
}

function buildScreenRegions(playableArea, surface, camera) {
  const bounds = playableArea.bounds_uv;
  const width = Math.max(0.001, bounds.right - bounds.left);
  const height = Math.max(0.001, bounds.bottom - bounds.top);
  const usable = {
    left: bounds.left + width * 0.06,
    right: bounds.right - width * 0.06,
    top: bounds.top + height * 0.06,
    bottom: bounds.bottom - height * 0.06,
  };
  const usableWidth = usable.right - usable.left;
  const usableHeight = usable.bottom - usable.top;
  const columns = {
    left: [usable.left, usable.left + usableWidth / 3],
    center: [usable.left + usableWidth / 3, usable.left + 2 * usableWidth / 3],
    right: [usable.left + 2 * usableWidth / 3, usable.right],
  };
  const rows = {
    top: [usable.top, usable.top + usableHeight / 3],
    center: [usable.top + usableHeight / 3, usable.top + 2 * usableHeight / 3],
    bottom: [usable.top + 2 * usableHeight / 3, usable.bottom],
  };
  const definitions = {
    top_left: ["left", "top"],
    top_center: ["center", "top"],
    top_right: ["right", "top"],
    center_left: ["left", "center"],
    center: ["center", "center"],
    center_right: ["right", "center"],
    bottom_left: ["left", "bottom"],
    bottom_center: ["center", "bottom"],
    bottom_right: ["right", "bottom"],
  };
  return Object.fromEntries(Object.entries(definitions).map(([name, [column, row]]) => {
    const regionBounds = {
      left: columns[column][0],
      right: columns[column][1],
      top: rows[row][0],
      bottom: rows[row][1],
    };
    const desiredUv = [
      (regionBounds.left + regionBounds.right) * 0.5,
      (regionBounds.top + regionBounds.bottom) * 0.5,
    ];
    const anchorWorld = screenPointOnSurface(desiredUv, surface, camera);
    const projectedAnchor = projectWorldPoint(anchorWorld, camera);
    const verifiedBounds = {
      left: Math.max(bounds.left, Math.min(regionBounds.left, projectedAnchor.uv[0] - 0.015)),
      right: Math.min(bounds.right, Math.max(regionBounds.right, projectedAnchor.uv[0] + 0.015)),
      top: Math.max(bounds.top, Math.min(regionBounds.top, projectedAnchor.uv[1] - 0.015)),
      bottom: Math.min(bounds.bottom, Math.max(regionBounds.bottom, projectedAnchor.uv[1] + 0.015)),
    };
    return [name, {
      bounds_uv: roundBounds(verifiedBounds),
      anchor: {
        surface_id: playableArea.surface_id,
        world_position: roundVector(anchorWorld),
        screen_uv: projectedAnchor.uv.map(roundNumber),
      },
    }];
  }));
}

function screenPointOnSurface(uv, surface, camera) {
  const ndc = new THREE.Vector2(Number(uv[0]) * 2 - 1, 1 - Number(uv[1]) * 2);
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, camera);
  const topZ = Number(surface.position[2] || 0) + Math.abs(Number(surface.size[2] || 0)) * 0.5;
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -topZ);
  const hit = raycaster.ray.intersectPlane(plane, new THREE.Vector3()) || toThree(surface.position);
  return clampPointToSurface(fromThree(hit), surface, 0.82);
}

function clampPointToSurface(point, surface, insetScale) {
  const center = surface.position.map(Number);
  const yaw = Number(surface.yaw || 0);
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const dx = Number(point[0]) - center[0];
  const dy = Number(point[1]) - center[1];
  const localX = cos * dx + sin * dy;
  const localY = -sin * dx + cos * dy;
  const halfX = Math.abs(Number(surface.size[0] || 0)) * 0.5 * insetScale;
  const halfY = Math.abs(Number(surface.size[1] || 0)) * 0.5 * insetScale;
  const clampedX = clamp(localX, -halfX, halfX);
  const clampedY = clamp(localY, -halfY, halfY);
  return [
    center[0] + cos * clampedX - sin * clampedY,
    center[1] + sin * clampedX + cos * clampedY,
    center[2] + Math.abs(Number(surface.size[2] || 0)) * 0.5,
  ];
}

function surfaceTopCorners(surface) {
  const center = surface.position.map(Number);
  const halfX = Math.abs(Number(surface.size[0] || 0)) * 0.5;
  const halfY = Math.abs(Number(surface.size[1] || 0)) * 0.5;
  const topZ = center[2] + Math.abs(Number(surface.size[2] || 0)) * 0.5;
  const yaw = Number(surface.yaw || 0);
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  return [[-halfX, -halfY], [halfX, -halfY], [halfX, halfY], [-halfX, halfY]].map(([x, y]) => [
    center[0] + cos * x - sin * y,
    center[1] + sin * x + cos * y,
    topZ,
  ]);
}

function describeProjectedObject(object, camera) {
  const center = object.position.map(Number);
  const projectedCenter = projectWorldPoint(center, camera);
  const points = objectBoundsCorners(object).map((point) => projectWorldPoint(point, camera));
  const visiblePoints = points.filter((point) => point.in_front);
  const bounds = uvBounds(visiblePoints.map((point) => point.uv));
  return {
    id: String(object.id || ""),
    semantic_type: String(object.semantic_type || ""),
    center_world: roundVector(center),
    center_uv: projectedCenter.uv.map(roundNumber),
    bounds_uv: bounds ? roundBounds(bounds) : null,
    visible: Boolean(projectedCenter.in_front && bounds && bounds.right >= 0 && bounds.left <= 1 && bounds.bottom >= 0 && bounds.top <= 1),
  };
}

function objectBoundsCorners(object) {
  const center = object.position.map(Number);
  const half = object.size.map((value) => Math.abs(Number(value || 0)) * 0.5);
  const yaw = Number(object.yaw || 0);
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const corners = [];
  for (const x of [-half[0], half[0]]) {
    for (const y of [-half[1], half[1]]) {
      for (const z of [-half[2], half[2]]) {
        corners.push([
          center[0] + cos * x - sin * y,
          center[1] + sin * x + cos * y,
          center[2] + z,
        ]);
      }
    }
  }
  return corners;
}

function projectWorldPoint(worldPoint, camera) {
  const cameraSpace = toThree(worldPoint).applyMatrix4(camera.matrixWorldInverse);
  const projected = toThree(worldPoint).project(camera);
  return {
    uv: [(projected.x + 1) * 0.5, (1 - projected.y) * 0.5],
    in_front: cameraSpace.z < -camera.near,
  };
}

function uvBounds(points) {
  const finite = points.filter((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite));
  if (!finite.length) return null;
  return {
    left: Math.min(...finite.map((point) => point[0])),
    top: Math.min(...finite.map((point) => point[1])),
    right: Math.max(...finite.map((point) => point[0])),
    bottom: Math.max(...finite.map((point) => point[1])),
  };
}

function roundBounds(bounds) {
  return Object.fromEntries(Object.entries(bounds).map(([key, value]) => [key, roundNumber(value)]));
}

function roundVector(values) {
  return values.map(roundNumber);
}

function physicsRotationToThree(values) {
  const source = values.map(Number);
  const matrix = new THREE.Matrix4();
  matrix.set(
    source[0], source[2], source[1], 0,
    source[6], source[8], source[7], 0,
    source[3], source[5], source[4], 0,
    0, 0, 0, 1,
  );
  return new THREE.Quaternion().setFromRotationMatrix(matrix).normalize();
}

function boxGeometry(size) {
  return new THREE.BoxGeometry(
    Math.max(MIN_SIZE, Number(size[0] || MIN_SIZE)),
    Math.max(MIN_SIZE, Number(size[2] || MIN_SIZE)),
    Math.max(MIN_SIZE, Number(size[1] || MIN_SIZE)),
  );
}

function primaryTerrainObject(visualScene) {
  return (visualScene?.objects || []).find((object) => object.visual_type === "terrain_grass") || null;
}

function createSurroundingTerrain(terrain, materials) {
  const group = new THREE.Group();
  const [width, depth, height] = terrain.size;
  const centerX = Number(terrain.position[0] || 0);
  const centerZ = Number(terrain.position[1] || 0);
  const surfaceY = Number(terrain.position[2] || 0) + Number(height || 0) / 2;
  const margin = Math.max(6, Math.min(10, Math.max(width, depth) * 0.34));
  const thickness = 0.028;
  const y = surfaceY - thickness / 2 - 0.003;
  const bands = [
    { x: centerX, z: centerZ + depth / 2 + margin / 2, w: width + margin * 2, d: margin },
    { x: centerX, z: centerZ - depth / 2 - margin / 2, w: width + margin * 2, d: margin },
    { x: centerX + width / 2 + margin / 2, z: centerZ, w: margin, d: depth },
    { x: centerX - width / 2 - margin / 2, z: centerZ, w: margin, d: depth },
  ];
  for (const band of bands) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(band.w, thickness, band.d), materials.meadow);
    mesh.position.set(band.x, y, band.z);
    mesh.receiveShadow = true;
    group.add(mesh);
  }

  const skirtWidth = 0.32;
  const skirts = [
    { x: centerX, z: centerZ + depth / 2 + skirtWidth / 2, w: width, d: skirtWidth },
    { x: centerX, z: centerZ - depth / 2 - skirtWidth / 2, w: width, d: skirtWidth },
    { x: centerX + width / 2 + skirtWidth / 2, z: centerZ, w: skirtWidth, d: depth },
    { x: centerX - width / 2 - skirtWidth / 2, z: centerZ, w: skirtWidth, d: depth },
  ];
  for (const skirt of skirts) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(skirt.w, 0.018, skirt.d), materials.meadowDark);
    mesh.position.set(skirt.x, surfaceY + 0.004, skirt.z);
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  return group;
}

function gradientTexture(top, bottom) {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, top);
  gradient.addColorStop(1, bottom);
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  return texture;
}

function createSkyDome(palette, worldSize) {
  const extent = Math.max(12, Number(worldSize[0] || 12), Number(worldSize[1] || 12));
  const geometry = new THREE.SphereGeometry(extent * 5, 32, 16);
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      topColor: { value: new THREE.Color(palette.sky_top || "#9DDDF7") },
      bottomColor: { value: new THREE.Color(palette.sky_bottom || "#DDF2FF") },
    },
    vertexShader: `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 topColor;
      uniform vec3 bottomColor;
      varying vec3 vWorldPosition;
      void main() {
        float h = normalize(vWorldPosition).y;
        float t = smoothstep(-0.05, 0.78, h);
        gl_FragColor = vec4(mix(bottomColor, topColor, t), 1.0);
      }
    `,
  });
  const dome = new THREE.Mesh(geometry, material);
  dome.renderOrder = -1000;
  return dome;
}

function createSun(palette, worldSize) {
  const extent = Math.max(12, Number(worldSize[0] || 12), Number(worldSize[1] || 12));
  const material = new THREE.SpriteMaterial({
    map: radialTexture("#FFF3A6", palette.sky_bottom || "#DDF2FF"),
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
  });
  const sun = new THREE.Sprite(material);
  sun.position.set(-extent * 1.45, extent * 1.55, -extent * 1.25);
  sun.scale.set(extent * 0.48, extent * 0.48, 1);
  sun.renderOrder = -900;
  return sun;
}

function radialTexture(center, edge) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(64, 64, 4, 64, 64, 62);
  gradient.addColorStop(0, center);
  gradient.addColorStop(0.48, center);
  gradient.addColorStop(1, edge);
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  return new THREE.CanvasTexture(canvas);
}

function clearGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((node) => {
      if (node.geometry) node.geometry.dispose();
      if (node.material) {
        if (Array.isArray(node.material)) {
          for (const material of node.material) disposeMaterial(material);
        } else {
          disposeMaterial(node.material);
        }
      }
    });
  }
}

function disposeMaterial(material) {
  if (material.map) material.map.dispose();
  material.dispose();
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundNumber(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}
