export const COURTYARD_ASSETS = Object.freeze({
  "courtyard_boundary:fence": asset("kenney_nature/fence_simple.glb", ["wall"]),
  "courtyard_boundary:hedge": asset("kenney_platformer/hedge.glb", ["wall"]),
  "courtyard_boundary:stone": fallback(["wall"]),
  "courtyard_boundary:plain": fallback(["wall"]),
  "courtyard_static_prop:planter": asset("kenney_platformer/plant.glb", ["static_box"]),
  "courtyard_static_prop:bench": fallback(["static_box"]),
  "courtyard_static_prop:crate": asset("kenney_platformer/crate.glb", ["static_box"]),
  courtyard_pushable_crate: asset("kenney_platformer/crate.glb", ["pushable_box"]),
  courtyard_barrel: asset("kenney_platformer/barrel.glb", ["cylinder"]),
  courtyard_floor_switch: asset("kenney_platformer/button-square.glb", ["floor_switch"]),
  courtyard_platform: asset("kenney_platformer/platform.glb", ["platform"]),
  courtyard_ramp: asset("kenney_platformer/platform-ramp.glb", ["ramp"]),
  courtyard_gate: asset("kenney_nature/fence_gate.glb", ["gate"]),
  courtyard_sign: asset("kenney_platformer/sign.glb", ["decoration"]),
  courtyard_tree: asset("kenney_nature/tree_default.glb", ["decoration"]),
  courtyard_tree_pine: asset("kenney_nature/tree_pineRoundA.glb", ["decoration"]),
  courtyard_shrub: asset("kenney_nature/plant_bushDetailed.glb", ["decoration"]),
  courtyard_rock: asset("kenney_nature/rock_smallA.glb", ["decoration"]),
  courtyard_flower: asset("kenney_nature/flower_redA.glb", ["decoration"]),
  courtyard_ground: fallback(["ground"]),
  courtyard_robot: fallback(["agent"]),
  courtyard_goal_pad: fallback(["goal"]),
  courtyard_hazard: fallback(["hazard"]),
});

export function courtyardAssetKey(appearance) {
  if (!appearance?.asset_id) return "";
  const variant = appearance.variant ? `:${appearance.variant}` : "";
  const exact = `${appearance.asset_id}${variant}`;
  if (COURTYARD_ASSETS[exact]) return exact;
  return COURTYARD_ASSETS[appearance.asset_id] ? appearance.asset_id : "";
}

function asset(filename, semantics) {
  return Object.freeze({
    url: `/assets/courtyard/${filename}`,
    semantics: Object.freeze(semantics),
    anchor: "bottom",
  });
}

function fallback(semantics) {
  return Object.freeze({ url: "", semantics: Object.freeze(semantics), anchor: "center" });
}
