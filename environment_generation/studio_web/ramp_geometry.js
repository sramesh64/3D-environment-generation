export function rampRenderGeometry(object) {
  const size = Array.isArray(object?.size) ? object.size : [1, 1, 0.25];
  const position = Array.isArray(object?.position) ? object.position : [0, 0, 0];
  const metadata = object?.metadata && typeof object.metadata === "object" ? object.metadata : {};
  const length = positiveNumber(size[0], 1);
  const width = positiveNumber(size[1], 1);
  const thickness = positiveNumber(size[2], 0.25);
  const rise = positiveNumber(metadata.rise ?? metadata.height, thickness);
  const yaw = finiteNumber(object?.yaw, 0);
  const angle = Math.atan2(rise, length);
  const slopeLength = Math.hypot(length, rise);
  const centerAlong = length * 0.5 + thickness * Math.sin(angle) * 0.5;
  const lowEnd = validVec3(metadata.low_end)
    ? metadata.low_end.map(Number)
    : [
        finiteNumber(position[0], 0) - Math.cos(yaw) * centerAlong,
        finiteNumber(position[1], 0) - Math.sin(yaw) * centerAlong,
        finiteNumber(position[2], 0) - rise * 0.5 + thickness * Math.cos(angle) * 0.5,
      ];
  return {
    position: [
      lowEnd[0] + Math.cos(yaw) * centerAlong,
      lowEnd[1] + Math.sin(yaw) * centerAlong,
      lowEnd[2] + rise * 0.5 - thickness * Math.cos(angle) * 0.5,
    ],
    size: [slopeLength, width, thickness],
    lowEnd,
    highEnd: [
      lowEnd[0] + Math.cos(yaw) * length,
      lowEnd[1] + Math.sin(yaw) * length,
      lowEnd[2] + rise,
    ],
    angle,
    yaw,
  };
}

function validVec3(value) {
  return Array.isArray(value) && value.length === 3 && value.every((item) => Number.isFinite(Number(item)));
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function positiveNumber(value, fallback) {
  const number = finiteNumber(value, fallback);
  return number > 0 ? number : fallback;
}
