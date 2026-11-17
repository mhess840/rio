// SPDX-FileCopyrightText: 2026 RIO Developers
// SPDX-License-Identifier: Apache-2.0

// WebXR view of rio scene layers. Plain WebGL2, no third-party code.
// Layers arrive in the robot base frame (metres, z up). In the headset they are drawn
// with the registration matrix xr_from_base; on the desktop with a z-up to y-up turn.

const HIDDEN_KEY = "rio.xr.visibility";
const REGISTRATION_KEY = "rio.xr.registration";
const ORDER_KEY = "rio.xr.order";
const MIN_ANCHORS = 3;

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl2", { xrCompatible: true, antialias: true, alpha: true });
const statusText = document.getElementById("status");
const enterButton = document.getElementById("enter");

const scene = { layers: new Map(), anchors: {}, version: 0 };
const gpu = new Map();
const visibility = JSON.parse(localStorage.getItem(HIDDEN_KEY) || "{}");
let registration = JSON.parse(localStorage.getItem(REGISTRATION_KEY) || "null");
let order = JSON.parse(localStorage.getItem(ORDER_KEY) || "[]");
let captured = [];
let xr = null;

// ---------- 4x4 matrices, column-major like WebGL and WebXR ----------

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

function fromRows(rows) {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = rows[r][c];
  return out;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function lookAt(eye, target, up) {
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

const IDENTITY = fromRows([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
const BASE_TO_YUP = fromRows([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]);

// ---------- GL program ----------

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, `#version 300 es
in vec3 a_pos;
uniform mat4 u_mvp;
uniform float u_point;
void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  gl_PointSize = u_point / max(gl_Position.w, 0.05);
}`));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
uniform vec4 u_color;
uniform bool u_round;
out vec4 color;
void main() {
  if (u_round) {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d, d) > 0.25) discard;
  }
  color = u_color;
}`));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
const loc = {
  pos: gl.getAttribLocation(program, "a_pos"),
  mvp: gl.getUniformLocation(program, "u_mvp"),
  point: gl.getUniformLocation(program, "u_point"),
  color: gl.getUniformLocation(program, "u_color"),
  round: gl.getUniformLocation(program, "u_round"),
};
const scratch = gl.createBuffer();

// ---------- layers on the GPU ----------

function upload(item) {
  const buffer = gl.createBuffer();
  if (item.type === "points" || item.type === "lines") {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(item.positions), gl.STATIC_DRAW);
    const mode = item.type === "points" ? gl.POINTS : gl.LINES;
    return { mode, buffer, count: item.positions.length / 3, color: item.color, size: item.size || 0.01 };
  }
  if (item.type === "mesh") {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(item.vertices), gl.STATIC_DRAW);
    const index = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(item.triangles), gl.STATIC_DRAW);
    return { mode: gl.TRIANGLES, buffer, index, count: item.triangles.length, color: item.color };
  }
  gl.deleteBuffer(buffer);
  return null;
}

function drop(name) {
  const old = gpu.get(name);
  if (!old) return;
  for (const item of old.items) {
    gl.deleteBuffer(item.buffer);
    if (item.index) gl.deleteBuffer(item.index);
  }
  gpu.delete(name);
}

function setLayer(layer) {
  drop(layer.name);
  scene.layers.set(layer.name, layer);
  gpu.set(layer.name, { items: layer.items.map(upload).filter(Boolean) });
  if (!(layer.name in visibility)) visibility[layer.name] = layer.visible;
  renderLayerList();
}

function removeLayer(name) {
  drop(name);
  scene.layers.delete(name);
  renderLayerList();
}

// ---------- drawing ----------

function drawItem(item, mvp, pxPerMetre) {
  gl.uniformMatrix4fv(loc.mvp, false, mvp);
  gl.uniform4fv(loc.color, item.color);
  gl.uniform1i(loc.round, item.mode === gl.POINTS ? 1 : 0);
  gl.uniform1f(loc.point, item.mode === gl.POINTS ? item.size * pxPerMetre : 1);
  gl.bindBuffer(gl.ARRAY_BUFFER, item.buffer);
  gl.enableVertexAttribArray(loc.pos);
  gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
  if (item.index) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, item.index);
    gl.drawElements(item.mode, item.count, gl.UNSIGNED_INT, 0);
  } else {
    gl.drawArrays(item.mode, 0, item.count);
  }
}

function drawLayers(viewProj, model, pxPerMetre) {
  const mvp = multiply(viewProj, model);
  for (const transparent of [false, true]) {
    gl.depthMask(!transparent);
    for (const [name, entry] of gpu) {
      if (!visibility[name]) continue;
      for (const item of entry.items) {
        if ((item.color[3] < 1) === transparent) drawItem(item, mvp, pxPerMetre);
      }
    }
  }
  gl.depthMask(true);
}

function drawPoints(points, color, size, mvp, pxPerMetre) {
  if (!points.length) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, scratch);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points.flat()), gl.DYNAMIC_DRAW);
  drawItem({ mode: gl.POINTS, buffer: scratch, count: points.length, color, size }, mvp, pxPerMetre);
}

function prepare(clearAlpha) {
  gl.useProgram(program);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.1, 0.1, 0.1, clearAlpha);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
}

// ---------- desktop preview: drag to orbit, wheel to zoom ----------

const orbit = { yaw: 0.6, pitch: 0.5, distance: 2.8, target: [0, 0.3, 0] };

canvas.addEventListener("pointermove", (event) => {
  if (!event.buttons) return;
  orbit.yaw -= event.movementX * 0.01;
  orbit.pitch = Math.min(1.5, Math.max(-1.5, orbit.pitch + event.movementY * 0.01));
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.distance = Math.min(10, Math.max(0.3, orbit.distance * Math.exp(event.deltaY * 0.001)));
}, { passive: false });

function desktopFrame() {
  if (xr) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  prepare(1);
  const { yaw, pitch, distance, target } = orbit;
  const eye = [
    target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
    target[1] + distance * Math.sin(pitch),
    target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
  ];
  const projection = perspective(0.9, width / Math.max(height, 1), 0.05, 50);
  const viewProj = multiply(projection, lookAt(eye, target, [0, 1, 0]));
  drawLayers(viewProj, BASE_TO_YUP, (projection[5] * height) / 2);
  requestAnimationFrame(desktopFrame);
}

// ---------- headset ----------

function registering() {
  return order.length >= MIN_ANCHORS && captured.length < order.length;
}

function qualityColor() {
  if (!registration) return [1, 1, 1, 1];
  if (registration.rms_m < 0.015) return [0.2, 1, 0.2, 1];
  return registration.rms_m < 0.03 ? [1, 0.85, 0.2, 1] : [1, 0.2, 0.2, 1];
}

async function finishRegistration() {
  const pairs = captured.map((point) => [scene.anchors[point.name], point.xr]);
  const response = await fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pairs }),
  });
  const result = await response.json();
  captured = [];
  if (!response.ok) {
    statusText.textContent = `registration failed: ${result.error}`;
    return;
  }
  registration = { rows: result.xr_from_base, rms_m: result.rms_m, space: xr ? xr.spaceType : "", at: new Date().toISOString() };
  localStorage.setItem(REGISTRATION_KEY, JSON.stringify(registration));
  order = [];
  localStorage.setItem(ORDER_KEY, "[]");
  renderRegistration();
}

function onSelect(event) {
  if (!registering()) return;
  const pose = event.frame.getPose(event.inputSource.targetRaySpace, xr.space);
  if (!pose) return;
  const p = pose.transform.position;
  captured.push({ name: order[captured.length], xr: [p.x, p.y, p.z] });
  if (captured.length === order.length) finishRegistration();
}

function xrFrame(time, frame) {
  const session = frame.session;
  session.requestAnimationFrame(xrFrame);
  const pose = frame.getViewerPose(xr.space);
  if (!pose) return;
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  prepare(xr.mode === "immersive-ar" ? 0 : 1);
  const tips = [];
  for (const source of session.inputSources) {
    const tip = frame.getPose(source.targetRaySpace, xr.space);
    if (tip) tips.push([tip.transform.position.x, tip.transform.position.y, tip.transform.position.z]);
  }
  const model = registration && !registering() ? fromRows(registration.rows) : null;
  for (const view of pose.views) {
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    const viewProj = multiply(view.projectionMatrix, view.transform.inverse.matrix);
    const pxPerMetre = (view.projectionMatrix[5] * viewport.height) / 2;
    if (model) {
      drawLayers(viewProj, model, pxPerMetre);
      drawPoints([[0, 0, 0]], qualityColor(), 0.03, multiply(viewProj, model), pxPerMetre);
    }
    drawPoints(tips, registering() ? [1, 0.85, 0.2, 1] : [1, 1, 1, 0.8], 0.012, viewProj, pxPerMetre);
    drawPoints(captured.map((point) => point.xr), [1, 0.5, 0, 1], 0.015, viewProj, pxPerMetre);
  }
}

async function enterXR() {
  const mode = (await navigator.xr.isSessionSupported("immersive-ar")) ? "immersive-ar" : "immersive-vr";
  const session = await navigator.xr.requestSession(mode, {
    requiredFeatures: ["local-floor"],
    optionalFeatures: ["bounded-floor"],
  });
  await gl.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  let space;
  let spaceType = "bounded-floor";
  try {
    space = await session.requestReferenceSpace("bounded-floor");
  } catch {
    spaceType = "local-floor";
    space = await session.requestReferenceSpace("local-floor");
  }
  xr = { session, space, spaceType, mode };
  captured = [];
  session.addEventListener("select", onSelect);
  session.addEventListener("squeeze", () => captured.pop());
  session.addEventListener("end", () => {
    xr = null;
    renderRegistration();
    requestAnimationFrame(desktopFrame);
  });
  session.requestAnimationFrame(xrFrame);
}

// ---------- page ----------

function renderLayerList() {
  const box = document.getElementById("layers");
  box.replaceChildren();
  for (const [name, layer] of scene.layers) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(visibility[name]);
    input.addEventListener("change", () => {
      visibility[name] = input.checked;
      localStorage.setItem(HIDDEN_KEY, JSON.stringify(visibility));
    });
    label.append(input, ` ${layer.title}`);
    box.append(label);
  }
}

function renderRegistration() {
  const text = document.getElementById("registration");
  if (registration) {
    const mm = Math.round(registration.rms_m * 1000);
    text.textContent = `registered: ${mm} mm RMS, ${registration.space}, ${registration.at.slice(0, 16)}`;
    if (registration.space !== "bounded-floor") text.textContent += " (redo after a new session)";
  } else {
    text.textContent = "not registered";
  }
  document.getElementById("order").textContent = order.length ? order.map((name, i) => `${i + 1}. ${name}`).join("  ") : "none";
  const box = document.getElementById("anchors");
  box.replaceChildren();
  for (const name of Object.keys(scene.anchors)) {
    const button = document.createElement("button");
    button.textContent = name;
    if (order.includes(name)) button.className = "chosen";
    button.addEventListener("click", () => {
      if (!order.includes(name)) order.push(name);
      localStorage.setItem(ORDER_KEY, JSON.stringify(order));
      renderRegistration();
    });
    box.append(button);
  }
}

document.getElementById("clear-order").addEventListener("click", () => {
  order = [];
  localStorage.setItem(ORDER_KEY, "[]");
  renderRegistration();
});
document.getElementById("reset").addEventListener("click", () => {
  registration = null;
  localStorage.removeItem(REGISTRATION_KEY);
  renderRegistration();
});
enterButton.addEventListener("click", () => {
  if (!registration && order.length < MIN_ANCHORS) {
    statusText.textContent = `choose at least ${MIN_ANCHORS} poses for the registration first`;
    return;
  }
  enterXR().catch((error) => { statusText.textContent = `XR failed: ${error.message}`; });
});

async function connect() {
  const response = await fetch("/scene");
  const data = await response.json();
  scene.anchors = data.anchors;
  for (const layer of data.layers) setLayer(layer);
  renderRegistration();
  const events = new EventSource(`/events?since=${data.version}`);
  events.addEventListener("changes", (event) => {
    const change = JSON.parse(event.data);
    for (const layer of change.layers) setLayer(layer);
    for (const name of change.removed) removeLayer(name);
  });
  events.onopen = () => { statusText.textContent = "live"; };
  events.onerror = () => { statusText.textContent = "reconnecting"; };
}

if (navigator.xr) {
  Promise.all([navigator.xr.isSessionSupported("immersive-ar"), navigator.xr.isSessionSupported("immersive-vr")])
    .then(([ar, vr]) => {
      enterButton.disabled = !(ar || vr);
      enterButton.textContent = ar ? "Enter AR" : "Enter VR";
    });
} else {
  enterButton.textContent = "WebXR not available";
}
connect().catch((error) => { statusText.textContent = `no scene: ${error.message}`; });
requestAnimationFrame(desktopFrame);
