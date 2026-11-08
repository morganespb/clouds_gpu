import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ---------- Settings ----------
const SIZE = 512;            // 512 = 262k particles, 1024 = 1M (needs a good GPU; restart to apply)
const COOL = new THREE.Color(0.42, 0.36, 0.34); // grey-brown smoke
const HOT = new THREE.Color(1.0, 0.27, 0.11);   // orange-red core

// Live settings: these are the starting values, change them with the sliders
const params = {
  flowSpeed: 0.3,    // how fast particles travel along the curl field
  noiseScale: 1.1,   // bigger = smaller, busier swirls
  rise: 0.02,        // upward drift, like smoke
  detail: 0.25,      // strength of the fine, wispy swirl layer
  shape: 4.0,        // how firmly particles are held inside the cloud shape
  lumps: 0.12,       // how lumpy / billowy the cloud's border is
  shell: 0.14,       // thickness of the surface layer (smaller = sharper edges)
  core: 0.3,         // share of particles that form the glowing core
  pointSize: 4.5,    // dot size
  opacity: 0.16,     // per-dot opacity (additive, so it adds up)
  heat: 0.7,         // how much of the cloud glows hot
  depth: 0.65,       // how much darker the back of the cloud is
  bloom: 0.45,       // glow strength
  grain: 0.06,       // film grain
  rotate: 0.25,      // camera auto-rotation speed
};

// Slider ranges: [key, label, min, max, step]
const SLIDERS = [
  ['flowSpeed', 'Speed', 0, 1, 0.01],
  ['noiseScale', 'Swirl size', 0.2, 4, 0.05],
  ['rise', 'Rise', -0.1, 0.2, 0.005],
  ['detail', 'Fine detail', 0, 1.5, 0.05],
  ['shape', 'Shape hold', 0, 10, 0.1],
  ['lumps', 'Lumpy border', 0, 0.5, 0.01],
  ['shell', 'Edge thickness', 0.02, 0.6, 0.01],
  ['core', 'Core amount', 0, 0.8, 0.01],
  ['pointSize', 'Dot size', 1, 20, 0.5],
  ['opacity', 'Opacity', 0.01, 0.5, 0.01],
  ['heat', 'Inner glow', 0, 1.5, 0.05],
  ['depth', 'Depth', 0, 1, 0.05],
  ['bloom', 'Glow', 0, 2, 0.05],
  ['grain', 'Film grain', 0, 0.3, 0.01],
  ['rotate', 'Rotation', -2, 2, 0.05],
];

// Colour modes: switch with the buttons or keys 1-6, H hides the buttons
const PRESETS = [
  { name: 'Ember',  cool: [0.42, 0.36, 0.34], hot: [1.00, 0.27, 0.11] },
  { name: 'Ice',    cool: [0.30, 0.36, 0.45], hot: [0.35, 0.75, 1.00] },
  { name: 'Nebula', cool: [0.32, 0.25, 0.45], hot: [1.00, 0.30, 0.75] },
  { name: 'Toxic',  cool: [0.25, 0.32, 0.28], hot: [0.55, 1.00, 0.35] },
  { name: 'Gold',   cool: [0.40, 0.36, 0.30], hot: [1.00, 0.72, 0.25] },
  { name: 'Mono',   cool: [0.35, 0.35, 0.35], hot: [0.95, 0.95, 0.95] },
];
// ----------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 5.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = params.rotate;

// ---------- Simplex + curl noise (GLSL) ----------
// 3D simplex noise: Ashima Arts / Stefan Gustavson, MIT License
// https://github.com/ashima/webgl-noise
const NOISE_GLSL = /* glsl */ `
vec3 mod289(vec3 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v){
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

vec3 snoiseVec3(vec3 x){
  return vec3(
    snoise(x),
    snoise(vec3(x.y - 19.1, x.z + 33.4, x.x + 47.2)),
    snoise(vec3(x.z + 74.2, x.x - 124.5, x.y + 99.4))
  );
}

// Curl noise: a swirling flow with no "sinks", so particles never clump into points
vec3 curlNoise(vec3 p){
  const float e = 0.1;
  vec3 dx = vec3(e, 0.0, 0.0);
  vec3 dy = vec3(0.0, e, 0.0);
  vec3 dz = vec3(0.0, 0.0, e);
  vec3 px0 = snoiseVec3(p - dx); vec3 px1 = snoiseVec3(p + dx);
  vec3 py0 = snoiseVec3(p - dy); vec3 py1 = snoiseVec3(p + dy);
  vec3 pz0 = snoiseVec3(p - dz); vec3 pz1 = snoiseVec3(p + dz);
  float x = py1.z - py0.z - pz1.y + pz0.y;
  float y = pz1.x - pz0.x - px1.z + px0.z;
  float z = px1.y - px0.y - py1.x + py0.x;
  return normalize(vec3(x, y, z) / (2.0 * e));
}
`;

// ---------- GPGPU: particle positions live in a texture ----------
const gpu = new GPUComputationRenderer(SIZE, SIZE, renderer);
const posTex = gpu.createTexture();
const originTex = gpu.createTexture();

// The cloud's shape: big billows + smaller bumps on top of them (cauliflower-like).
// Each blob = [x, y, z, radius]. Change SHAPE_SEED to get a different cloud.
const SHAPE_SEED = 7;
let seed = SHAPE_SEED;
const rand = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
const mainBlobs = [
  [0.0, 0.3, 0.0, 0.8], [-0.75, 0.0, 0.1, 0.65], [0.75, -0.1, -0.1, 0.72],
  [0.2, -0.7, 0.2, 0.7], [-0.35, 0.9, -0.15, 0.55], [0.6, 0.7, 0.1, 0.5],
  [-0.6, -0.65, -0.2, 0.58],
];
const blobs = [...mainBlobs];
for (const b of mainBlobs) {
  for (let k = 0; k < 2; k++) {
    // a random direction, pushed outwards from the cloud's centre
    let dx = rand() * 2 - 1 + b[0], dy = rand() * 2 - 1 + b[1], dz = rand() * 2 - 1 + b[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len; dy /= len; dz /= len;
    const r = b[3] * (0.35 + rand() * 0.2);
    blobs.push([b[0] + dx * b[3] * 0.85, b[1] + dy * b[3] * 0.85, b[2] + dz * b[3] * 0.85, r]);
  }
}

// shapeField() is > 0 inside the cloud, 0 on its surface, < 0 outside,
// and gets bigger the deeper you go. Used to hold the shape AND to colour the core.
const blobUniform = blobs.map((b) => new THREE.Vector4(b[0], b[1], b[2], b[3]));
const SHAPE_GLSL = /* glsl */ `
#define NB ${blobs.length}
uniform vec4 uBlobs[NB];
float shapeField(vec3 p, out vec3 toCentre){
  float best = -10.0;
  toCentre = vec3(0.0);
  for (int i = 0; i < NB; i++) {
    vec3 d = uBlobs[i].xyz - p;
    float f = 1.0 - length(d) / uBlobs[i].w;
    if (f > best) { best = f; toCentre = d; }
  }
  return best;
}
`;
// Pick blobs in proportion to their volume, so particles fill the shape evenly
const volumes = blobs.map((b) => b[3] ** 3);
const totalVol = volumes.reduce((a, v) => a + v, 0);
const pickBlob = () => {
  let r = Math.random() * totalVol;
  for (let k = 0; k < blobs.length; k++) { r -= volumes[k]; if (r <= 0) return blobs[k]; }
  return blobs[0];
};
const pd = posTex.image.data, od = originTex.image.data;
for (let i = 0; i < SIZE * SIZE; i++) {
  const b = pickBlob();
  // random point inside the blob's sphere
  const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2;
  const s = Math.sqrt(1 - u * u), rr = b[3] * 0.95 * Math.cbrt(Math.random());
  const x = b[0] + s * Math.cos(th) * rr;
  const y = b[1] + s * Math.sin(th) * rr;
  const z = b[2] + u * rr;
  od[i * 4] = pd[i * 4] = x;
  od[i * 4 + 1] = pd[i * 4 + 1] = y;
  od[i * 4 + 2] = pd[i * 4 + 2] = z;
  od[i * 4 + 3] = Math.random();      // seed
  pd[i * 4 + 3] = Math.random();      // life, staggered so respawns don't sync
}
posTex.needsUpdate = true;
originTex.needsUpdate = true;

const simShader = /* glsl */ `
uniform float uTime;
uniform float uDelta;
uniform sampler2D uOrigin;
uniform float uFlowSpeed;
uniform float uNoiseScale;
uniform float uRise;
uniform float uDetail;
uniform float uShape;
uniform float uLumps;
uniform float uShell;
uniform float uCoreFrac;
${NOISE_GLSL}
${SHAPE_GLSL}
void main(){
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 p = texture2D(texturePosition, uv);
  vec4 o = texture2D(uOrigin, uv);
  // Longer lives = you can follow each particle's path
  float life = p.w - uDelta * (0.05 + 0.06 * fract(o.w * 13.7));
  if (life <= 0.0) {
    p.xyz = o.xyz;            // respawn at its home in the cloud
    life = 1.0;
  } else {
    // Big, slow-changing swirl: clear, readable streams
    vec3 flow = curlNoise(p.xyz * uNoiseScale + vec3(0.0, 0.0, uTime * 0.015));
    // Second, smaller swirl layer: adds wisps and filaments
    if (uDetail > 0.001) {
      flow += uDetail * curlNoise(p.xyz * uNoiseScale * 2.7 - vec3(uTime * 0.03, 0.0, 0.0));
    }
    p.xyz += flow * uDelta * uFlowSpeed;
    p.y += uDelta * uRise;    // drift, like smoke

    // Hold the cloud's shape. f > 0 inside, 0 on the lumpy surface, < 0 outside.
    vec3 toCentre;
    float f = shapeField(p.xyz, toCentre)
            + uLumps * snoise(p.xyz * 2.2 + vec3(uTime * 0.02));
    vec3 inward = normalize(toCentre);
    bool isCore = o.w < uCoreFrac;
    if (isCore) {
      // Core particles stay deep inside: the glowing heart
      float target = 0.25;
      if (f < target) p.xyz += inward * (target - f) * uDelta * uShape;
    } else {
      // Shell particles live in a thin band under the surface:
      // pushed in when outside, pushed out when too deep.
      // This draws crisp billow edges with darker gaps between them.
      if (f < 0.0) p.xyz += inward * (-f) * uDelta * uShape;
      else if (f > uShell) p.xyz -= inward * (f - uShell) * uDelta * uShape * 0.5;
    }
  }
  gl_FragColor = vec4(p.xyz, life);
}
`;

const posVar = gpu.addVariable('texturePosition', simShader, posTex);
gpu.setVariableDependencies(posVar, [posVar]);
posVar.material.uniforms.uTime = { value: 0 };
posVar.material.uniforms.uDelta = { value: 0 };
posVar.material.uniforms.uOrigin = { value: originTex };
posVar.material.uniforms.uFlowSpeed = { value: params.flowSpeed };
posVar.material.uniforms.uNoiseScale = { value: params.noiseScale };
posVar.material.uniforms.uRise = { value: params.rise };
posVar.material.uniforms.uDetail = { value: params.detail };
posVar.material.uniforms.uShape = { value: params.shape };
posVar.material.uniforms.uLumps = { value: params.lumps };
posVar.material.uniforms.uBlobs = { value: blobUniform };
posVar.material.uniforms.uShell = { value: params.shell };
posVar.material.uniforms.uCoreFrac = { value: params.core };
const err = gpu.init();
if (err) console.error(err);

// Warm-up: let the smoke develop before the first frame
for (let i = 0; i < 180; i++) {
  posVar.material.uniforms.uDelta.value = 1 / 60;
  posVar.material.uniforms.uTime.value += 1 / 60;
  gpu.compute();
}

// ---------- Render the particles ----------
const count = SIZE * SIZE;
const geometry = new THREE.BufferGeometry();
const refs = new Float32Array(count * 2);
for (let i = 0; i < count; i++) {
  refs[i * 2] = ((i % SIZE) + 0.5) / SIZE;
  refs[i * 2 + 1] = (Math.floor(i / SIZE) + 0.5) / SIZE;
}
geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
geometry.setAttribute('reference', new THREE.BufferAttribute(refs, 2));

const material = new THREE.ShaderMaterial({
  uniforms: {
    uPositions: { value: null },
    uTime: { value: 0 },
    uSize: { value: params.pointSize },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uOpacity: { value: params.opacity },
    uCool: { value: COOL },
    uHot: { value: HOT },
    uHeat: { value: params.heat },
    uDepth: { value: params.depth },
    uLumps: { value: params.lumps },
    uBlobs: { value: blobUniform },
    uOrigin: { value: originTex },
    uCoreFrac: { value: params.core },
  },
  vertexShader: /* glsl */ `
    uniform sampler2D uPositions;
    uniform float uTime;
    uniform float uSize;
    uniform float uPixelRatio;
    uniform float uHeat;
    uniform float uLumps;
    uniform sampler2D uOrigin;
    uniform float uCoreFrac;
    attribute vec2 reference;
    varying float vHeat;
    varying float vAlpha;
    varying float vNear;
    ${NOISE_GLSL}
    ${SHAPE_GLSL}
    void main(){
      vec4 p = texture2D(uPositions, reference);
      vec4 mv = modelViewMatrix * vec4(p.xyz, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uSize * uPixelRatio / -mv.z;
      // Colour from inside: the deeper a particle is in the cloud shape,
      // the hotter it is. The surface stays cool and smoky.
      vec3 toCentre;
      float inside = shapeField(p.xyz, toCentre)
                   + uLumps * snoise(p.xyz * 2.2 + vec3(uTime * 0.02));
      // Core particles glow; shell particles only warm up where the shell is deep
      float isCore = step(texture2D(uOrigin, reference).w, uCoreFrac);
      float depthIn = mix(smoothstep(0.1, 0.5, inside) * 0.35, smoothstep(0.0, 0.35, inside), isCore);
      // A slow noise field breaks the core up so it flickers like embers,
      // with a bias towards the lower right like the reel
      float n = snoise(p.xyz * 1.3 + vec3(0.0, uTime * 0.04, uTime * 0.02));
      float bias = 0.6 + 0.4 * smoothstep(1.6, 0.0, length(p.xyz - vec3(0.3, -0.3, 0.2)));
      vHeat = depthIn * (0.55 + 0.45 * smoothstep(-0.3, 0.6, n)) * bias * uHeat * 1.5;
      // 1 near the camera, 0 at the back of the cloud
      vNear = smoothstep(7.4, 4.2, -mv.z);
      // Fade in after spawning, fade out before dying
      vAlpha = smoothstep(1.0, 0.85, p.w) * smoothstep(0.0, 0.2, p.w);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uOpacity;
    uniform float uDepth;
    uniform vec3 uCool;
    uniform vec3 uHot;
    varying float vHeat;
    varying float vAlpha;
    varying float vNear;
    void main(){
      float r = length(gl_PointCoord - 0.5);
      if (r > 0.5) discard;
      // Crisp centre with a short falloff: reads as grain, not fuzz
      float soft = smoothstep(0.5, 0.15, r);
      vec3 col = mix(uCool, uHot, clamp(vHeat, 0.0, 1.0));
      // Darker at the back: gives the cloud volume
      col *= mix(1.0 - uDepth, 1.0, vNear);
      gl_FragColor = vec4(col, soft * vAlpha * uOpacity);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});

const points = new THREE.Points(geometry, material);
points.frustumCulled = false;
scene.add(points);

// ---------- Post-processing: soft glow ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight), params.bloom, 0.6, 0.25
);
composer.addPass(bloomPass);

// Film grain + vignette: the cinematic finish
const filmPass = new ShaderPass({
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uGrain: { value: params.grain },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uGrain;
    varying vec2 vUv;
    float rand(vec2 co){ return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      float g = rand(vUv * 1000.0 + fract(uTime) * 100.0) - 0.5;
      // grain only where there is light, so the background stays pure black
      float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));
      c.rgb += g * uGrain * smoothstep(0.0, 0.12, lum);
      c.rgb = max(c.rgb, 0.0);
      float v = smoothstep(0.95, 0.35, length(vUv - 0.5));
      c.rgb *= mix(0.55, 1.0, v);
      gl_FragColor = c;
    }
  `,
});
composer.addPass(filmPass);
composer.addPass(new OutputPass());

// ---------- Colour modes UI ----------
const targetCool = COOL.clone();
const targetHot = HOT.clone();
const BTN = 'background:rgba(255,255,255,0.06);color:#ddd;border:1px solid rgba(255,255,255,0.18);border-radius:6px;padding:6px 10px;cursor:pointer;font:12px system-ui,sans-serif;';

// Colour buttons, bottom left
const ui = document.createElement('div');
ui.style.cssText = 'position:fixed;left:16px;bottom:16px;display:flex;gap:6px;flex-wrap:wrap;';
document.body.appendChild(ui);
const buttons = PRESETS.map((p, i) => {
  const b = document.createElement('button');
  b.textContent = `${i + 1} ${p.name}`;
  b.style.cssText = BTN;
  b.onclick = () => setPreset(i);
  ui.appendChild(b);
  return b;
});

// Slider panel, top right
const panel = document.createElement('div');
panel.style.cssText = 'position:fixed;right:16px;top:56px;width:220px;padding:12px;background:rgba(10,10,10,0.7);border:1px solid rgba(255,255,255,0.12);border-radius:8px;color:#ddd;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 72px);overflow:auto;';
document.body.appendChild(panel);
const defaults = { ...params };
const inputs = {};
for (const [key, label, min, max, step] of SLIDERS) {
  const row = document.createElement('label');
  row.style.cssText = 'display:flex;flex-direction:column;gap:4px;';
  const head = document.createElement('div');
  head.style.cssText = 'display:flex;justify-content:space-between;';
  const name = document.createElement('span');
  name.textContent = label;
  const val = document.createElement('span');
  val.style.opacity = '0.6';
  head.append(name, val);
  const input = document.createElement('input');
  input.type = 'range';
  input.min = min; input.max = max; input.step = step; input.value = params[key];
  input.style.width = '100%';
  const decimals = String(step).split('.')[1]?.length ?? 0;
  const show = () => { val.textContent = Number(params[key]).toFixed(decimals); };
  input.oninput = () => { params[key] = Number(input.value); show(); applyParams(); };
  show();
  row.append(head, input);
  panel.appendChild(row);
  inputs[key] = { input, show };
}
const reset = document.createElement('button');
reset.textContent = 'Reset';
reset.style.cssText = BTN;
reset.onclick = () => {
  Object.assign(params, defaults);
  for (const k in inputs) { inputs[k].input.value = params[k]; inputs[k].show(); }
  applyParams();
};
panel.appendChild(reset);

function applyParams() {
  posVar.material.uniforms.uFlowSpeed.value = params.flowSpeed;
  posVar.material.uniforms.uNoiseScale.value = params.noiseScale;
  posVar.material.uniforms.uRise.value = params.rise;
  posVar.material.uniforms.uDetail.value = params.detail;
  posVar.material.uniforms.uShape.value = params.shape;
  posVar.material.uniforms.uLumps.value = params.lumps;
  material.uniforms.uLumps.value = params.lumps;
  posVar.material.uniforms.uShell.value = params.shell;
  posVar.material.uniforms.uCoreFrac.value = params.core;
  material.uniforms.uCoreFrac.value = params.core;
  material.uniforms.uSize.value = params.pointSize;
  material.uniforms.uOpacity.value = params.opacity;
  material.uniforms.uHeat.value = params.heat;
  material.uniforms.uDepth.value = params.depth;
  bloomPass.strength = params.bloom;
  filmPass.uniforms.uGrain.value = params.grain;
  controls.autoRotateSpeed = params.rotate;
}

// Hide / show button, top right (H hides everything, this button included)
const toggle = document.createElement('button');
toggle.textContent = 'Hide controls';
toggle.style.cssText = BTN + 'position:fixed;right:16px;top:16px;';
document.body.appendChild(toggle);
let controlsVisible = true;
function setControlsVisible(v) {
  controlsVisible = v;
  ui.style.display = v ? 'flex' : 'none';
  panel.style.display = v ? 'flex' : 'none';
  toggle.textContent = v ? 'Hide controls' : 'Show controls';
}
toggle.onclick = () => setControlsVisible(!controlsVisible);

// Start with everything hidden (clean view on reload). Press H to show the controls.
setControlsVisible(false);
toggle.style.display = 'none';

// Keep keyboard shortcuts from firing while dragging a slider
panel.addEventListener('keydown', (e) => e.stopPropagation());
function setPreset(i) {
  targetCool.setRGB(...PRESETS[i].cool);
  targetHot.setRGB(...PRESETS[i].hot);
  buttons.forEach((b, k) => { b.style.borderColor = k === i ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.18)'; });
}
setPreset(0);
window.addEventListener('keydown', (e) => {
  const n = Number(e.key);
  if (n >= 1 && n <= PRESETS.length) setPreset(n - 1);
  if (e.key === 'h' || e.key === 'H') {
    const allHidden = toggle.style.display === 'none';
    toggle.style.display = allHidden ? '' : 'none';
    setControlsVisible(allHidden);
  }
});

// ---------- Capture: P = high-res PNG, R = start / stop video ----------
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function captureStill() {
  const base = Math.min(window.devicePixelRatio, 2);
  const scale = Math.min(4, 4096 / Math.max(window.innerWidth, window.innerHeight));
  renderer.setPixelRatio(scale);
  composer.setPixelRatio(scale);
  material.uniforms.uPixelRatio.value = scale;
  composer.render();
  renderer.domElement.toBlob((b) => download(b, `clouds-${Date.now()}.png`), 'image/png');
  renderer.setPixelRatio(base);
  composer.setPixelRatio(base);
  material.uniforms.uPixelRatio.value = base;
}

const recDot = document.createElement('div');
recDot.textContent = '● REC';
recDot.style.cssText = 'position:fixed;left:16px;top:16px;color:#ff3b30;font:12px system-ui,sans-serif;display:none;';
document.body.appendChild(recDot);
let recorder = null;
function toggleRecording() {
  if (recorder) { recorder.stop(); return; }
  const types = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'];
  const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || '';
  const chunks = [];
  recorder = new MediaRecorder(renderer.domElement.captureStream(60), {
    mimeType, videoBitsPerSecond: 25_000_000,
  });
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const ext = mimeType.startsWith('video/mp4') ? 'mp4' : 'webm';
    download(new Blob(chunks, { type: mimeType || 'video/webm' }), `clouds-${Date.now()}.${ext}`);
    recorder = null;
    recDot.style.display = 'none';
  };
  recorder.start();
  recDot.style.display = 'block';
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'p' || e.key === 'P') captureStill();
  if (e.key === 'r' || e.key === 'R') toggleRecording();
});

// ---------- Loop ----------
const clock = new THREE.Clock();
function tick() {
  const dt = Math.min(clock.getDelta(), 1 / 30);
  posVar.material.uniforms.uDelta.value = dt;
  posVar.material.uniforms.uTime.value += dt;
  gpu.compute();

  material.uniforms.uPositions.value = gpu.getCurrentRenderTarget(posVar).texture;
  material.uniforms.uTime.value += dt;
  filmPass.uniforms.uTime.value += dt;
  // Smooth fade towards the chosen colour mode
  material.uniforms.uCool.value.lerp(targetCool, 0.04);
  material.uniforms.uHot.value.lerp(targetHot, 0.04);

  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
tick();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});
