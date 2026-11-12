// A real heart for the 3D views: the average healthy adult heart from 19 CT scans
// (Rodero, Strocchi et al. 2021, CC BY 4.0), simplified by build_heart_mesh.py into
// assets/heart_mesh.json, with a simulated activation wave.
//
// Same interface as the schematic heart in heart_model.js, so the pages can fall back to it:
//   group, update(...), setTheme(...), setScar(walls), setGlow(on), setOpacity(a, rim)
// plus setScenario("normal" | "lbbb" | "rbbb") and `wave` in update():
//   wave = { mode, progress }
//     mode 1 = P wave (atria activating), 2 = QRS / ST (ventricles activating, then all active),
//     3 = T wave (ventricles recovering), 0 = quiet
//     progress = 0 at the start of that phase, 1 at its end (beyond 1 = phase over)
//
// The wave is simulated (a simplified eikonal model on the real anatomy), not this patient's
// measured activation. Timing is stretched to the phase durations of the recording shown.

const vertexShader = /* glsl */ `
  attribute float act;      // activation time, 0..1 across the phase
  attribute float repol;    // recovery time, 0..1 across the T wave
  attribute float role;     // 0 vessel, 1 atrium, 2 ventricle
  attribute vec3 wall;      // LV walls: front + septum, bottom, left side
  uniform vec3 uScarMask;
  varying vec3 vN;
  varying vec3 vV;
  varying float vAct;
  varying float vRepol;
  varying float vRole;
  varying float vScar;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vAct = act; vRepol = repol; vRole = role;
    vScar = clamp(dot(wall, uScarMask), 0.0, 1.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uRim;
  uniform vec3 uGlow;
  uniform vec3 uScar;
  uniform float uAlpha;
  uniform float uRimAlpha;
  uniform float uTime;
  uniform float uMode;
  uniform float uProgress;
  varying vec3 vN;
  varying vec3 vV;
  varying float vAct;
  varying float vRepol;
  varying float vRole;
  varying float vScar;
  void main() {
    float facing = abs(dot(normalize(vN), normalize(vV)));
    float rim = pow(1.0 - facing, 2.0);
    float vessel = vRole < 0.5 ? 1.0 : 0.0;
    vec3 col = uBase * (0.4 + 0.6 * rim) + uRim * rim * 0.75;
    float a = uAlpha * (1.0 - 0.4 * vessel) + rim * uRimAlpha;

    // Activation: a bright band at the wavefront, tissue behind it stays softly lit.
    float lit = 0.0;
    bool atrialPhase = uMode > 0.5 && uMode < 1.5 && vRole > 0.5 && vRole < 1.5;
    bool ventPhase = uMode > 1.5 && uMode < 2.5 && vRole > 1.5;
    if (atrialPhase || ventPhase) {
      float d = uProgress - vAct;                       // > 0: already activated
      float front = exp(-pow(d / 0.06, 2.0)) * step(-0.04, d);
      float behind = smoothstep(0.0, 0.12, d) * 0.32;
      lit = front + behind;
    }
    // Recovery during the T wave: activated tissue goes dark as it recovers.
    if (uMode > 2.5 && vRole > 1.5) {
      lit = 0.32 * (1.0 - smoothstep(-0.02, 0.1, uProgress - vRepol));
    }
    col += uGlow * lit * 1.25;
    a += lit * 0.55;

    float pulse = 0.7 + 0.3 * sin(uTime * 3.2);
    col = mix(col, uScar, vScar * 0.85 * pulse);
    a += vScar * 0.35;
    gl_FragColor = vec4(col, clamp(a, 0.0, 0.92));
  }
`;

const decode = (b64, Type) => {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
};

export async function loadRealHeart(THREE, url = "assets/heart_mesh.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`heart mesh: HTTP ${res.status}`);
  const data = await res.json();

  const uniforms = {
    uBase: { value: new THREE.Color(0xff6b6b) },
    uRim: { value: new THREE.Color(0x7dd3fc) },
    uGlow: { value: new THREE.Color(0xff7a59) },
    uScar: { value: new THREE.Color(0xfbbf24) },
    uAlpha: { value: 0.05 },
    uRimAlpha: { value: 0.36 },
    uScarMask: { value: new THREE.Vector3(0, 0, 0) },
    uTime: { value: 0 },
    uMode: { value: 0 },
    uProgress: { value: 0 },
  };
  const material = new THREE.ShaderMaterial({
    uniforms, vertexShader, fragmentShader,
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });

  const group = new THREE.Group();
  const ventricles = [];
  for (const part of data.parts) {
    const g = new THREE.BufferGeometry();
    const n = part.count;
    g.setAttribute("position", new THREE.BufferAttribute(decode(part.positions, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(decode(part.indices, part.index32 ? Uint32Array : Uint16Array), 1));
    g.computeVertexNormals();
    const role = ["LV", "RV"].includes(part.name) ? 2 : ["LA", "RA"].includes(part.name) ? 1 : 0;
    const u8 = key => (part[key] ? Float32Array.from(decode(part[key], Uint8Array), x => x / 255) : new Float32Array(n));
    g.setAttribute("role", new THREE.BufferAttribute(new Float32Array(n).fill(role), 1));
    g.setAttribute("act", new THREE.BufferAttribute(role === 1 ? u8("atria") : u8("normal"), 1));
    g.setAttribute("repol", new THREE.BufferAttribute(u8("repol"), 1));
    g.setAttribute("wall", new THREE.BufferAttribute(part.walls ? u8("walls") : new Float32Array(n * 3), 3));
    const mesh = new THREE.Mesh(g, material);
    mesh.renderOrder = 1;
    group.add(mesh);
    if (role === 2) ventricles.push({ g, scenarios: { normal: u8("normal"), lbbb: u8("lbbb"), rbbb: u8("rbbb") } });
  }

  const WALLS = ["anterior", "inferior", "lateral"];
  let glowOn = true, scenario = "normal";
  return {
    group,
    real: true,
    credit: data.credit,
    simulatedMs: data.scenarios,
    /** Which conduction pattern drives the ventricular wave. */
    setScenario(name) {
      if (!ventricles[0]?.scenarios[name] || name === scenario) return;
      scenario = name;
      for (const v of ventricles) {
        v.g.getAttribute("act").array.set(v.scenarios[name]);
        v.g.getAttribute("act").needsUpdate = true;
      }
    },
    update(v, strength, phase, color, time, wave) {
      uniforms.uTime.value = time / 1000;
      if (color) uniforms.uGlow.value.copy(color);
      uniforms.uMode.value = glowOn && wave ? wave.mode : 0;
      uniforms.uProgress.value = wave ? wave.progress : 0;
    },
    setTheme({ base, rim, scar, dark }) {
      uniforms.uBase.value.copy(base);
      uniforms.uRim.value.copy(rim);
      uniforms.uScar.value.copy(scar);
      uniforms.uAlpha.value = dark ? 0.05 : 0.09;
      uniforms.uRimAlpha.value = dark ? 0.36 : 0.45;
    },
    setScar(walls) { uniforms.uScarMask.value.set(...WALLS.map(w => (walls.includes(w) ? 1 : 0))); },
    setGlow(on) { glowOn = on; },
    setOpacity(alpha, rimAlpha) { uniforms.uAlpha.value = alpha; uniforms.uRimAlpha.value = rimAlpha; },
  };
}
