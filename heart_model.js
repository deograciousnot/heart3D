// A schematic anatomical heart for the 3D views, built from simple shapes so every surface
// point knows which wall it belongs to.
//
//   left ventricle (LV)   bullet-shaped, apex down-left-forward
//   right ventricle (RV)  a shell wrapping the LV's front and right side
//   atria                 two rounded chambers at the base, behind and to the right
//   great vessels         aorta (arching left and back), pulmonary trunk, venae cavae
//
// It is drawn as a translucent "hologram": a soft rim at the edges, plus
//   - glow:  the part of the surface the heart vector points at lights up, in the phase colour
//            (atria during P, ventricles during QRS and T). It shows where the wave is heading,
//            not a measured activation map.
//   - scar:  LV walls ('anterior', 'inferior', 'lateral') can be shaded to show a likely MI site.
//
// Coordinates match the viewers: scene x = patient's left, y = up, z = toward the viewer (front).
// The heart's long axis points from the base (up, right, back) to the apex (down, left, front).

const APEX_DIR = [0.62, -0.5, 0.6]; // scene coordinates; Frank (left, feet, back) = (0.62, 0.5, -0.6)

const vertexShader = /* glsl */ `
  attribute float activation;
  attribute vec3 wall;
  uniform vec3 uScarMask;
  varying vec3 vN;
  varying vec3 vV;
  varying float vAct;
  varying float vScar;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vAct = activation;
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
  varying vec3 vN;
  varying vec3 vV;
  varying float vAct;
  varying float vScar;
  void main() {
    float facing = abs(dot(normalize(vN), normalize(vV)));
    float rim = pow(1.0 - facing, 2.2);
    vec3 col = uBase * (0.45 + 0.55 * rim) + uRim * rim * 0.8;
    float a = uAlpha + rim * uRimAlpha;
    col += uGlow * vAct * 1.15;
    a += vAct * 0.5;
    float pulse = 0.7 + 0.3 * sin(uTime * 3.2);
    col = mix(col, uScar, vScar * 0.85 * pulse);
    a += vScar * 0.35;
    gl_FragColor = vec4(col, clamp(a, 0.0, 0.92));
  }
`;

export function createHeart(THREE, { scale = 1.15 } = {}) {
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const group = new THREE.Group();

  // Local frame: +Y toward the base, +Z as close to the patient's front as possible, X = Y x Z.
  const apex = V(...APEX_DIR).normalize();
  const ly = apex.clone().negate();
  const front = V(0, 0, 1);
  const lz = front.clone().sub(ly.clone().multiplyScalar(front.dot(ly))).normalize();
  const lx = ly.clone().cross(lz);
  const basis = new THREE.Matrix4().makeBasis(lx, ly, lz);
  const toScene = (x, y, z) => V(x, y, z).applyMatrix4(basis).multiplyScalar(scale);
  // Centre of the ventricles (local y = -0.2) sits at the origin, where the heart vector starts.
  const offset = toScene(0, -0.2, 0).negate();

  const uniforms = {
    uBase: { value: new THREE.Color(0xff6b6b) },
    uRim: { value: new THREE.Color(0x7dd3fc) },
    uGlow: { value: new THREE.Color(0xff7a59) },
    uScar: { value: new THREE.Color(0xfb7185) },
    uAlpha: { value: 0.07 },
    uRimAlpha: { value: 0.42 },
    uScarMask: { value: new THREE.Vector3(0, 0, 0) },
    uTime: { value: 0 },
  };
  const material = new THREE.ShaderMaterial({
    uniforms, vertexShader, fragmentShader,
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });

  const parts = []; // { mesh, role, dirs (unit vectors from the part's centre, scene space) }

  /** Geometry built in local coordinates -> placed in the scene frame. */
  function addLocal(geometry, role, { position = [0, 0, 0], size = [1, 1, 1], centre } = {}) {
    const m = new THREE.Matrix4()
      .makeTranslation(offset.x, offset.y, offset.z)
      .multiply(new THREE.Matrix4().makeScale(scale, scale, scale))
      .multiply(basis)
      .multiply(new THREE.Matrix4().makeTranslation(...position))
      .multiply(new THREE.Matrix4().makeScale(...size));
    geometry.applyMatrix4(m);
    return addScene(geometry, role, centre);
  }

  /** centre: where "the direction the wave is heading" is measured from (default: heart centre). */
  function addScene(geometry, role, centre = V(0, 0, 0)) {
    geometry.computeVertexNormals();
    const n = geometry.attributes.position.count;
    geometry.setAttribute("activation", new THREE.Float32BufferAttribute(new Float32Array(n), 1));
    geometry.setAttribute("wall", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = 1;
    group.add(mesh);
    const dirs = new Float32Array(n * 3), p = geometry.attributes.position;
    for (let i = 0; i < n; i++) {
      const d = V(p.getX(i), p.getY(i), p.getZ(i)).sub(centre).normalize();
      dirs.set([d.x, d.y, d.z], i * 3);
    }
    const part = { mesh, role, dirs };
    parts.push(part);
    return part;
  }

  /** Bullet / dome profile turned around local Y. */
  function lathe(yMin, yMax, radius, power, phiStart = 0, phiLength = Math.PI * 2) {
    const pts = [];
    for (let k = 0; k <= 36; k++) {
      const t = k / 36;
      pts.push(new THREE.Vector2(Math.max(0.002, radius * Math.pow(Math.sin(t * Math.PI / 2), power)), yMin + (yMax - yMin) * t));
    }
    return new THREE.LatheGeometry(pts, 56, phiStart, phiLength);
  }

  // Ventricles. In LatheGeometry, phi = 0 faces +Z (front) and phi = -pi/2 faces -X.
  const lv = addLocal(lathe(-1.0, 0.55, 0.46, 0.62), "ventricle");
  addLocal(lathe(-0.72, 0.5, 0.6, 0.7, -1.9, 2.7), "ventricle", { position: [-0.04, 0, 0.05] });
  // Atria at the base: right atrium to the right, left atrium behind. Their glow is measured from
  // the middle of the two atria, since the P wave starts there and spreads down and left.
  const base = (x, y, z) => toScene(x, y, z).add(offset);
  const atriaCentre = base(-0.05, 0.72, -0.17);
  addLocal(new THREE.SphereGeometry(1, 40, 28), "atrium", { position: [-0.32, 0.72, 0.02], size: [0.3, 0.32, 0.3], centre: atriaCentre });
  addLocal(new THREE.SphereGeometry(1, 40, 28), "atrium", { position: [0.22, 0.72, -0.36], size: [0.34, 0.25, 0.3], centre: atriaCentre });

  // Great vessels, routed in scene space from points on the base.
  const tube = (start, steps, radius) => {
    const pts = [start, ...steps.map(([x, y, z]) => start.clone().add(V(x, y, z).multiplyScalar(scale)))];
    addScene(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 72, radius * scale, 16, false), "vessel");
  };
  const aortaRoot = base(0.06, 0.55, 0.12);
  tube(aortaRoot, [[-0.06, 0.45, 0.05], [0.08, 0.78, -0.18], [0.32, 0.74, -0.5], [0.4, 0.25, -0.64], [0.4, -0.55, -0.64]], 0.11);
  const pulmRoot = base(-0.06, 0.5, 0.48);
  tube(pulmRoot, [[0.04, 0.32, 0.02], [0.14, 0.55, -0.14], [0.42, 0.56, -0.34]], 0.088);
  tube(pulmRoot.clone().add(V(0.14, 0.55, -0.14).multiplyScalar(scale)), [[-0.22, 0.06, -0.12], [-0.46, 0.04, -0.2]], 0.07);
  const raCentre = base(-0.32, 0.72, 0.02);
  tube(raCentre.clone().add(V(0, 0.2, 0).multiplyScalar(scale)), [[0, 0.4, 0], [0.02, 0.75, -0.02]], 0.08);
  tube(raCentre.clone().add(V(0, -0.22, -0.04).multiplyScalar(scale)), [[0.02, -0.35, -0.03], [0.02, -0.7, -0.05]], 0.085);

  // LV walls, as soft weights so a shaded wall fades into its neighbours.
  // Directions are taken around the LV's long axis: the front (plus the septum, toward the RV),
  // the bottom (plus the back), and the patient's left side.
  {
    const rvSide = toScene(-1, 0, 1).normalize();
    const anteriorDir = V(0, 0, 1).add(rvSide).normalize();
    const inferiorDir = V(0, -1, -0.5).normalize();
    const lateralDir = V(1, 0.15, -0.2).normalize();
    const walls = lv.mesh.geometry.attributes.wall, p = lv.mesh.geometry.attributes.position;
    const axisPoint = offset.clone();
    const ramp = d => Math.min(1, Math.max(0, (d - 0.3) / 0.45));
    for (let i = 0; i < p.count; i++) {
      const v = V(p.getX(i), p.getY(i), p.getZ(i)).sub(axisPoint);
      const perp = v.sub(apex.clone().multiplyScalar(v.dot(apex))).normalize();
      walls.setXYZ(i, ramp(perp.dot(anteriorDir)), ramp(perp.dot(inferiorDir)), ramp(perp.dot(lateralDir)));
    }
    walls.needsUpdate = true;
  }

  const WALLS = ["anterior", "inferior", "lateral"];
  const glowColor = new THREE.Color();
  const tmp = V(0, 0, 0);
  let glowOn = true;

  return {
    group,
    /** Per frame: v = current heart vector (scene coords), strength 0..1 (its length vs the biggest),
        phase 0-3, colour of that phase, time in ms. */
    update(v, strength, phase, color, time) {
      uniforms.uTime.value = time / 1000;
      glowColor.copy(color);
      uniforms.uGlow.value.copy(glowColor);
      const len = v.length();
      if (len > 1e-6) tmp.copy(v).divideScalar(len);
      for (const part of parts) {
        const weight = !glowOn || len < 1e-6 ? 0
          : part.role === "atrium" ? (phase === 1 ? 1 : 0)
          : part.role === "ventricle" ? (phase === 2 ? 1 : phase === 3 ? 0.6 : 0)
          : 0;
        const act = part.mesh.geometry.attributes.activation;
        if (weight === 0) {
          if (part.lit) { act.array.fill(0); act.needsUpdate = true; part.lit = false; }
          continue;
        }
        // Brightest where the surface points along the heart vector, fading to nothing at ~80 degrees off.
        const s = Math.min(1, strength * 1.4) * weight, dirs = part.dirs;
        for (let i = 0, n = act.count; i < n; i++) {
          const d = dirs[i * 3] * tmp.x + dirs[i * 3 + 1] * tmp.y + dirs[i * 3 + 2] * tmp.z;
          act.array[i] = d > 0.2 ? Math.pow((d - 0.2) / 0.8, 2) * s : 0;
        }
        act.needsUpdate = true;
        part.lit = true;
      }
    },
    /** Colours from the page theme. */
    setTheme({ base, rim, scar, dark }) {
      uniforms.uBase.value.copy(base);
      uniforms.uRim.value.copy(rim);
      uniforms.uScar.value.copy(scar);
      uniforms.uAlpha.value = dark ? 0.07 : 0.12;
      uniforms.uRimAlpha.value = dark ? 0.42 : 0.5;
    },
    /** Shade LV walls, e.g. ["inferior"]; [] clears. */
    setScar(walls) {
      uniforms.uScarMask.value.set(...WALLS.map(w => (walls.includes(w) ? 1 : 0)));
    },
    setGlow(on) { glowOn = on; },
    setOpacity(alpha, rimAlpha) { uniforms.uAlpha.value = alpha; uniforms.uRimAlpha.value = rimAlpha; },
  };
}
