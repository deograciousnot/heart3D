"""
Build the web heart (assets/heart_mesh.json) from a real, CT-derived four-chamber heart mesh,
and simulate how the electrical wave spreads through it.

Source: the average healthy adult heart of Rodero, Strocchi et al. (2021), built from 19 CT scans,
"Virtual cohort of extreme and average four-chamber heart meshes from statistical shape model",
Zenodo 10.5281/zenodo.4593739, licence CC BY 4.0. Put average.tar.gz in mesh_source/, extract it
to mesh_source/extracted/average.vtk, then:

    python build_heart_mesh.py

Steps:
  1. read the tetrahedral mesh (379k points, 1.77M tetrahedra, a tag per element, muscle fibre
     direction per element, universal ventricular coordinates per point); cached as average.npz
  2. orient it like a heart in the chest (apex down-left-forward, right ventricle in front)
  3. simulate activation times on the volume mesh (simulate_activation): normal conduction,
     left bundle branch block, right bundle branch block, and the atria
  4. take the surface of each part, simplify it for the browser, carry the times and the
     left-ventricle wall labels over, and write a compact JSON (~1-2 MB)

The wave is a simple "eikonal" model: activation spreads from early sites along the mesh, faster
along muscle fibres than across them, with a fast inner lining standing in for the Purkinje
network. Real research simulators (e.g. openCARP) solve the same idea more fully. It shows a
plausible sequence, not this patient's measured activation.
"""

import base64
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mesh_source")
VTK = os.path.join(SRC, "extracted", "average.vtk")
CACHE = os.path.join(SRC, "average.npz")
OUT = os.path.join(HERE, "assets", "heart_mesh.json")

# Element tags (from the dataset) grouped into the parts the viewer draws.
PARTS = {
    "LV": [1, 7],                        # left ventricle myocardium (+ mitral valve plane)
    "RV": [2, 8],                        # right ventricle (+ tricuspid valve plane)
    "LA": [3, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22],  # left atrium, appendage, pulmonary veins
    "RA": [4, 16, 17, 23, 24],           # right atrium, venae cavae
    "aorta": [5, 9],
    "pulmonary": [6, 10],
}
TRIANGLES = {"LV": 16000, "RV": 11000, "LA": 7000, "RA": 6500, "aorta": 4000, "pulmonary": 3500}

# Conduction velocities, mm per ms (= m/s).
V_FIBRE, V_CROSS = 0.6, 0.25     # ventricular muscle along / across fibres
V_FAST = 2.0                     # fast endocardial layer (Purkinje surrogate)
# Bundle branch block: the blocked side's network is only reached through the septum and conducts
# partly; tuned so the whole ventricles take ~150 ms (LBBB) and ~140 ms (RBBB), typical QRS widths.
V_BLOCKED = {"lbbb": 1.0, "rbbb": 1.3}
V_ATRIUM = 0.9                   # atrial muscle (isotropic here)
FAST_RHO, FAST_Z = 0.1, 0.85     # fast layer: inner 10% of the wall, not the top 15% near the valves

APEX_DIR = np.array([0.62, -0.5, 0.6])   # scene coords (x patient's left, y up, z front), as in heart_model.js
HEART_LENGTH = 1.55                        # base-to-apex length of the LV in scene units (whole heart ~2.6 tall)


# ---------------------------------------------------------------------------------------------
# 1. Reading
# ---------------------------------------------------------------------------------------------
def read_vtk(path):
    """ASCII VTK unstructured grid -> points (mm), tets, tags, fibres per element, UVCs per point."""
    import pandas as pd

    starts = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if line[:1].isalpha():
                parts = line.split()
                word = parts[0].decode()
                if word in ("POINTS", "CELLS", "CELL_DATA", "POINT_DATA", "SCALARS", "VECTORS"):
                    name = parts[1].decode() if word in ("SCALARS", "VECTORS") else word
                    starts[name] = (i, [p.decode() for p in parts])
    n_pts, n_cells = int(starts["POINTS"][1][1]), int(starts["CELLS"][1][1])

    def read(first, n, cols):
        return pd.read_csv(path, sep=r"\s+", header=None, skiprows=first, nrows=n,
                           usecols=range(cols), engine="c").to_numpy()

    pts = read(starts["POINTS"][0] + 1, n_pts, 3).astype(np.float32)
    cells = read(starts["CELLS"][0] + 1, n_cells, 5)[:, 1:].astype(np.int32)
    tags = read(starts["ID"][0] + 2, n_cells, 1)[:, 0].astype(np.int16)
    fibres = read(starts["fibres"][0] + 1, n_cells, 3).astype(np.float32)
    uvc = {key.split(".")[0].lower(): read(starts[key][0] + 2, n_pts, 1)[:, 0].astype(np.float32)
           for key in ("RHO.dat", "PHI.dat", "Z.dat", "V.dat")}
    return pts, cells, tags, fibres, uvc


def load():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if "fibres" in d:
            return d["pts"], d["cells"], d["tags"], d["fibres"], {k: d[k] for k in ("rho", "phi", "z", "v")}
    t = time.time()
    pts, cells, tags, fibres, uvc = read_vtk(VTK)
    np.savez_compressed(CACHE, pts=pts, cells=cells, tags=tags, fibres=fibres, **uvc)
    print(f"Read {os.path.basename(VTK)}: {len(pts)} points, {len(cells)} tetrahedra in {time.time() - t:.0f} s")
    return pts, cells, tags, fibres, uvc


# ---------------------------------------------------------------------------------------------
# 2. Orientation
# ---------------------------------------------------------------------------------------------
def unit(v):
    return v / np.linalg.norm(v)


def orientation(pts, cells, tags, uvc):
    """Rotation + centre that put the mesh in the viewer's frame, from anatomical landmarks."""
    centroid = lambda t: pts[np.unique(cells[np.isin(tags, t)])].mean(axis=0)
    lv, rv, la, ra, mitral = centroid([1]), centroid([2]), centroid([3]), centroid([4]), centroid([7])
    apex = pts[(uvc["v"] < 0) & (uvc["v"] > -5) & (uvc["z"] < 0.02)].mean(axis=0)
    axis = unit(apex - mitral)                                    # base -> apex
    to_rv = rv - lv
    side = unit(to_rv - axis * to_rv.dot(axis))                   # LV -> RV, across the long axis

    target_axis = unit(APEX_DIR)
    want = np.array([-0.55, 0.0, 1.0])                            # RV sits in front and to the patient's right
    target_side = unit(want - target_axis * want.dot(target_axis))

    def rotation(mirror):
        m = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        a, s = m @ axis, m @ side
        src = np.stack([a, s, np.cross(a, s)], axis=1)
        dst = np.stack([target_axis, target_side, np.cross(target_axis, target_side)], axis=1)
        return dst @ src.T @ m

    # The source frame's handedness is unknown, so try both and keep the one that's anatomically right:
    # left atrium behind the right atrium, right atrium to the patient's right of the left atrium.
    best = None
    for mirror in (False, True):
        r = rotation(mirror)
        la_s, ra_s, lv_s = r @ (la - lv), r @ (ra - lv), np.zeros(3)
        score = int(la_s[2] < lv_s[2]) + int(ra_s[0] < la_s[0])
        if best is None or score > best[0]:
            best = (score, mirror, r)
    score, mirror, r = best
    length = np.linalg.norm(apex - mitral)
    print(f"Orientation: mirrored={mirror}, anatomy checks passed {score}/2, LV length {length:.0f} mm")
    return r, lv, HEART_LENGTH / length


# ---------------------------------------------------------------------------------------------
# 3. Activation
# ---------------------------------------------------------------------------------------------
def edge_graph(pts, cells, tags, fibres, keep_tags, fast=None, isotropic_v=None, partial=None):
    """
    Undirected graph over the mesh edges of the chosen tetrahedra, weighted by travel time (ms).
    Ventricles: anisotropic (fibre direction per tetrahedron); fast-layer edges use V_FAST;
    partial = (mask, speed) for a slower, partly working fast layer (the blocked side in BBB).
    """
    from scipy.sparse import coo_matrix

    sel = np.isin(tags, keep_tags)
    c, f = cells[sel], fibres[sel]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    a = np.concatenate([c[:, i] for i, _ in pairs])
    b = np.concatenate([c[:, j] for _, j in pairs])
    fib = np.concatenate([f] * 6)
    d = pts[b] - pts[a]
    length = np.linalg.norm(d, axis=1)
    if isotropic_v:
        t = length / isotropic_v
    else:
        cos = np.abs((d * fib).sum(1)) / np.maximum(length * np.linalg.norm(fib, axis=1), 1e-9)
        v = 1.0 / np.sqrt((cos / V_FIBRE) ** 2 + ((1 - cos ** 2) / V_CROSS ** 2))  # ellipse of speeds
        t = length / v
        if partial is not None:
            mask, speed = partial
            both = mask[a] & mask[b]
            t[both] = np.minimum(t[both], length[both] / speed)
        if fast is not None:
            both = fast[a] & fast[b]
            t[both] = length[both] / V_FAST
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * len(pts) + hi
    order = np.argsort(key, kind="stable")
    key, t = key[order], t[order]
    first = np.r_[True, key[1:] != key[:-1]]
    tmin = np.minimum.reduceat(t, np.nonzero(first)[0])          # an edge shared by tets: fastest wins
    k = key[first]
    n = len(pts) + 1                                              # + a virtual source node
    return coo_matrix((np.maximum(tmin, 1e-6), (k // len(pts), k % len(pts))), shape=(n, n)).tocsr(), n - 1


def run_wave(graph, src_node, sources):
    """Earliest arrival from several (node set, start time ms) sources, via a virtual source node."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    rows, cols, vals = [], [], []
    for nodes, t0 in sources:
        rows += [src_node] * len(nodes)
        cols += list(nodes)
        vals += [max(t0, 1e-3)] * len(nodes)
    g = graph + csr_matrix((vals, (rows, cols)), shape=graph.shape)
    return dijkstra(g, directed=False, indices=src_node)


def simulate_activation(pts, cells, tags, fibres, uvc, anterior_sign):
    """Activation time (ms) per mesh point for each scenario; NaN where a wave doesn't reach."""
    rho, phi, z, v = uvc["rho"], uvc["phi"], uvc["z"], uvc["v"]
    vent = v > -5
    lvp, rvp = vent & (v < 0), vent & (v > 0)
    endo = vent & (rho < FAST_RHO)
    th = phi * anterior_sign                                       # > 0 toward the front of the heart
    pick = lambda m, name: (np.nonzero(m)[0], name)

    # Early sites (Durrer et al., 1970): three left-sided septal/paraseptal areas, then the RV a little later.
    lv_septal = pick(lvp & endo & (np.abs(th) < 0.35) & (z > 0.4) & (z < 0.6), "LV mid-septum")
    lv_ant = pick(lvp & endo & (th > 0.5) & (th < 1.1) & (z > 0.6) & (z < 0.75), "LV high anterior paraseptal")
    lv_post = pick(lvp & endo & (th < -0.5) & (th > -1.1) & (z > 0.25) & (z < 0.4), "LV low posterior paraseptal")
    rv_ant = pick(rvp & endo & (th > 0.3) & (z > 0.3) & (z < 0.5), "RV anterior wall (papillary muscle base)")
    for nodes, name in (lv_septal, lv_ant, lv_post, rv_ant):
        print(f"  source {name}: {len(nodes)} points")
        assert len(nodes) > 0, name

    fast_lv, fast_rv = endo & lvp & (z < FAST_Z), endo & rvp & (z < FAST_Z)
    lv_sources = [(lv_septal[0], 0), (lv_ant[0], 0), (lv_post[0], 0)]
    scenarios = {  # name: (working fast layer, partly working one, early sites)
        "normal": (fast_lv | fast_rv, None, lv_sources + [(rv_ant[0], 5)]),
        "lbbb": (fast_rv, (fast_lv, V_BLOCKED["lbbb"]), [(rv_ant[0], 0)]),   # starts in the RV, LV reached via the septum
        "rbbb": (fast_lv, (fast_rv, V_BLOCKED["rbbb"]), lv_sources),         # starts in the LV, RV reached late
    }
    times = {}
    for name, (fast, partial, sources) in scenarios.items():
        g, s = edge_graph(pts, cells, tags, fibres, [1, 2], fast=fast, partial=partial)
        t = run_wave(g, s, sources)[:-1]
        t[~vent | ~np.isfinite(t)] = np.nan
        times[name] = t
        print(f"  {name:6}: ventricles fully activated after {np.nanmax(t):.0f} ms "
              f"(10-90% in {np.nanpercentile(t, 90) - np.nanpercentile(t, 10):.0f} ms)")

    # Atria: from the sinus node, where the superior vena cava meets the right atrium.
    atrial = np.zeros(len(pts), bool)
    atrial[np.unique(cells[np.isin(tags, [3, 4])])] = True
    svc = pts[np.unique(cells[tags == 23])].mean(axis=0)
    d = np.linalg.norm(pts - svc, axis=1)
    sa = np.nonzero(atrial & (d < np.min(d[atrial]) + 6))[0]
    g, s = edge_graph(pts, cells, tags, fibres, [3, 4], isotropic_v=V_ATRIUM)
    ta = run_wave(g, s, [(sa, 0)])[:-1]
    ta[~atrial | ~np.isfinite(ta)] = np.nan
    times["atria"] = ta
    print(f"  atria : fully activated after {np.nanmax(ta):.0f} ms from {len(sa)} sinus-node points")

    # Repolarisation (for the T wave): activation + action potential duration, shorter toward the
    # outer wall and the base, so the last areas to activate recover first (an upright T wave).
    apd = 260 - 40 * rho - 25 * z
    times["repol"] = np.where(vent, times["normal"] + apd, np.nan)
    return times


# ---------------------------------------------------------------------------------------------
# 4. Surfaces
# ---------------------------------------------------------------------------------------------
def boundary_faces(pts, cells):
    """Triangles on the surface of a set of tetrahedra, wound to face outward."""
    faces = np.concatenate([cells[:, [1, 2, 3]], cells[:, [0, 3, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 1]]])
    opposite = np.concatenate([cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]])
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = counts[inverse.ravel()] == 1
    faces, opposite = faces[once], opposite[once]
    p0, p1, p2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    inward = (np.cross(p1 - p0, p2 - p0) * (pts[opposite] - p0)).sum(1) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    import fast_simplification
    from scipy.spatial import cKDTree

    t0 = time.time()
    pts, cells, tags, fibres, uvc = load()
    rot, centre, scale = orientation(pts, cells, tags, uvc)
    scene = ((pts - centre) @ rot.T * scale).astype(np.float32)

    # Which sign of the angle around the LV points to the front of the heart (scene +z)?
    lvp = (uvc["v"] < 0) & (uvc["v"] > -5)
    front = lambda m: scene[lvp & m, 2].mean()
    anterior_sign = 1.0 if front(uvc["phi"] > 0.8) > front(uvc["phi"] < -0.8) else -1.0
    print(f"Front of the LV is at {'positive' if anterior_sign > 0 else 'negative'} angle")

    print("Simulating activation...")
    times = simulate_activation(pts, cells, tags, fibres, uvc, anterior_sign)

    # LV wall weights for MI shading: front wall + septum, bottom (inferior/posterior), left side (lateral).
    th = uvc["phi"] * anterior_sign
    soft = lambda x, a, b, w=0.3: np.clip((x - a) / w + 0.5, 0, 1) * np.clip((b - x) / w + 0.5, 0, 1)
    in_lv = (lvp & (uvc["z"] < 0.95)).astype(np.float32)
    walls = np.stack([soft(th, -0.4, 2.1), soft(th, -2.3, -0.4),
                      np.maximum(soft(th, 2.1, 4.5), soft(th, -4.5, -2.3))], axis=1) * in_lv[:, None]

    out = {"credit": "Heart anatomy: average of 19 healthy adult hearts from CT, Rodero, Strocchi et al. (2021), "
                     "Zenodo 10.5281/zenodo.4593739, CC BY 4.0. Activation: simplified eikonal simulation (heart3D).",
           "scenarios": {k: round(float(np.nanmax(times[k])), 1) for k in ("normal", "lbbb", "rbbb", "atria")},
           "repolRange": [round(float(np.nanmin(times["repol"])), 1), round(float(np.nanmax(times["repol"])), 1)],
           "parts": []}
    for name, part_tags in PARTS.items():
        faces = boundary_faces(pts, cells[np.isin(tags, part_tags)])
        used = np.unique(faces)
        remap = np.full(len(pts), -1)
        remap[used] = np.arange(len(used))
        v_in, f_in = scene[used], remap[faces]
        reduction = max(0.0, 1 - TRIANGLES[name] / len(f_in))
        v_out, f_out = fast_simplification.simplify(v_in, f_in.astype(np.int32), target_reduction=reduction)
        # Carry point data from the nearest original surface point.
        nearest = used[cKDTree(v_in).query(v_out)[1]]
        u8 = lambda x, top: np.where(np.isfinite(x), np.clip(np.round(x / top * 255), 0, 255), 0).astype(np.uint8)
        part = {"name": name, "count": len(v_out),
                "positions": b64(v_out.astype(np.float32)),
                "indices": b64(f_out.astype(np.uint16 if len(v_out) < 65536 else np.uint32).ravel()),
                "index32": len(v_out) >= 65536}
        if name in ("LV", "RV"):
            for k in ("normal", "lbbb", "rbbb"):
                part[k] = b64(u8(times[k][nearest], out["scenarios"][k]))     # 0-255 = start..end of QRS
            lo, hi = out["repolRange"]
            part["repol"] = b64(u8(times["repol"][nearest] - lo, hi - lo))
            part["walls"] = b64(np.round(walls[nearest] * 255).astype(np.uint8).ravel())
        if name in ("LA", "RA"):
            part["atria"] = b64(u8(times["atria"][nearest], out["scenarios"]["atria"]))
        out["parts"].append(part)
        print(f"  {name:9}: {len(f_in):7d} -> {len(f_out):6d} triangles, {len(v_out)} points")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {os.path.relpath(OUT, HERE)} ({os.path.getsize(OUT) / 1e6:.2f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
