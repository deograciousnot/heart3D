# 2. The computer science view: an ECG is a projection problem

Forget the biology for a moment.

## The heart is one moving 3D vector

At every instant, all the heart's electrical activity can be summarised as **one 3D vector** (an arrow with direction and strength):

```
h(t) = [x, y, z]      x = patient's left, y = feet, z = back   (Frank convention)
```

During one beat the arrow swings around and traces a loop: a small loop for P, a big loop for QRS, a medium loop for T.

## Every lead is a dot product

Each lead has a fixed unit direction `L_k` (where its "camera" sits). The voltage it records is the projection of the heart vector onto that direction:

```python
lead_k(t) = np.dot(h(t), L_k)
```

That *is* the up/down rule from the biology notes:

| Arrow vs lead | cos(angle) | Trace |
|---|---|---|
| pointing at it | +1 | big spike up |
| pointing away | -1 | big spike down |
| perpendicular | 0 | nearly flat |

Stack the 12 lead directions into a matrix `L` with shape `(12, 3)`:

```python
leads = L @ h          # forward model: (12, 3) @ (3, T) -> (12, T)
```

So a 12-lead ECG is **12 one-dimensional shadows of a single moving 3D arrow**.

## Going backwards = least squares

What a cardiologist does in their head is the inverse: given the 12 shadows, recover the arrow.

```python
h = np.linalg.lstsq(L, leads, rcond=None)[0]   # (3, T)
```

12 equations, 3 unknowns, so it's over-determined, and least squares averages away some noise.

For real bodies the lead directions are not ideal (the torso isn't a uniform sphere, the heart isn't at the centre), so we use a matrix **fitted to real data** instead:

- **Dower transform** (1980s): 12-lead -> 3D, and its inverse.
- **Kors matrix** (Kors et al., 1990): regression from leads I, II, V1-V6 to Frank X, Y, Z. Generally the most accurate simple option, and what this project uses for real data.

The 3D loop has a name: the **vectorcardiogram (VCG)**. It's been around since the 1950s; doctors mostly stopped using it because flat traces were easier to print.

## Analogy

It's like reconstructing a 3D object's motion from several security cameras, except each camera only gives you **one number per frame** (how much the object moved towards it).

## The honest limit

- 12 leads contain only about **8 independent signals** (the 4 augmented/III leads are computed from I and II).
- The dipole model mostly describes **one big arrow**, not the shape of the heart muscle.
- So a 12-lead ECG **cannot** give you a real anatomical image. AI can infer surprising things from it (e.g. weak pumping), but it's recognising patterns, not seeing the heart.
- Getting electrical activity onto the actual heart surface needs many more electrodes plus a CT scan, and a regularised inverse solve (see `03-landscape-and-ideas.md`).
