# 1. How an ECG works

An ECG is **not a picture of the heart**. It is a recording of **voltage over time**, measured on the skin, from several angles at once.

## The heart is electrical before it is mechanical

- Every heart muscle cell is a tiny battery: ion pumps keep the inside at about **-90 mV** relative to the outside.
- When a cell fires, sodium rushes in and the voltage flips positive (**depolarisation**). That triggers the squeeze *and* the neighbouring cell, so a wave of electricity spreads across the heart.
- Afterwards the cells pump ions back and reset (**repolarisation**).

## Why the skin can see it

The body is salty water, which conducts. The moving boundary between "fired" and "not yet fired" tissue acts like a dipole (a + end and a - end) and drives currents through the torso. By the time they reach the skin they are about **1 mV**: tiny, but measurable.

## One heartbeat, step by step

```
        R
        |
   P    |      T
  / \   |     / \
_/   \_Q|S___/   \____
```

| Step | What happens | ECG feature |
|---|---|---|
| SA node (natural pacemaker, top right) fires | Wave spreads over the atria | **P wave** |
| AV node holds the signal ~0.1 s | Atria finish emptying into ventricles | Flat **PR segment** |
| His bundle -> bundle branches -> Purkinje fibres | Fast wiring fires the ventricles almost at once | **QRS complex** (big, sharp) |
| Ventricles reset | Repolarisation | **T wave** |

Atrial reset happens during the QRS, so it is hidden.

## The 12 leads = 12 cameras

**The one rule:** a wave travelling **towards** a lead's positive electrode makes the trace go **up**; **away** makes it go **down**; **sideways** gives a small up-down blip.

10 stickers give 12 viewpoints:

- **Limb leads I, II, III** (Einthoven's triangle) and **augmented leads aVR, aVL, aVF** look at the heart in the **frontal plane** (like a clock face from the front).
  Angles: I = 0°, II = 60°, III = 120°, aVF = 90°, aVL = -30°, aVR = -150° (0° = patient's left, +90° = feet).
- **Chest leads V1-V6** look at it in the **horizontal plane** (a slice through the chest), from front-right (V1) round to the left side (V6).

Comparing which leads point up and which point down tells you the **direction of the wave in 3D**, and from that, which wall of the heart is misbehaving.
Example: ST elevation in II, III, aVF means trouble on the bottom (inferior) wall.

## Inside the machine

1. **Ag/AgCl electrodes + gel** convert ionic current (body) into electron current (wire).
2. **Instrumentation amplifier** measures the *difference* between electrodes. Mains hum (50/60 Hz) is on the whole body equally, so subtracting cancels it (**common-mode rejection**).
3. **Right-leg drive** feeds an inverted copy of the noise back to cancel it further.
4. **Filters**: diagnostic ~0.05-150 Hz, monitors ~0.5-40 Hz (removes breathing drift and muscle noise).
5. **ADC + display**: 25 mm/s, 10 mm/mV. Small box = 0.04 s, big box = 0.2 s. Heart rate ≈ 300 / (big boxes between R peaks).

## What it can and can't tell you

- **Good at:** rhythm problems (AF, blocks), heart attacks / ischaemia, thickened walls, electrolytes (high potassium = tall peaked T), drug effects.
- **Can't see:** how well the heart pumps, or the valves. That needs an echocardiogram (ultrasound), which *is* an actual picture.

## Fun facts

- Einthoven's first practical ECG (1903) weighed ~270 kg; patients put their limbs in **buckets of salt water**. Nobel Prize 1924.
- A smartwatch ECG is just **lead I**: watch on one wrist, finger from the other hand on the crown.
