# UNBOUND
### Exploring the Gravitational Chaos Caused by a Rogue Star

## What this demonstrates
An interactive 2D gravitational simulation showing how a passing compact 
stellar object disturbs a planetary system, alters planetary orbits, and can 
eject planets from their host star. Motion is computed from real gravitational 
forces, not scripted paths.

## Overview
What happens when a rogue star passes close to a planetary system? Even 
without collision, its gravity alone can significantly reshape planetary 
trajectories. A planet may:
- Remain gravitationally bound to its host star
- Shift into a highly eccentric orbit
- Gain enough energy to become potentially unbound
- Undergo major orbital disruption during the encounter

**Important:** This is a synthetic scientific model, not a reconstruction of 
a real astronomical event.

## Features
- Real-time multi-body gravitational simulation
- Rogue-star flyby and planetary disruption
- Newtonian N-body gravity, Velocity Verlet integration
- Orbital energy and bound/unbound analysis
- Minimum rogue-star distance tracking, encounter-phase monitoring
- Glowing celestial bodies, orbital trails, scientific HUD

## Physics
Newtonian gravity in astronomical units — distance (AU), mass (M☉), time 
(years), G = 4π². A softening parameter reduces singularities during close 
approaches.

Specific orbital energy per planet:

E = v²/2 − GM/r

- E < 0: bound (two-body approximation)
- E ≥ 0: potentially unbound

Since multiple bodies interact simultaneously, host-relative energy is not 
independently conserved during the encounter — this is expected, not an error.

## System
| Body | Mass | Role |
|---|---:|---|
| Host Star | 1.0 M☉ | Reference star |
| Inner Planet | 3×10⁻⁶ M☉ | Initial orbit near 1 AU |
| Outer Planet | 1×10⁻⁵ M☉ | Initial orbit near 1.8 AU |
| Compact Intruder | 0.6 M☉ | Rogue flyby object |

Initial conditions are manually chosen to demonstrate gravitational disruption.

## Validation
A timestep refinement test compared dt = 0.00002 with dt/2 = 0.00001.
The refined runs showed improved numerical agreement, with energy differences
below 0.4%. Close encounters remain sensitive to timestep and initial
conditions.

## Controls
| Key | Action |
|---|---|
| SPACE | Pause / Resume |
| S | Toggle rogue star |
| R | Reset simulation |
| UP/DOWN | Adjust speed |

## Assumptions and Limitations
- Two-dimensional model
- Newtonian gravity only — no general relativity, stellar evolution, or 
  hydrodynamics
- Celestial parameters are manually defined synthetic values, not from any 
  real star catalog
- The intruder is a gravitational point mass, not a physically detailed 
  compact object
- Visual sizes are enlarged for clarity, not to scale
- Bound/unbound classification is an approximation in this multi-body system

## Simulation Preview
![UNBOUND Simulation](assets/unbound.png)

## Running it
```bash
git clone https://github.com/vaibhavitej-a11y/UNBOUND
cd UNBOUND
pip install taichi numpy
python sim.py
```
Or with uv:
```bash
uv run --python 3.12 --with taichi sim.py
```
Runs on CUDA GPU when available. CPU fallback via Taichi has not been 
independently verified on a non-GPU machine as of submission.

## Technology
Python, Taichi (GPU-accelerated), Velocity Verlet integration. Single-file 
implementation: `sim.py`.
