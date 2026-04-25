# Beam Dashboard – CNT Concrete Beam (3‑point bending)

## Description

Structural health dashboard for a carbon nanotube (CNT) doped concrete beam subjected to a 3‑point bending test.

The code computes:
- non‑linear mechanical response (Hognestad compression, linear brittle tension)
- neutral axis depth and curvature
- electrical resistance change via piezoresistive effect (gauge factor)
- health status (SAFE / CAUTION / WARNING / FAILURE)

## Features

- Exact stress integration over the beam height (500 layers by default)
- Robust equilibrium solving (bisection for axial force, Newton with bisection fallback for target moment)
- External JSON configuration (material, geometry, numerical parameters)
- Logging for debugging
- Built‑in unit tests (`--test`)
- JSON result export
