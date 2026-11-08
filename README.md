# Clouds

A real-time particle cloud rendered in the browser. Over 260,000 particles drift through a curl-noise flow field, held inside a billowing cloud shape and lit from within.

Built with three.js and GLSL. The simulation runs entirely on the GPU.

## Run it

Requires [Node.js](https://nodejs.org) 18 or later.

```bash
npm install
npm run dev
```

Then open http://localhost:5173.

To build a static version for hosting:

```bash
npm run build
```

The output goes to `dist/`.

## Controls

| Input | Action |
| --- | --- |
| `H` | Show or hide all controls |
| `1` – `6` | Colour modes: Ember, Ice, Nebula, Toxic, Gold, Mono |
| `P` | Save a high-resolution PNG (up to 4096 px) |
| `R` | Start / stop recording a video (MP4 or WebM, depending on the browser) |
| Drag | Orbit the camera |
| Scroll | Zoom |

The control panel has live sliders for speed, swirl size, rise, fine detail, shape hold, lumpy border, edge thickness, core amount, dot size, opacity, inner glow, depth, glow, film grain and rotation.

Settings that need a reload are at the top of `main.js`:

- `SIZE`: particle grid size. `512` gives 262k particles, `1024` gives 1M.
- `SHAPE_SEED`: change it to generate a different cloud shape.

## How it works

**GPU simulation.** Each particle's position is stored as one pixel of a floating-point texture. A fragment shader updates every pixel each frame using `GPUComputationRenderer`, so all particles move in parallel on the GPU.

**Curl noise.** Particles follow the curl of a 3D simplex noise field. Curl noise is divergence-free, so particles swirl like smoke without clumping. A second, smaller noise layer adds fine wisps.

**Cloud shape.** The cloud is a set of overlapping spheres: large billows with smaller bumps on top, perturbed by noise. Most particles are kept in a thin band under the surface, which draws crisp edges. The rest stay deep inside and form the glowing core.

**Life cycle.** Each particle has a lifetime. It fades in, travels, fades out, and respawns at its origin, which keeps the cloud alive indefinitely.

**Rendering.** Particles are drawn as soft points with additive blending, so dense areas glow brighter. Colour depends on how deep a particle sits inside the shape, and particles further from the camera are darker. A bloom pass adds glow, and a final pass adds film grain and a vignette.

## Project structure

```
.
├── index.html     Page and canvas container
├── main.js        Scene, simulation, shaders, controls
└── package.json   Dependencies and scripts
```

## Credits

3D simplex noise by Ashima Arts and Stefan Gustavson ([webgl-noise](https://github.com/ashima/webgl-noise), MIT License).
