# Acoustic Space Rendering – Simulation Details

This document describes how the acoustic simulation works, with pointers to the
main files and functions in the repository.

## High‑Level Overview

- The 2D window is treated as a fixed grid of air pressure samples.
- A damped wave equation is solved on this grid using an OpenCL kernel defined
  in `opencl_wave.go`.
- Walls carve holes in the grid, the outer border reflects waves, and one or
  more sources inject pressure (step impulses and/or an audio waveform).
- Each Ebiten tick, `Game.Update` in `game.go` advances the wave field by many
  small solver steps on the GPU, then the results are rendered and (optionally)
  turned into audio.

## Wave Field and Solver

### CPU Wave Field

- The CPU representation of the wave field is `waveField` in `wave_field.go`.
- It owns three full‑resolution buffers:
  - `curr`: pressure at the current time step.
  - `prev`: pressure at the previous time step.
  - `next`: scratch buffer for the next time step.
- `newWaveField` allocates these buffers for a grid of size `w × h`, where
  `w` and `h` are defined in `config.go`.

### GPU Solver

- `openCLWaveSolver` in `opencl_wave.go` owns:
  - An OpenCL context, command queue, and compiled program.
  - Device buffers mirroring the CPU field (`currBuf`, `prevBuf`, `nextBuf`).
  - Extra buffers for pixels, accumulated intensity, wall masks, visibility
    masks, impulses, and ear sample capture.
- `newOpenCLWaveSolver`:
  - Chooses an OpenCL device (GPU when available, otherwise CPU).
  - Builds kernels from the string `waveKernelSource`.
  - Allocates all device buffers sized from `width` and `height`.
  - Optionally enables half‑precision (`fp16`) storage when the device
    advertises `cl_khr_fp16` or `cl_khr_half_float`.
- The solver exposes a single high‑level entry point:
  - `(*openCLWaveSolver).Step` runs one batch of simulation steps and prepares
    pixels and ear samples for the host.

### Discrete Wave Equation

- The core physics live in the `wave_step` kernel defined inside
  `waveKernelSource` in `opencl_wave.go`.
- For each interior grid cell, `wave_step`:
  - Reads the current pressure at the cell and its four direct neighbors.
  - Computes a discrete Laplacian from those neighbors.
  - Uses a leapfrog update: the next value depends on the current and previous
    values plus the Laplacian term.
  - Multiplies by a damping factor so energy decays over time.
- The constants that control this behavior are defined in `config.go`:
  - The solver uses the maximum stable wave speed coefficient for this stencil
    (`speedCoeff=0.5`) to preserve audio fidelity at our performance limits.
  - The world scale (`dx`, meters per cell) is derived once at startup from an
    assumed real-time step rate and `-air-temp-c`.
  - `damp` is computed from a target RT60 (seconds) as `exp(-ln(1000)·dt/RT60)`,
    using the current simulation step rate.
- When an emitter is active (see “Continuous audio‑driven emitter” below),
  `wave_step` overwrites the computed value at the emitter’s cell with a
  supplied source value instead of the normal update.

## Sources of Energy

### Footstep‑Style Impulses

- Movement‑driven impulses are created in `Game.Update` in `game.go`:
  - When the listener is moving, a step timer increments each frame.
  - Every `stepDelay` frames (configured in `config.go`), a cluster of
    impulses is queued around the listener.
- The pattern of cells to excite is defined by `emitterFootprint` in
  `emitter_footprint.go`, which contains a precomputed disk of offsets with
  radius `emitterRad` from `config.go`.
- For each footprint cell that lies inside the grid and is not a wall,
  `Game.Update` calls `waveField.queueImpulse`:
  - This writes a pressure value (`stepImpulseStrength` from `config.go`)
    into the CPU `curr` buffer.
  - It appends a `waveImpulse` record to an internal list in `waveField`.
- At the start of each GPU batch, `openCLWaveSolver.applyQueuedImpulses`
  drains those impulses and uploads them to the GPU:
  - It writes impulse indices and values into device buffers.
  - It dispatches the `apply_impulses` kernel, which sets each targeted cell
    to the requested value in the GPU field.
  - Impulses can be applied to both the current and previous buffers so they
    integrate cleanly into the leapfrog scheme.

### Continuous Audio‑Driven Emitter

- When `-enable-audio` and `-audio-loop` are used, a WAV file is decoded in
  `audio_loop.go`:
  - `loadLoopSamples` uses `wav.DecodeWithSampleRate` to obtain PCM frames.
  - `decodeStereoI16ToFloat` converts interleaved stereo `int16` samples to a
    mono `[]float32` in the range approximately `[-1, 1]`.
- `audioPressureSource` in `audio_loop.go` stores this sample buffer:
  - `fillChunk` writes a requested number of samples into a caller‑provided
    slice, looping when the end is reached.
- In `Game.Update`, if an `audioPressureSource` is available:
  - `fillAudioChunk` fills a slice whose length equals the number of wave
    steps being run this frame.
  - `emitterAudioIndex` computes the grid index underneath the listener.
  - `Game.Update` passes both index and per‑step samples to
    `openCLWaveSolver.Step` via `audioEmitterData`.
- Inside `Step`, when an emitter is present:
  - The solver records `emitterIndex` and the slice of `emitterSamples`.
  - For each simulation substep, it calls `setEmitterValue` with the
    appropriate sample for that substep.
  - In `wave_step`, the kernel adds a source term derived from `emitter_value`
    onto the computed next value, distributing it across a 3×3 neighborhood
    around `emitter_index` using binomial weights. This reduces wideband
    artifacts compared to hard overwriting a single cell each step.

## Walls and Boundaries

### Interior Walls

- The interior layout is generated procedurally by `Game.generateWalls` in
  `environment.go`:
  - It clears any previous wall state.
  - It randomly chooses segment positions, orientations, lengths, and
    thicknesses based on runtime flags:
    - `-room-wall-segments` controls how many segments are generated.
    - `-room-wall-min-len-m` / `-room-wall-max-len-m` define segment lengths in meters.
    - `-room-wall-thickness-m` / `-room-wall-thickness-jitter-m` define thickness in meters.
    - These meter values are converted to cell units using the derived `dx`.
  - For each segment, it calls `trySetWall` to mark grid cells as walls.
- `trySetWall` enforces:
  - Walls stay away from the window border.
  - Walls stay outside an exclusion radius around the listener
    (`-wall-exclusion-radius-m`) so the player does not start inside solid
    geometry.
  - When a cell becomes a wall, `waveField.zeroCell` clears its wave values
    to zero.
- At simulation time, walls are represented by a Boolean slice `walls` on the
  CPU and by an 8‑bit `wall_mask` buffer on the GPU.
  - `openCLWaveSolver.refreshWallMask` uploads this mask as needed.
  - In the `wave_step` kernel, any cell with `wall_mask[idx]` set is treated
    as solid. Neighboring fluid cells treat wall-adjacent samples as
    reflective boundaries (controlled by `-room-wall-material` and
    `-room-wall-reflect`) when computing the Laplacian, producing room-wall
    reflections instead of “holes”.

### Outer Grid Boundaries

- The four outer edges of the grid act like wall boundaries.
- This behavior is implemented in the `wave_step` kernel inside
  `waveKernelSource`:
  - For cells adjacent to the world boundary, the Laplacian treats the
    out-of-bounds neighbor as a reflected “ghost” value derived from the cell
    itself. This uses the same style of boundary condition as interior room
    walls.
- `worldBoundaryReflect` in `config.go` holds the configured reflection coefficient.
  - In `main.go`, `-world-boundary-absorb` and `-world-boundary-reflect` are
    parsed and stored into `worldBoundaryReflect`.
  - `-wall-reflect-mult` scales both world and room wall reflection
    coefficients as a quick global “dry/wet” control.
  - When `-world-boundary-absorb=true` (the default), `main.go` sets
    `worldBoundaryReflect` to `0`, making the world boundary absorbing.
- After each call to `wave_step`, `openCLWaveSolver.runBoundaryAccumulate`
  runs `boundary_accumulate` to accumulate a scaled magnitude into a separate
  accumulation buffer used for visualization.

## Visibility and Field‑of‑View Masking

- The simulation can optionally hide regions that are not in the listener’s
  current field of view when rendering.
- The visibility system is implemented in `visibility.go`:
  - `refreshVisibleMask` is called from `Game.Update` when
    `-occlude-line-of-sight` is enabled.
  - It ensures `visibleStamp` matches the grid size and increments a
    generation counter on each refresh.
  - It computes the listener’s facing vector using `listenerForwardX` and
    `listenerForwardY` from `Game`, falling back to “upwards” when the
    listener is stationary.
  - It clamps the field‑of‑view angle using the `-fov-deg` flag.
- `refreshVisibleMask` then performs two passes:
  1. A shadowcasting pass via `computeFOVShadow` and `castLight`:
     - This explores eight octants around the listener, marking cells that
       are reachable, inside the FOV cone, and not blocked by walls.
  2. A fallback pass using `castVisibilityRay`:
     - If too few cells are visible, it casts Bresenham line‑of‑sight rays
       from the listener to precomputed perimeter targets from
       `buildLOSPerimeterTargets` in `grid.go`.
- The result is stored in `visibleStamp` with the current generation.
- On the GPU, `openCLWaveSolver.refreshVisibilityMask` converts
  `visibleStamp` into a compact 8‑bit `visibility_mask` for the current
  generation.
- The `render_intensity` kernel uses `visibility_mask` when
  `use_visibility` is set:
  - Cells where the mask is zero are darkened to black.
  - This affects rendering only; the wave equation still runs everywhere.

## Per‑Frame Simulation Pipeline

Each Ebiten tick, `Game.Update` in `game.go` performs the following steps:

1. **Movement and collision**
   - `movementVector` selects either manual WASD control or auto‑walk.
   - The listener position (`ex`, `ey` in `Game`) is updated and clamped to
     stay inside the grid.
   - If the new position is inside a wall (as determined by `isWall`), the
     move is reverted.
   - When moving, the listener’s forward vector is updated from the movement
     direction.

2. **Step impulses**
   - A frame counter (`stepTimer`) increments while the listener is moving.
   - When `stepTimer` reaches `stepDelay`, it resets and, unless
     `-disable-walking-pulses` is set, queues impulses at every cell in the
     `emitterFootprint` around the listener.
   - Each impulse is added to `waveField` via `queueImpulse`.

3. **Visibility**
   - If `-occlude-line-of-sight` is enabled, `refreshVisibleMask` recomputes
     the visible region around the listener.

4. **Audio emitter samples**
   - If an `audioPressureSource` is present and at least one simulation step
     will be run, `fillAudioChunk` requests a slice of samples whose length
     equals the number of wave steps.
   - `emitterAudioIndex` converts the listener position into a field index.
   - If both index and samples are valid, an `audioEmitterData` is created
     for the GPU solver.

5. **GPU wave steps**
   - `openCLWaveSolver.Step` is called with:
     - The `waveField` buffers.
     - The current wall map.
     - The number of steps (`simStepMultiplier`).
     - Flags for wall drawing, last‑frame‑only rendering, and LOS masking.
     - The visibility stamp and generation (if LOS masking is enabled).
     - Optional `audioEmitterData` for the continuous source.
   - Inside `Step`, the solver:
     - Uploads the field to the GPU if this is the first run.
     - Applies any queued impulses to GPU buffers via
       `applyQueuedImpulses` and `apply_impulses`.
     - Uploads or refreshes the wall and visibility masks if needed.
     - Clears the accumulation buffer.
     - Prepares ear sample capture if the `-capture-step-samples` flag is
       enabled.
     - Loops `steps` times:
       - Configures the emitter value for this substep, if any.
       - Rebinds the current, previous, and next buffers if they have been
         swapped.
       - Dispatches the `wave_step` kernel across the grid.
       - Swaps `currBuf`, `prevBuf`, and `nextBuf`.
       - Calls `runBoundaryAccumulate` to enforce boundary conditions and
         accumulate energy along the outer edges.
       - Optionally calls `sampleCenter` to record the center cell value for
         this substep.

6. **Readback of samples and pixels**
   - If per‑step sampling is enabled, the solver reads an interleaved sequence
     of left/right ear samples from the listener’s ear offsets. Otherwise, it
     reads a single left/right pair for the current buffer.
   - It configures the render kernel to use either the accumulation buffer
     (for blended visualization) or the current buffer (for a single
     wavefront), based on the `-show-last-frame` flag.
   - It runs the `render_intensity` kernel to turn scalar magnitudes and
     masks into RGBA pixels in a device buffer.
   - It enqueues a non‑blocking readback of the pixel buffer into a host
     slice held by `openCLWaveSolver`.

7. **Audio output**
   - Back in `Game.Update`, if a `centerAudioStream` is active:
     - When `-capture-step-samples` is enabled, the last batch’s per‑step
       left/right ear frames are enqueued as interleaved stereo.
   - `centerAudioStream` implements the `io.ReadCloser` interface expected by
     Ebiten’s audio player:
     - `Read` consumes pending samples at the audio device rate, applies a
       simple DC‑blocking filter, and writes stereo `int16` frames.
     - `Close` is a no‑op that satisfies the interface.

## Rendering

- The `Game.Draw` method in `render.go` is responsible for drawing a frame:
  - It calls `PixelBytes` on `openCLWaveSolver`, which waits for any pending
    pixel readback and returns the latest RGBA pixel buffer.
  - It passes those pixels to the Ebiten screen via `WritePixels`.
  - It draws the emitter footprint as a red disk at the listener position.
  - It draws ear offset indicators using `earOffsets` from `environment.go`
    to visualize the listener’s orientation.
  - It draws a small red marker at the grid center to show where audio
    samples are taken.
  - If the `-debug` flag is enabled, it overlays performance statistics such
    as FPS, TPS, simulation speed, and the last simulation batch duration.

Together, these pieces form a GPU‑accelerated visualization of acoustic wave
propagation in a simple, procedurally generated environment, with optional
line‑of‑sight masking and an audio output derived from the simulated pressure
field.
