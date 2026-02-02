# Acoustic Space Rendering

This project uses [Ebiten](https://ebiten.org/) for graphics and an OpenCL-backed simulation. OpenCL is now required; follow the steps below to install the necessary Ubuntu packages and build the application. No build tags are needed.

For a deeper look at how the simulation is structured internally, see
`details.md`.

## Demo

![Acoustic Space Rendering screenshot](screenshot.png)

## Ubuntu dependency setup

### 1. Common build tools
Install basic build tooling used by Go modules and native dependencies:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config git
```

### 2. Ebiten native libraries
Ebiten relies on OpenGL and X11. Install the development headers so the Go compiler can link against them:

```bash
sudo apt install -y libgl1-mesa-dev xorg-dev
```

- `libgl1-mesa-dev`: OpenGL headers and libraries for rendering.
- `xorg-dev`: X11 development headers required for window creation.

### 3. OpenCL toolchain (required)
Install the OpenCL ICD loader, headers, and diagnostic tool:

```bash
sudo apt install -y ocl-icd-opencl-dev opencl-headers clinfo
```

- `ocl-icd-opencl-dev`: OpenCL ICD loader and development files.
- `opencl-headers`: C headers for compiling against OpenCL.
- `clinfo`: Utility to verify that the system detects OpenCL platforms.

> **Tip:** Vendor-specific GPU drivers (e.g., NVIDIA, AMD, Intel) may provide additional optimized OpenCL implementations. Install the appropriate driver package from your vendor to access hardware acceleration.

### 4. Verify OpenCL availability
Confirm that OpenCL platforms are visible:

```bash
clinfo | head
```

If `clinfo` lists at least one platform, you are ready to build:

```bash
go build ./...
```

### Troubleshooting OpenCL startup errors

If the runtime prints `OpenCL initialization failed: querying OpenCL platforms: cl: error -1001`, no ICD loader reported any
available platforms. Install the OpenCL packages from step 3 and your GPU vendor's driver, then rerun `clinfo` to confirm a
platform is detected before launching the application.

## Building the project

Once dependencies are installed, install Go (if not already available) and build the project:

```bash
go build ./...
```

Run the application:

```bash
go run .
```

### Runtime options

Customize simulation behavior with additional flags:

- `-world-boundary-reflect=<value>` — amplitude reflection coefficient for the outer world boundary (0–1), used when `-world-boundary-absorb=false`; default is `0.98`.
- `-world-boundary-absorb=<true|false>` — when `true`, force the world boundary to be absorbing (no reflection), overriding `-world-boundary-reflect` (default `true`).
- `-wall-reflect-mult=<value>` — multiplier applied to wall reflection coefficients (scales both `-room-wall-reflect` and `-world-boundary-reflect`); set to `0.5` to globally halve reflections (default `1.0`).
- `-emitter-gain=<value>` — scales `-audio-loop` samples before they are injected into the wave field (default `0.2`).
- `-air-abs-dbpm=<value>` — approximate air absorption in dB per meter (amplitude), applied as extra per-step damping (default `0.01`).
- `-air-temp-c=<value>` — sets air temperature in °C; used to compute the speed of sound that the derived world scale is based on (default `20`).
- `-rt60-s=<value>` — sets an approximate RT60 decay time in seconds; used to compute per-step damping (`<=0` disables damping). When `-rt60-auto=true` and `-rt60-s` is not explicitly set, a derived RT60 is used instead.
- `-rt60-auto=<true|false>` — when `true`, derive a more realistic RT60 default from the derived world size and `-room-wall-material` (default `true`).
- `-run-speed-mps=<value>` — sets listener running speed in meters/second; used for WASD/autowalk movement (default `3.0`).
- `-walk-speed-mps=<value>` — sets listener walking speed in meters/second while holding Shift (default `1.4`).
- `-ear-directivity=<value>` — ear directionality strength (0–1) applied vs the emitter direction; `0` disables (default `0.8`).
- `-room-wall-segments=<value>` — number of random interior “room wall” segments to generate (default `5`).
- `-room-wall-min-len-m=<value>` — minimum room wall segment length in meters (default `1.0`).
- `-room-wall-max-len-m=<value>` — maximum room wall segment length in meters (default `12.0`).
- `-room-wall-thickness-m=<value>` — approximate room wall thickness in meters (default `0.15`).
- `-room-wall-thickness-jitter-m=<value>` — random room wall thickness variation in meters (default `0.10`).
- `-room-wall-exclusion-radius-m=<value>` — minimum distance from the listener to place room walls, in meters (default `0.5`).
- `-room-wall-material=<value>` — room wall material preset (`drywall`, `concrete`, `brick`, `glass`, `wood`, `curtain`, `acoustic`); default is `drywall`.
- `-room-wall-reflect=<value>` — amplitude reflection coefficient at room wall surfaces (0–1); overrides `-room-wall-material` when explicitly set; default is `0.90` (drywall preset).
- `-prefer-fp16=<true|false>` — toggles 16-bit OpenCL wave buffers when the GPU advertises `cl_khr_fp16`/`cl_khr_half_float`. Leave enabled to reduce bandwidth on capable devices; set to `false` to force 32-bit floats.
- `-enable-audio=<true|false>` — toggles experimental audio output driven by the simulator’s center samples; enable it to hear the impulse stream.
- `-audio-loop=<path>` — when audio is enabled, specify a WAV file (RIFF/PCM) that is resampled to 44.1 kHz and used to drive the emitter’s pressure waveform; audio output still comes from the simulator’s center sample stream.
- `-disable-walking-pulses=<true|false>` — when `true`, walking no longer queues the default impulse pulses so you only see the WAV-driven source or the silent field.
- `-show-last-frame=<true|false>` — when `true`, render only the most recent simulation frame instead of blending values across previous steps so you can inspect the raw wavefront (default `false`).

By default the application already runs with `-debug`, `-enable-audio`, `-capture-step-samples`, and `-disable-walking-pulses` while looping `test2.wav`. Pass flag overrides such as `-debug=false` or `-audio-loop=their.wav` to change that behavior.
