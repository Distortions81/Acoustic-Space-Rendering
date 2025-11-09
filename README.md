# Acoustic Space Rendering

This project uses [Ebiten](https://ebiten.org/) for graphics and an optional OpenCL backend for simulation. The instructions below describe how to install the required Ubuntu packages so you can build and run the application with both features enabled.

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

### 3. OpenCL toolchain
To build with the OpenCL backend you need the ICD loader, headers, and diagnostic tools:

```bash
sudo apt install -y ocl-icd-opencl-dev opencl-headers clinfo
```

- `ocl-icd-opencl-dev`: OpenCL ICD loader and development files.
- `opencl-headers`: C headers for compiling against OpenCL.
- `clinfo`: Utility to verify that the system detects OpenCL platforms.

> **Tip:** Vendor-specific GPU drivers (e.g., NVIDIA, AMD, Intel) may provide additional optimized OpenCL implementations. Install the appropriate driver package from your vendor to access hardware acceleration.

### 4. Verify OpenCL availability (optional)
After installation, confirm that OpenCL platforms are visible:

```bash
clinfo | head
```

If `clinfo` lists at least one platform, you are ready to build with the `opencl` build tag:

```bash
go build -tags opencl ./...
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

To enable the OpenCL solver, include the build tag:

```bash
go build -tags opencl ./...
```

Run the application with OpenCL enabled:

```bash
go run -tags opencl .
```

Use the `-use-opencl` runtime flag to toggle the solver on or off after compiling with the tag.

### Runtime options

Customize simulation behavior with additional flags:

- `-wall-reflect=<value>` — sets how strongly the outer walls reflect waves. Use a value between 0 (fully absorbing) and 1 (perfect reflection); the default is `0.4`.
