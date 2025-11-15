package main

import "flag"

// Command-line flags that control optional rendering, simulation, and runtime
// behavior. Each flag mirrors the original configuration options available in
// the monolithic main.go file.
var (
	// showWallsFlag toggles rendering of wall geometry overlays.
	showWallsFlag = flag.Bool("show-walls", true, "render wall geometry overlays")

	// wallReflectFlag adjusts how strongly the simulation boundaries reflect waves.
	wallReflectFlag = flag.Float64("wall-reflect", defaultBoundaryReflect, "reflection coefficient for map boundaries (0-1)")

	// preferFP16Flag enables 16-bit wave buffers on devices that support half precision.
	preferFP16Flag = flag.Bool("prefer-fp16", true, "use 16-bit floats for the OpenCL solver when supported")

	// recordDefaultPGO triggers a scripted walk to produce default.pgo.
	recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")

	// occludeLineOfSightFlag hides regions outside of the listener's line of
	// sight while rendering.
	occludeLineOfSightFlag = flag.Bool("occlude-line-of-sight", false, "hide regions that are not in the listener's line of sight when rendering")

	// fovDegreesFlag adjusts the field of view for visibility calculations.
	fovDegreesFlag = flag.Float64("fov-deg", 90.0, "field of view angle for LOS (degrees)")

	// debugFlag enables the FPS and simulation overlay.
	debugFlag = flag.Bool("debug", false, "show FPS and simulation speed overlay")

	verifyOpenCLSyncFlag = flag.Bool("verify-opencl-sync", false, "compare OpenCL buffers before/after simulation steps when skipping host uploads")

	// enableAudioFlag toggles optional audio output driven by center samples.
	enableAudioFlag = flag.Bool("enable-audio", false, "enable experimental audio output from center samples")

	// captureStepSamplesFlag enables per-step center sampling on the GPU.
	captureStepSamplesFlag = flag.Bool("capture-step-samples", false, "capture per-step center samples on the GPU (higher GPU/CPU overhead)")
)
