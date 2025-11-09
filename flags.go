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

	// recordDefaultPGO triggers a scripted walk to produce default.pgo.
	recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")

	// occludeLineOfSightFlag hides regions outside of the listener's line of
	// sight while rendering.
	occludeLineOfSightFlag = flag.Bool("occlude-line-of-sight", false, "hide regions that are not in the listener's line of sight when rendering")

	// fovDegreesFlag adjusts the field of view for visibility calculations.
	fovDegreesFlag = flag.Float64("fov-deg", 90.0, "field of view angle for LOS (degrees)")

	// threadCountFlag specifies how many worker goroutines to run.
	threadCountFlag = flag.Int("threads", 0, "number of worker threads; 0 auto-detects")

	// debugFlag enables the FPS and simulation overlay.
	debugFlag = flag.Bool("debug", false, "show FPS and simulation speed overlay")

	// useOpenCLFlag enables the optional OpenCL solver.
	useOpenCLFlag = flag.Bool("use-opencl", true, "attempt to run the wave simulation via OpenCL (build with -tags opencl)")

	// adaptiveStepScalingFlag enables scaling physics work with ActualTPS.
	adaptiveStepScalingFlag = flag.Bool("scale-steps-with-tps", false, "scale the per-frame physics work based on ActualTPS instead of using a fixed batch size")

	// maxStepBurstFlag limits how aggressively the simulation catches up.
	maxStepBurstFlag = flag.Int("max-step-burst", 4, "maximum multiple of the base physics step count to execute when recovering from lag while step scaling is enabled (0 disables the clamp)")

)
