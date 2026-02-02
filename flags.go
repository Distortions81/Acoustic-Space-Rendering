package main

import "flag"

// Command-line flags that control optional rendering, simulation, and runtime
// behavior. Each flag mirrors the original configuration options available in
// the monolithic main.go file.
var (
	// showWallsFlag toggles rendering of wall geometry overlays.
	showWallsFlag = flag.Bool("show-walls", true, "render wall geometry overlays")

	// worldBoundaryReflectFlag adjusts how strongly the outer boundary reflects waves.
	worldBoundaryReflectFlag = flag.Float64("world-boundary-reflect", defaultBoundaryReflect, "amplitude reflection coefficient for the world boundary (0-1); used when -world-boundary-absorb=false")

	// worldBoundaryAbsorbFlag forces the outer boundary to be absorbing (no reflection).
	worldBoundaryAbsorbFlag = flag.Bool("world-boundary-absorb", false, "treat the world boundary as absorbing (overrides world boundary reflection when true)")

	// preferFP16Flag enables 16-bit wave buffers on devices that support half precision.
	preferFP16Flag = flag.Bool("prefer-fp16", false, "use 16-bit floats for the OpenCL solver when supported")

	// recordDefaultPGO triggers a scripted walk to produce default.pgo.
	recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")

	// occludeLineOfSightFlag hides regions outside of the listener's line of
	// sight while rendering.
	occludeLineOfSightFlag = flag.Bool("occlude-line-of-sight", false, "hide regions that are not in the listener's line of sight when rendering")

	// fovDegreesFlag adjusts the field of view for visibility calculations.
	fovDegreesFlag = flag.Float64("fov-deg", 90.0, "field of view angle for LOS (degrees)")

	// lastFrameOnlyFlag forces the renderer to show only the most recent frame.
	lastFrameOnlyFlag = flag.Bool("show-last-frame", false, "render only the latest simulation frame instead of the blended accumulation")

	// debugFlag enables the FPS and simulation overlay.
	debugFlag = flag.Bool("debug", true, "show FPS and simulation speed overlay")

	verifyOpenCLSyncFlag = flag.Bool("verify-opencl-sync", false, "compare OpenCL buffers before/after simulation steps when skipping host uploads")

	// enableAudioFlag toggles optional audio output driven by center samples.
	enableAudioFlag = flag.Bool("enable-audio", true, "enable experimental audio output from center samples")

	// audioLoopFlag lets the user provide a WAV file that will loop instead of the impulse samples.
	audioLoopFlag = flag.String("audio-loop", "test2.wav", "path to a WAV file to loop when audio output is enabled")

	// disableWalkingPulsesFlag suppresses the walking-generated pressure pulses.
	disableWalkingPulsesFlag = flag.Bool("disable-walking-pulses", true, "prevent movement from queuing impulses into the wave field")

	// captureStepSamplesFlag enables per-step center sampling on the GPU.
	captureStepSamplesFlag = flag.Bool("capture-step-samples", true, "capture per-step center samples on the GPU (higher GPU/CPU overhead)")

	// emitterGainFlag scales the audio-loop-driven emitter samples before they are injected into the field.
	emitterGainFlag = flag.Float64("emitter-gain", defaultEmitterGain, "gain applied to audio-loop emitter samples before injection")

	// airAbsDbPerMFlag controls additional per-step damping from frequency-agnostic air absorption.
	airAbsDbPerMFlag = flag.Float64("air-abs-dbpm", defaultAirAbsDbPerM, "approximate air absorption in dB per meter (amplitude), applied as per-step damping")

	// cellSizeMFlag defines the physical size of one grid cell in meters.
	cellSizeMFlag = flag.Float64("cell-size-m", defaultCellSizeM, "meters per grid cell used for solver coefficient calibration")

	// cellSizeCMFlag defines the physical size of one grid cell in centimeters. When set to >0, it overrides -cell-size-m.
	cellSizeCMFlag = flag.Float64("cell-size-cm", 0, "centimeters per grid cell (overrides -cell-size-m when >0)")

	// cellSizeMMFlag defines the physical size of one grid cell in millimeters. When set to >0, it overrides -cell-size-m.
	cellSizeMMFlag = flag.Float64("cell-size-mm", 0, "millimeters per grid cell (overrides -cell-size-m when >0)")

	// worldWidthFeetFlag sets the physical world width in feet. When set to >0, it
	// overrides the other cell size flags by defining dx = worldWidthFeet / w.
	worldWidthFeetFlag = flag.Float64("world-width-ft", 0, "physical world width in feet (overrides cell size flags when >0)")

	// airTempCFlag configures the air temperature (°C) used to compute the speed of sound.
	airTempCFlag = flag.Float64("air-temp-c", defaultAirTempC, "air temperature in °C used to compute the speed of sound")

	// rt60SecondsFlag configures the target decay time used to compute per-step damping.
	rt60SecondsFlag = flag.Float64("rt60-s", defaultRT60Seconds, "target RT60 decay time in seconds (controls per-step damping); <=0 disables damping")

	// runSpeedMPSFlag sets the listener running speed used when converting movement to grid cells.
	runSpeedMPSFlag = flag.Float64("run-speed-mps", defaultRunSpeedMPS, "listener running speed in meters/second used for movement")

	// walkSpeedMPSFlag sets the listener walking speed used when converting movement to grid cells.
	walkSpeedMPSFlag = flag.Float64("walk-speed-mps", defaultWalkSpeedMPS, "listener walking speed in meters/second used when holding Shift")

	// Room wall generation parameters (converted from meters to cells using -cell-size-m).
	roomWallSegmentsFlag         = flag.Int("room-wall-segments", defaultWallSegments, "number of random room wall segments to generate")
	roomWallMinLenMFlag          = flag.Float64("room-wall-min-len-m", defaultWallMinLenM, "minimum room wall segment length in meters")
	roomWallMaxLenMFlag          = flag.Float64("room-wall-max-len-m", defaultWallMaxLenM, "maximum room wall segment length in meters")
	roomWallThicknessMFlag       = flag.Float64("room-wall-thickness-m", defaultWallThicknessM, "approximate room wall thickness in meters")
	roomWallThicknessJitterMFlag = flag.Float64("room-wall-thickness-jitter-m", defaultWallThicknessJitM, "random room wall thickness variation in meters")
	roomWallExclusionRadiusMFlag = flag.Float64("room-wall-exclusion-radius-m", defaultWallExclusionM, "minimum distance from listener to place room walls (meters)")
	roomWallReflectFlag          = flag.Float64("room-wall-reflect", defaultRoomWallReflect, "amplitude reflection coefficient at room wall surfaces (0-1)")
)
