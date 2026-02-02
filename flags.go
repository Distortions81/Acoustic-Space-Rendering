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
	worldBoundaryAbsorbFlag = flag.Bool("world-boundary-absorb", true, "treat the world boundary as absorbing (overrides world boundary reflection when true)")

	// wallReflectMultFlag scales all wall reflection coefficients (room walls + world boundary).
	wallReflectMultFlag = flag.Float64("wall-reflect-mult", 1, "multiplier applied to wall reflection coefficients (scales -room-wall-reflect and -world-boundary-reflect); 0 disables reflections")

	// preferFP16Flag enables 16-bit wave buffers on devices that support half precision.
	preferFP16Flag = flag.Bool("prefer-fp16", false, "use 16-bit floats for the OpenCL solver when supported")

	// recordDefaultPGO triggers a scripted walk to produce default.pgo.
	recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")

	// tpsFlag controls Ebiten's tick/update rate (lower values reduce CPU/GPU overhead).
	tpsFlag = flag.Int("tps", int(defaultTPS), "target ticks per second (frame/update rate), e.g. 15, 30, 60")

	// debugFlag enables the FPS and simulation overlay.
	debugFlag = flag.Bool("debug", true, "show FPS and simulation speed overlay")

	// enableAudioFlag toggles optional audio output driven by center samples.
	enableAudioFlag = flag.Bool("enable-audio", true, "enable experimental audio output from center samples")

	// audioBufferMSFlag controls the audio output buffer size. Lower values reduce
	// latency but increase underrun risk.
	audioBufferMSFlag = flag.Int("audio-buffer-ms", defaultAudioBufferMS, "audio output buffer size in milliseconds")

	// audioLoopFlag lets the user provide a WAV file that will loop instead of the impulse samples.
	audioLoopFlag = flag.String("audio-loop", "speech.wav", "path to a WAV file to loop when audio output is enabled")

	// emitterGainFlag scales the audio-loop-driven emitter samples before they are injected into the field.
	emitterGainFlag = flag.Float64("emitter-gain", defaultEmitterGain, "gain applied to audio-loop emitter samples before injection")

	// airAbsDbPerMFlag controls additional per-step damping from frequency-agnostic air absorption.
	airAbsDbPerMFlag = flag.Float64("air-abs-dbpm", defaultAirAbsDbPerM, "approximate air absorption in dB per meter (amplitude), applied as per-step damping")

	// airTempCFlag configures the air temperature (°C) used to compute the speed of sound.
	airTempCFlag = flag.Float64("air-temp-c", defaultAirTempC, "air temperature in °C used to compute the speed of sound")

	// rt60SecondsFlag configures the target decay time used to compute per-step damping.
	rt60SecondsFlag = flag.Float64("rt60-s", defaultRT60Seconds, "target RT60 decay time in seconds (controls per-step damping); <=0 disables damping")

	// rt60AutoFlag derives a more realistic default RT60 when -rt60-s is not explicitly set.
	rt60AutoFlag = flag.Bool("rt60-auto", true, "derive RT60 from world size and room wall material when -rt60-s is not explicitly set")

	// runSpeedMPSFlag sets the listener running speed used when converting movement to grid cells.
	runSpeedMPSFlag = flag.Float64("run-speed-mps", defaultRunSpeedMPS, "listener running speed in meters/second used for movement")

	// walkSpeedMPSFlag sets the listener walking speed used when converting movement to grid cells.
	walkSpeedMPSFlag = flag.Float64("walk-speed-mps", defaultWalkSpeedMPS, "listener walking speed in meters/second used when holding Shift")

	// earDirectivityFlag applies a simple directionality pattern to each ear based on the emitter direction.
	// 0 disables directionality, 1 uses a full cardioid response.
	earDirectivityFlag = flag.Float64("ear-directivity", defaultEarDirectivity, "ear directionality strength (0-1); 0 disables, 1 is full cardioid vs emitter direction")

	// Room wall generation parameters (converted from meters to cells using the derived world scale).
	roomWallSegmentsFlag         = flag.Int("room-wall-segments", defaultWallSegments, "number of random room wall segments to generate")
	roomWallMinLenMFlag          = flag.Float64("room-wall-min-len-m", defaultWallMinLenM, "minimum room wall segment length in meters")
	roomWallMaxLenMFlag          = flag.Float64("room-wall-max-len-m", defaultWallMaxLenM, "maximum room wall segment length in meters")
	roomWallThicknessMFlag       = flag.Float64("room-wall-thickness-m", defaultWallThicknessM, "approximate room wall thickness in meters")
	roomWallThicknessJitterMFlag = flag.Float64("room-wall-thickness-jitter-m", defaultWallThicknessJitM, "random room wall thickness variation in meters")
	roomWallExclusionRadiusMFlag = flag.Float64("room-wall-exclusion-radius-m", defaultWallExclusionM, "minimum distance from listener to place room walls (meters)")
	roomWallMaterialFlag         = flag.String("room-wall-material", defaultRoomWallMaterial, "room wall material preset: drywall, concrete, brick, glass, wood, curtain, acoustic")
	roomWallReflectFlag          = flag.Float64("room-wall-reflect", defaultRoomWallReflect, "amplitude reflection coefficient at room wall surfaces (0-1)")
)
