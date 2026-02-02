package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size and timing for
// the acoustic wave simulation.
const (
	w, h                     = 800, 800
	windowScale              = 2
	defaultCellSizeM         = 0.0125
	defaultAirTempC          = 20.0
	defaultRT60Seconds       = 1.2
	emitterRad               = 1
	defaultWalkSpeedMPS      = 1.4
	defaultRunSpeedMPS       = 3.0
	defaultEmitterGain       = 0.25
	stepDelay                = 60 / 4
	defaultTPS               = 60.0
	defaultSimMultiplier     = 735
	simMultiplierStep        = 10
	minSimMultiplier         = 1
	maxSimMultiplier         = 1000
	earOffsetCells           = 5
	defaultBoundaryReflect   = 0.98
	defaultAirAbsDbPerM      = 0.01
	visualGamma              = 1.25
	stepImpulseStrength      = 20.0
	defaultWallSegments      = 10
	defaultWallMinLenM       = 1.0
	defaultWallMaxLenM       = 12.0
	defaultWallThicknessM    = 0.15
	defaultWallThicknessJitM = 0.10
	defaultWallExclusionM    = 0.5
	pgoRecordDuration        = 15 * time.Second
	sampleCaptureLogInterval = 500 * time.Millisecond
	audioPlayerBufferLatency = 40 * time.Millisecond
)

var boundaryReflect = defaultBoundaryReflect
