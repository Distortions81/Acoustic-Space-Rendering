package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size and timing for
// the acoustic wave simulation.
const (
	w, h                     = 950, 950
	windowScale              = 1
	defaultWorldWidthFeet    = 100
	defaultCellSizeM         = defaultWorldWidthFeet * 0.3048 / float64(w)
	defaultAirTempC          = 20.0
	defaultRT60Seconds       = 0.5
	emitterRad               = 1
	defaultWalkSpeedMPS      = 1.4
	defaultRunSpeedMPS       = 3.0
	defaultEmitterGain       = 1
	defaultTPS               = 60.0
	defaultSimMultiplier     = 735
	simMultiplierStep        = 10
	minSimMultiplier         = 1
	maxSimMultiplier         = 1000
	defaultEarSpacingM       = 0.18
	defaultEarDirectivity    = 0.8
	defaultBoundaryReflect   = 0.98
	defaultAirAbsDbPerM      = 0.01
	visualGamma              = 1.8
	defaultWallSegments      = 4
	defaultWallMinLenM       = 1.0
	defaultWallMaxLenM       = 12.0
	defaultWallThicknessM    = 0.15
	defaultWallThicknessJitM = 0.10
	defaultWallExclusionM    = 0.5
	defaultRoomWallMaterial  = "brick"
	defaultRoomWallReflect   = 0.90
	pgoRecordDuration        = 15 * time.Second
	sampleCaptureLogInterval = 500 * time.Millisecond
	defaultAudioBufferMS     = 40
)

var worldBoundaryReflect = defaultBoundaryReflect

var worldScaleMetersPerCell float64
var targetTPS = defaultTPS

var (
	roomWallSegments      = defaultWallSegments
	roomWallMinLenM       = defaultWallMinLenM
	roomWallMaxLenM       = defaultWallMaxLenM
	roomWallThicknessM    = defaultWallThicknessM
	roomWallThicknessJitM = defaultWallThicknessJitM
	roomWallExclusionM    = defaultWallExclusionM
	roomWallReflect       = defaultRoomWallReflect
)
