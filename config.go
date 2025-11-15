package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size and timing for
// the acoustic wave simulation.
const (
	w, h                     = 800, 800
	windowScale              = 2
	damp                     = 0.9997
	speed                    = 0.5
	waveDamp32               = float32(damp)
	waveSpeed32              = float32(speed)
	emitterRad               = 1
	moveSpeed                = 2
	stepDelay                = 60 / 4
	defaultTPS               = 60.0
	defaultSimMultiplier     = 735
	simMultiplierStep        = 10
	minSimMultiplier         = 1
	maxSimMultiplier         = 1000
	earOffsetCells           = 5
	defaultBoundaryReflect   = 0.4
	visualGamma              = 1.8
	stepImpulseStrength      = 20.0
	wallSegments             = 50
	wallMinLen               = 12
	wallMaxLen               = 300
	wallExclusionRadius      = 1
	wallThicknessVariance    = 5
	pgoRecordDuration        = 15 * time.Second
	sampleCaptureLogInterval = 500 * time.Millisecond
	audioPlayerBufferLatency = 40 * time.Millisecond
)

var boundaryReflect = defaultBoundaryReflect
