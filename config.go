package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size and timing for
// the acoustic wave simulation.
const (
	w, h                   = 512, 512
	windowScale            = 1
	damp                   = 0.998
	speed                  = 0.5
	waveDamp32             = float32(damp)
	waveSpeed32            = float32(speed)
	emitterRad             = 1
	moveSpeed              = 2
	stepDelay              = 60 / 4
	defaultTPS             = 60.0
	defaultSimMultiplier   = 10
	simMultiplierStep      = 10
	minSimMultiplier       = 1
	maxSimMultiplier       = 1000
	earOffsetCells         = 5
	defaultBoundaryReflect = 0.4
	stepImpulseStrength    = 10.0
	wallSegments           = 20
	wallMinLen             = 12
	wallMaxLen             = 300
	wallExclusionRadius    = 1
	wallThicknessVariance  = 5
	pgoRecordDuration      = 15 * time.Second
)

var boundaryReflect = defaultBoundaryReflect
