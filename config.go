package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size, timing, and audio behavior for
// the acoustic wave simulation.
const (
	w, h                  = 1024, 1024
	windowScale           = 1
	damp                  = 0.9995
	speed                 = 0.5
	waveDamp32            = float32(damp)
	waveSpeed32           = float32(speed)
	emitterRad            = 3
	moveSpeed             = 2
	stepDelay             = 60 / 4
	defaultTPS            = 60.0
	defaultSimMultiplier  = 370
	simMultiplierStep     = 10
	minSimMultiplier      = 1
	maxSimMultiplier      = 1000
	earOffsetCells        = 5
	boundaryReflect       = 0.60
	stepImpulseStrength   = 10
	wallSegments          = 50
	wallMinLen            = 12
	wallMaxLen            = 100
	wallExclusionRadius   = 1
	wallThicknessVariance = 2
	pgoRecordDuration     = 15 * time.Second
	audioSampleRate       = 48000
	audioBufferDuration   = 80 * time.Millisecond
	pcm16MaxValue         = 32767
	pcm16MinValue         = -32768
)
