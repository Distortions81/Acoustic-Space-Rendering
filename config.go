package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size, timing, and audio behavior for
// the acoustic wave simulation.
const (
	w, h                  = 512, 512
	windowScale           = 2
	damp                  = 0.9994
	speed                 = 0.5
	waveDamp32            = float32(damp)
	waveSpeed32           = float32(speed)
	emitterRad            = 3
	moveSpeed             = 2
	stepDelay             = 60 / 4
	defaultTPS            = 60.0
	defaultSimMultiplier  = 300
	simMultiplierStep     = 10
	minSimMultiplier      = 1
	maxSimMultiplier      = 1000
	earOffsetCells        = 5
	boundaryReflect       = 0.90
	stepImpulseStrength   = 1.0
	wallSegments          = 25
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
