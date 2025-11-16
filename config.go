package main

import "time"

// Simulation and rendering configuration constants used throughout the
// application. These values define the grid size and timing for
// the acoustic wave simulation.
const (
	referenceWidth              = 1920
	referenceHeight             = 1080
	simulationResolutionDivisor = 2

	w, h                   = referenceWidth / simulationResolutionDivisor, referenceHeight / simulationResolutionDivisor
	windowScale            = 1
	damp                   = 0.999
	speed                  = 0.5
	waveDamp32             = float32(damp)
	waveSpeed32            = float32(speed)
	emitterRad             = 5
	moveSpeed              = 2
	stepDelay              = defaultTPS / 4
	defaultTPS             = 60.0
	defaultSimMultiplier   = 10
	simMultiplierStep      = 10
	minSimMultiplier       = 1
	maxSimMultiplier       = 1000
	defaultBoundaryReflect = 0.4
	stepImpulseStrength    = 1.0
	wallSegments           = 20
	wallMinLen             = 12
	wallMaxLen             = 300
	wallExclusionRadius    = 1
	wallThicknessVariance  = 5
	pgoRecordDuration      = 15 * time.Second
	defaultViewportWidth   = w
	defaultViewportHeight  = h
	defaultBlockWidth      = 128
	defaultBlockHeight     = 128
)

var (
	viewportWidth  = defaultViewportWidth
	viewportHeight = defaultViewportHeight
	blockWidth     = defaultBlockWidth
	blockHeight    = defaultBlockHeight
	windowWidth    = referenceWidth
	windowHeight   = referenceHeight
)

var boundaryReflect = defaultBoundaryReflect
