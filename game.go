package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

// Game encapsulates the full simulation state, rendering buffers, and audio pipeline.
type Game struct {
	field *waveField

	ex float64
	ey float64

	stepTimer           int
	physicsAccumulator  float64
	lastSimDuration     time.Duration
	simStepMultiplier   int
	adaptiveStepScaling bool
	maxStepBurst        int

	walls     []bool
	levelRand *rand.Rand

	listenerForwardX float64
	listenerForwardY float64

	pixelBuf []byte
	// audio removed

	autoWalk           bool
	autoWalkDeadline   time.Time
	autoWalkRand       *rand.Rand
	autoWalkDirX       float64
	autoWalkDirY       float64
	autoWalkFrameCount int

	visibleStamp []uint32
	visibleGen   uint32
	lastVisCX    int
	lastVisCY    int

	gpuSolver      *openCLWaveSolver
	impulsesActive bool
}

// newGame constructs a fully initialized Game instance.
func newGame() *Game {
	g := &Game{
		field:               newWaveField(w, h),
		ex:                  float64(w / 2),
		ey:                  float64(h / 2),
		levelRand:           rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:               make([]bool, w*h),
		listenerForwardX:    0,
		listenerForwardY:    -1,
		pixelBuf:            make([]byte, w*h*4),
		autoWalkRand:        rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
		simStepMultiplier:   defaultSimMultiplier,
		adaptiveStepScaling: *adaptiveStepScalingFlag,
		maxStepBurst:        *maxStepBurstFlag,
	}
	// Audio removed
	if solver, err := newOpenCLWaveSolver(w, h); err != nil {
		log.Fatalf("OpenCL initialization failed: %v", err)
	} else {
		log.Printf("OpenCL solver enabled (device: %s)", solver.DeviceName())
		g.gpuSolver = solver
	}
	g.generateWalls()
	g.lastVisCX, g.lastVisCY = -1, -1
	return g
}

// Update advances the simulation, produces optional audio, and refreshes visibility data.
func (g *Game) Update() error {
	dx, dy := g.movementVector()
	oldX, oldY := g.ex, g.ey
	g.ex = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.ex+dx))
	g.ey = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.ey+dy))
	if g.isWall(int(g.ex), int(g.ey)) {
		g.ex, g.ey = oldX, oldY
	}

	g.handleDebugControls()

	moving := dx != 0 || dy != 0
	impulsesFired := false
	if moving {
		length := math.Hypot(dx, dy)
		if length > 0 {
			g.listenerForwardX = dx / length
			g.listenerForwardY = dy / length
		}
		g.stepTimer++
		if g.stepTimer >= stepDelay {
			g.stepTimer = 0
			for y := -emitterRad; y <= emitterRad; y++ {
				for x := -emitterRad; x <= emitterRad; x++ {
					if x*x+y*y <= emitterRad*emitterRad {
						cx := int(g.ex) + x
						cy := int(g.ey) + y
						if cx <= 0 || cx >= w-1 || cy <= 0 || cy >= h-1 {
							continue
						}
						if g.isWall(cx, cy) {
							continue
						}
						if g.field.setCurr(cx, cy, stepImpulseStrength) {
							impulsesFired = true
						}
					}
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

	g.impulsesActive = impulsesFired

	actualTPS := ebiten.ActualTPS()
	if actualTPS < 1 {
		actualTPS = defaultTPS
	}
	baseSteps := g.simStepMultiplier
	steps := baseSteps
	if g.adaptiveStepScaling {
		g.physicsAccumulator += g.simStepsPerSecond() / actualTPS
		steps = int(g.physicsAccumulator)
		if steps < 1 {
			steps = 1
		}
		if g.maxStepBurst > 0 {
			burstLimit := baseSteps * g.maxStepBurst
			if steps > burstLimit {
				steps = burstLimit
			}
		}
		g.physicsAccumulator -= float64(steps)
	} else {
		g.physicsAccumulator = 0
	}
	simStart := time.Now()
	if err := g.gpuSolver.Step(g.field, g.walls, steps, false); err != nil {
		return err
	}
	g.lastSimDuration = time.Since(simStart)

	if *occludeLineOfSightFlag {
		g.refreshVisibleMask()
	}

	return nil
}
