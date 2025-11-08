package main

import (
	"io"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
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

	workerCount   int
	workerMasks   []workerMask
	workerMu      sync.Mutex
	workerCond    *sync.Cond
	workerStep    int
	workerPending int

	listenerForwardX float64
	listenerForwardY float64

	pixelBuf              []byte
	latestPressureSamples []int16
	pressureSampleIndex   int

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
	workersStarted bool

	audioCtx       *audio.Context
	audioPlayer    *audio.Player
	audioPipe      *io.PipeWriter
	audioPCM       []int16
	audioWriteBuf  []byte
	audioElapsed   float64
	audioNextTime  float64
	audioSampleDur float64
	audioDisabled  bool
}

// newGame constructs a fully initialized Game instance.
func newGame(workerCount int, enableOpenCL bool) *Game {
	if workerCount < 1 {
		workerCount = 1
	}
	sampleIndex := defaultPressureSampleIndex(w, h)
	g := &Game{
		field:               newWaveField(w, h),
		ex:                  float64(w / 2),
		ey:                  float64(h / 2),
		levelRand:           rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:               make([]bool, w*h),
		workerCount:         workerCount,
		listenerForwardX:    0,
		listenerForwardY:    -1,
		pixelBuf:            make([]byte, w*h*4),
		autoWalkRand:        rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
		simStepMultiplier:   defaultSimMultiplier,
		adaptiveStepScaling: *adaptiveStepScalingFlag,
		maxStepBurst:        *maxStepBurstFlag,
		pressureSampleIndex: sampleIndex,
		audioDisabled:       *disableAudioFlag,
	}
	g.workerCond = sync.NewCond(&g.workerMu)
	if !g.audioDisabled {
		g.initAudio()
	}
	if enableOpenCL {
		if solver, err := newOpenCLWaveSolver(w, h, sampleIndex); err != nil {
			log.Printf("OpenCL initialization failed: %v", err)
		} else {
			log.Printf("OpenCL solver enabled (device: %s)", solver.DeviceName())
			g.gpuSolver = solver
		}
	}
	if g.gpuSolver == nil {
		g.startWorkers()
	}
	g.generateWalls()
	g.rebuildInteriorMask()
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
						g.field.setCurr(cx, cy, stepImpulseStrength)
					}
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

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
	var producedSamples []int16
	if g.gpuSolver != nil {
		samples, err := g.gpuSolver.Step(g.field, g.walls, steps, false)
		if err != nil {
			log.Printf("OpenCL solver error: %v; falling back to CPU", err)
			g.gpuSolver.Close()
			g.gpuSolver = nil
			g.startWorkers()
			g.stepWaveCPUBatch(steps)
		} else {
			g.setPressureSamples(samples)
		}
	} else {
		g.stepWaveCPUBatch(steps)
	}
	if !g.audioDisabled {
		producedSamples = g.latestPressureSamples
	}
	g.lastSimDuration = time.Since(simStart)

	if !g.audioDisabled && producedSamples != nil {
		sourceRate := g.simStepsPerSecond()
		actual := ebiten.ActualTPS()
		if actual < 1 {
			actual = defaultTPS
		}
		if g.adaptiveStepScaling {
			sourceRate = float64(steps) * actual
		}
		g.streamAudioSamples(producedSamples, sourceRate)
	}

	if *occludeLineOfSightFlag {
		g.refreshVisibleMask()
	}

	return nil
}
