package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// Game encapsulates the full simulation state, rendering buffers, and audio pipeline.
type Game struct {
	field *waveField

	ex float64
	ey float64

	stepTimer         int
	lastSimDuration   time.Duration
	simStepMultiplier int

	walls     []bool
	levelRand *rand.Rand

	listenerForwardX float64
	listenerForwardY float64

	autoWalk           bool
	autoWalkDeadline   time.Time
	lastSampleLog      time.Time
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
	wallsDirty     bool

	audioCtx    *audio.Context
	audioStream *centerAudioStream
	audioPlayer *audio.Player
}

// newGame constructs a fully initialized Game instance.
func newGame() *Game {
	g := &Game{
		field:             newWaveField(w, h),
		ex:                float64(w / 2),
		ey:                float64(h / 2),
		levelRand:         rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:             make([]bool, w*h),
		listenerForwardX:  0,
		listenerForwardY:  -1,
		autoWalkRand:      rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
		simStepMultiplier: defaultSimMultiplier,
	}
	if solver, err := newOpenCLWaveSolver(w, h); err != nil {
		log.Fatalf("OpenCL initialization failed: %v", err)
	} else {
		log.Printf("OpenCL solver enabled (device: %s)", solver.DeviceName())
		g.gpuSolver = solver
		if enableAudioFlag != nil && *enableAudioFlag {
			ctx := audio.NewContext(audioSampleRate)
			g.audioCtx = ctx
			stream := newCenterAudioStream()
			g.audioStream = stream
			if player, err := ctx.NewPlayer(stream); err != nil {
				log.Printf("Audio player creation failed: %v", err)
			} else {
				g.audioPlayer = player
				g.audioPlayer.SetBufferSize(audioPlayerBufferLatency)
				g.audioPlayer.Play()
			}
		}
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
			baseX := int(g.ex)
			baseY := int(g.ey)
			for _, offset := range emitterFootprint {
				cx := baseX + offset.dx
				cy := baseY + offset.dy
				if cx <= 0 || cx >= w-1 || cy <= 0 || cy >= h-1 {
					continue
				}
				if g.isWall(cx, cy) {
					continue
				}
				if g.field.queueImpulse(cx, cy, stepImpulseStrength) {
					impulsesFired = true
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

	g.impulsesActive = impulsesFired

	if *occludeLineOfSightFlag {
		g.refreshVisibleMask()
	}

	steps := g.simStepMultiplier
	simStart := time.Now()
	var visibleStamp []uint32
	var visibleGen uint32
	if *occludeLineOfSightFlag {
		visibleStamp = g.visibleStamp
		visibleGen = g.visibleGen
	}
	if err := g.gpuSolver.Step(g.field, g.walls, steps, g.wallsDirty, *showWallsFlag, *occludeLineOfSightFlag, visibleStamp, visibleGen); err != nil {
		return err
	}
	if g.audioStream != nil && g.gpuSolver != nil {
		g.audioStream.SetSample(g.gpuSolver.CenterSample())
	}
	if captureStepSamplesFlag != nil && *captureStepSamplesFlag && g.gpuSolver != nil {
		if samples := g.gpuSolver.CenterSamples(); len(samples) > 0 {
			if g.audioStream != nil {
				g.audioStream.Enqueue(samples)
			}
			g.logCapturedCenterSamples(samples)
		}
	}
	g.wallsDirty = false
	g.lastSimDuration = time.Since(simStart)

	return nil
}

func (g *Game) logCapturedCenterSamples(samples []float32) {
	if len(samples) == 0 {
		return
	}
	now := time.Now()
	if now.Sub(g.lastSampleLog) < sampleCaptureLogInterval {
		return
	}
	minVal := samples[0]
	maxVal := samples[0]
	var sum float32
	for _, v := range samples {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
		sum += v
	}
	avg := sum / float32(len(samples))
	last := samples[len(samples)-1]
	log.Printf("Captured %d center samples (min %.3f max %.3f avg %.3f last %.3f)",
		len(samples), minVal, maxVal, avg, last)
	g.lastSampleLog = now
}
