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

	listenerX float64
	listenerY float64

	emitterX float64
	emitterY float64

	controlEmitter bool

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

	audioCtx      *audio.Context
	audioStream   *centerAudioStream
	audioPlayer   *audio.Player
	audioPressure *audioPressureSource
	audioChunk    []float32
}

const minEmitterStartDistancePixels = 128

// newGame constructs a fully initialized Game instance.
func newGame() *Game {
	g := &Game{
		field:             newWaveField(w, h),
		listenerX:         float64(w / 2),
		listenerY:         float64(h / 2),
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
		var loopSamples []float32
		if audioLoopFlag != nil && *audioLoopFlag != "" {
			samples, err := loadLoopSamples(audioSampleRate, *audioLoopFlag)
			if err != nil {
				log.Printf("Audio loop %q failed to load: %v", *audioLoopFlag, err)
			} else {
				loopSamples = samples
			}
		}
		if enableAudioFlag != nil && *enableAudioFlag {
			ctx := audio.NewContext(audioSampleRate)
			g.audioCtx = ctx
			stream := newCenterAudioStream()
			g.audioStream = stream
			player, err := ctx.NewPlayer(stream)
			if err != nil {
				log.Printf("Audio player creation failed: %v", err)
			} else {
				player.SetBufferSize(audioPlayerBufferLatency)
				player.Play()
				g.audioPlayer = player
			}
			if loopSamples != nil {
				g.audioPressure = newAudioPressureSource(loopSamples)
			}
		}
	}

	g.generateWalls()
	g.randomizeEmitterStart(minEmitterStartDistancePixels)
	g.lastVisCX, g.lastVisCY = -1, -1
	return g
}

func (g *Game) randomizeEmitterStart(minDist int) {
	if g.levelRand == nil {
		g.levelRand = rand.New(rand.NewSource(time.Now().UnixNano() + 6))
	}
	minX := emitterRad
	maxX := w - emitterRad - 1
	minY := emitterRad
	maxY := h - emitterRad - 1
	if minX >= maxX || minY >= maxY {
		return
	}
	centerX := int(math.Round(g.listenerX))
	centerY := int(math.Round(g.listenerY))
	minDist2 := float64(minDist * minDist)

	pick := func() (int, int) {
		x := minX + g.levelRand.Intn(maxX-minX+1)
		y := minY + g.levelRand.Intn(maxY-minY+1)
		return x, y
	}

	for attempt := 0; attempt < 256; attempt++ {
		x, y := pick()
		dx := float64(x - centerX)
		dy := float64(y - centerY)
		if dx*dx+dy*dy < minDist2 {
			continue
		}
		if !g.isWall(x, y) {
			g.emitterX, g.emitterY = float64(x), float64(y)
			return
		}
	}

	// Fallback: scan for any valid non-wall cell satisfying the distance constraint.
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			dx := float64(x - centerX)
			dy := float64(y - centerY)
			if dx*dx+dy*dy < minDist2 {
				continue
			}
			if g.isWall(x, y) {
				continue
			}
			g.emitterX, g.emitterY = float64(x), float64(y)
			return
		}
	}
}

// Update advances the simulation, produces optional audio, and refreshes visibility data.
func (g *Game) Update() error {
	g.handleControlToggle()
	dx, dy := g.movementVector()
	if g.controlEmitter {
		oldX, oldY := g.emitterX, g.emitterY
		g.emitterX = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.emitterX+dx))
		g.emitterY = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.emitterY+dy))
		if g.isWall(int(g.emitterX), int(g.emitterY)) {
			g.emitterX, g.emitterY = oldX, oldY
		}
	} else {
		oldX, oldY := g.listenerX, g.listenerY
		g.listenerX = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.listenerX+dx))
		g.listenerY = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.listenerY+dy))
		if g.isWall(int(g.listenerX), int(g.listenerY)) {
			g.listenerX, g.listenerY = oldX, oldY
		}
	}

	g.handleDebugControls()

	movingListener := !g.controlEmitter && (dx != 0 || dy != 0)
	impulsesFired := false
	if movingListener {
		length := math.Hypot(dx, dy)
		if length > 0 {
			g.listenerForwardX = dx / length
			g.listenerForwardY = dy / length
		}
		g.stepTimer++
		if g.stepTimer >= stepDelay {
			g.stepTimer = 0
			if !(*disableWalkingPulsesFlag) {
				baseX := int(g.listenerX)
				baseY := int(g.listenerY)
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
	var emitterData *audioEmitterData
	if g.audioPressure != nil && steps > 0 {
		if samples := g.fillAudioChunk(steps); len(samples) > 0 {
			if idx, ok := g.emitterAudioIndex(); ok {
				emitterData = &audioEmitterData{index: idx, samples: samples}
			}
		}
	}

	listenerIndex, _ := g.listenerAudioIndex()
	if err := g.gpuSolver.Step(g.field, g.walls, steps, g.wallsDirty, *showWallsFlag, *lastFrameOnlyFlag, *occludeLineOfSightFlag, visibleStamp, visibleGen, listenerIndex, emitterData); err != nil {
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

func (g *Game) fillAudioChunk(size int) []float32 {
	if g.audioPressure == nil || size <= 0 {
		return nil
	}
	if cap(g.audioChunk) < size {
		g.audioChunk = make([]float32, size)
	}
	g.audioChunk = g.audioChunk[:size]
	g.audioPressure.fillChunk(g.audioChunk)
	gain := 1.0
	if emitterGainFlag != nil {
		gain = *emitterGainFlag
	}
	if gain != 1.0 {
		gain32 := float32(gain)
		for i, v := range g.audioChunk {
			g.audioChunk[i] = v * gain32
		}
	}
	return g.audioChunk
}

func (g *Game) emitterAudioIndex() (int32, bool) {
	x := int(g.emitterX)
	y := int(g.emitterY)
	if x < 0 || x >= w || y < 0 || y >= h {
		return -1, false
	}
	return int32(y*w + x), true
}

func (g *Game) listenerAudioIndex() (int32, bool) {
	x := int(g.listenerX)
	y := int(g.listenerY)
	if x < 0 || x >= w || y < 0 || y >= h {
		return -1, false
	}
	return int32(y*w + x), true
}

func (g *Game) controlModeLabel() string {
	if g.controlEmitter {
		return "emitter"
	}
	return "listener"
}
