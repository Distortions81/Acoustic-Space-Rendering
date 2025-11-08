package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image/color"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

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

var showWallsFlag = flag.Bool("show-walls", false, "render wall geometry overlays")
var recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")
var occludeLineOfSightFlag = flag.Bool("occlude-line-of-sight", true, "hide regions that are not in the listener's line of sight when rendering")
var fovDegreesFlag = flag.Float64("fov-deg", 90.0, "field of view angle for LOS (degrees)")
var threadCountFlag = flag.Int("threads", 0, "number of worker threads; 0 auto-detects")
var debugFlag = flag.Bool("debug", false, "show FPS and simulation speed overlay")
var useOpenCLFlag = flag.Bool("use-opencl", true, "attempt to run the wave simulation via OpenCL (build with -tags opencl)")
var adaptiveStepScalingFlag = flag.Bool("scale-steps-with-tps", false, "scale the per-frame physics work based on ActualTPS instead of using a fixed batch size")
var maxStepBurstFlag = flag.Int("max-step-burst", 4, "maximum multiple of the base physics step count to execute when recovering from lag while step scaling is enabled (0 disables the clamp)")

type intPoint struct {
	x int
	y int
}

var losPerimeterTargets = buildLOSPerimeterTargets()

func buildLOSPerimeterTargets() []intPoint {
	points := make([]intPoint, 0, 2*(w+h))
	for x := 0; x < w; x++ {
		points = append(points, intPoint{x: x, y: 0})
		points = append(points, intPoint{x: x, y: h - 1})
	}
	for y := 1; y < h-1; y++ {
		points = append(points, intPoint{x: 0, y: y})
		points = append(points, intPoint{x: w - 1, y: y})
	}
	return points
}

type waveField struct {
	width, height int
	curr          []float32
	prev          []float32
	next          []float32
}

func defaultPressureSampleIndex(width, height int) int {
	return (height/2)*width + (width / 2)
}

func floatToPCM16(v float32) int16 {
	if v > 1 {
		v = 1
	} else if v < -1 {
		v = -1
	}
	return int16(math.Round(float64(v) * float64(pcm16MaxValue)))
}

func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr: make([]float32, width*height),
		prev: make([]float32, width*height),
		next: make([]float32, width*height),
	}
}

func (f *waveField) setCurr(x, y int, value float32) {
	f.curr[y*f.width+x] = value
}

func (f *waveField) zeroCell(x, y int) {
	idx := y*f.width + x
	f.curr[idx] = 0
	f.prev[idx] = 0
	f.next[idx] = 0
}

func (f *waveField) readCurr(x, y int) float32 {
	return f.curr[y*f.width+x]
}

func (f *waveField) swap() {
	f.prev, f.curr, f.next = f.curr, f.next, f.prev
}

func (f *waveField) zeroBoundaries() {
	lastRow := f.height - 1
	lastCol := f.width - 1
	reflect := float32(boundaryReflect)
	// Top and bottom rows
	for x := 0; x < f.width; x++ {
		top := f.next[1*f.width+x]
		bottom := f.next[(lastRow-1)*f.width+x]
		f.next[0*f.width+x] = -top * reflect
		f.next[lastRow*f.width+x] = -bottom * reflect
	}
	// Left and right columns
	for y := 1; y < lastRow; y++ {
		left := f.next[y*f.width+1]
		right := f.next[y*f.width+lastCol-1]
		f.next[y*f.width+0] = -left * reflect
		f.next[y*f.width+lastCol] = -right * reflect
	}
}

type span struct{ start, end int } // inclusive [start,end]

type rowMask struct {
	y     int
	spans []span
}

type workerMask struct {
	rows []rowMask
}

type rowCache struct {
	center []float32
	prev   []float32
	top    []float32
	bottom []float32
}

func newRowCache(width int) *rowCache {
	return &rowCache{
		center: make([]float32, width),
		prev:   make([]float32, width),
		top:    make([]float32, width),
		bottom: make([]float32, width),
	}
}

func (g *Game) waveWorkerLoop(index int) {
	cache := newRowCache(g.field.width)
	lastStep := 0
	g.workerMu.Lock()
	for {
		for g.workerStep == lastStep {
			g.workerCond.Wait()
		}
		lastStep = g.workerStep
		var mask workerMask
		if index < len(g.workerMasks) {
			mask = g.workerMasks[index]
		}
		g.workerMu.Unlock()

		if len(mask.rows) > 0 {
			processMask(g.field, &mask, cache)
		}

		g.workerMu.Lock()
		g.workerPending--
		if g.workerPending == 0 {
			g.workerCond.Broadcast()
		}
	}
}

func processMask(field *waveField, mask *workerMask, _ *rowCache) {
	// Compute directly from flat slices and contiguous spans.
	// Inner loops are written in fixed-stride form to encourage auto-vectorization.
	width := field.width
	wd := waveDamp32
	ws := waveSpeed32
	for _, row := range mask.rows {
		y := row.y
		rowBase := y * width
		topBase := (y - 1) * width
		bottomBase := (y + 1) * width
		// set row boundary guard values; main boundary reflection still applied separately
		field.next[rowBase+0] = 0
		field.next[rowBase+width-1] = 0

		// Pre-slice rows to help bounds-check elimination
		center := field.curr[rowBase : rowBase+width]
		prev := field.prev[rowBase : rowBase+width]
		top := field.curr[topBase : topBase+width]
		bottom := field.curr[bottomBase : bottomBase+width]
		nextRow := field.next[rowBase : rowBase+width]

		for _, sp := range row.spans {
			// clamp spans to interior just in case
			start := sp.start
			if start < 1 {
				start = 1
			}
			end := sp.end
			if end > width-2 {
				end = width - 2
			}

			// 4-wide unrolled loop
			x := start
			for ; x+3 <= end; x += 4 {
				// lane 0
				c0 := center[x]
				lap0 := center[x-1] + center[x+1] + top[x] + bottom[x] - 4*c0
				nextRow[x] = ((2*c0 - prev[x]) + ws*lap0) * wd

				// lane 1
				x1 := x + 1
				c1 := center[x1]
				lap1 := center[x1-1] + center[x1+1] + top[x1] + bottom[x1] - 4*c1
				nextRow[x1] = ((2*c1 - prev[x1]) + ws*lap1) * wd

				// lane 2
				x2 := x + 2
				c2 := center[x2]
				lap2 := center[x2-1] + center[x2+1] + top[x2] + bottom[x2] - 4*c2
				nextRow[x2] = ((2*c2 - prev[x2]) + ws*lap2) * wd

				// lane 3
				x3 := x + 3
				c3 := center[x3]
				lap3 := center[x3-1] + center[x3+1] + top[x3] + bottom[x3] - 4*c3
				nextRow[x3] = ((2*c3 - prev[x3]) + ws*lap3) * wd
			}
			// tail
			for ; x <= end; x++ {
				c := center[x]
				lap := center[x-1] + center[x+1] + top[x] + bottom[x] - 4*c
				nextRow[x] = ((2*c - prev[x]) + ws*lap) * wd
			}
		}
	}
}

func convertRow(src []float32, dst []float32) {
	copy(dst, src)
}

func assignRowMasks(workerCount int, rows []rowMask) []workerMask {
	if workerCount < 1 {
		workerCount = 1
	}
	masks := make([]workerMask, workerCount)
	for idx, row := range rows {
		workerIdx := idx % workerCount
		masks[workerIdx].rows = append(masks[workerIdx].rows, row)
	}
	return masks
}

func (g *Game) startWorkers() {
	if g.workersStarted {
		return
	}
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	if g.workerCond == nil {
		g.workerCond = sync.NewCond(&g.workerMu)
	}
	g.workersStarted = true
	for i := 0; i < g.workerCount; i++ {
		go g.waveWorkerLoop(i)
	}
}

type Game struct {
	field                 *waveField
	ex, ey                float64
	stepTimer             int
	physicsAccumulator    float64
	lastSimDuration       time.Duration
	simStepMultiplier     int
	adaptiveStepScaling   bool
	maxStepBurst          int
	walls                 []bool
	levelRand             *rand.Rand
	workerCount           int
	workerMasks           []workerMask
	maskDirty             bool
	workerMu              sync.Mutex
	workerCond            *sync.Cond
	workerStep            int
	workerPending         int
	listenerForwardX      float64
	listenerForwardY      float64
	pixelBuf              []byte
	latestPressureSamples []int16
	pressureSampleIndex   int
	autoWalk              bool
	autoWalkDeadline      time.Time
	autoWalkRand          *rand.Rand
	autoWalkDirX          float64
	autoWalkDirY          float64
	autoWalkFrameCount    int
	visibleStamp          []uint32
	visibleGen            uint32
	// cache the last integer listener position used for LOS to avoid recomputing
	lastVisCX      int
	lastVisCY      int
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
}

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
		maskDirty:           true,
		listenerForwardX:    0,
		listenerForwardY:    -1,
		pixelBuf:            make([]byte, w*h*4),
		autoWalkRand:        rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
		simStepMultiplier:   defaultSimMultiplier,
		adaptiveStepScaling: *adaptiveStepScalingFlag,
		maxStepBurst:        *maxStepBurstFlag,
		pressureSampleIndex: sampleIndex,
	}
	g.workerCond = sync.NewCond(&g.workerMu)
	g.initAudio()
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

func (g *Game) ensurePressureSampleCapacity(n int) {
	if n <= 0 {
		if g.latestPressureSamples != nil {
			g.latestPressureSamples = g.latestPressureSamples[:0]
		}
		return
	}
	if cap(g.latestPressureSamples) < n {
		g.latestPressureSamples = make([]int16, n)
	} else {
		g.latestPressureSamples = g.latestPressureSamples[:n]
	}
}

func (g *Game) setPressureSamples(samples []int16) {
	g.ensurePressureSampleCapacity(len(samples))
	copy(g.latestPressureSamples, samples)
}

func (g *Game) initAudio() {
	g.audioSampleDur = 1.0 / float64(audioSampleRate)
	ctx := audio.CurrentContext()
	if ctx == nil {
		ctx = audio.NewContext(audioSampleRate)
	} else if ctx.SampleRate() != audioSampleRate {
		log.Printf("audio context sample rate %d does not match expected %d; disabling audio", ctx.SampleRate(), audioSampleRate)
		return
	}
	pr, pw := io.Pipe()
	player, err := ctx.NewPlayer(pr)
	if err != nil {
		log.Printf("creating audio player failed: %v", err)
		_ = pr.Close()
		_ = pw.Close()
		return
	}
	player.SetBufferSize(audioBufferDuration)
	g.audioCtx = ctx
	g.audioPlayer = player
	g.audioPipe = pw
	g.audioElapsed = 0
	g.audioNextTime = 0
	g.audioPCM = g.audioPCM[:0]
	player.Play()
}

func (g *Game) flushAudioBuffer() {
	if len(g.audioPCM) == 0 {
		return
	}
	if g.audioPipe == nil {
		g.audioPCM = g.audioPCM[:0]
		return
	}
	byteLen := len(g.audioPCM) * 2
	if cap(g.audioWriteBuf) < byteLen {
		g.audioWriteBuf = make([]byte, byteLen)
	}
	buf := g.audioWriteBuf[:byteLen]
	for i, sample := range g.audioPCM {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(sample))
	}
	if _, err := g.audioPipe.Write(buf); err != nil {
		log.Printf("audio stream write failed: %v", err)
		_ = g.audioPipe.Close()
		g.audioPipe = nil
	} else if g.audioPlayer != nil && !g.audioPlayer.IsPlaying() {
		g.audioPlayer.Play()
	}
	g.audioPCM = g.audioPCM[:0]
}

func (g *Game) streamAudioSamples(samples []int16, sourceRate float64) {
	if sourceRate <= 0 {
		sourceRate = float64(audioSampleRate)
	}
	chunkDuration := 0.0
	if len(samples) > 0 {
		chunkDuration = float64(len(samples)) / sourceRate
	}
	chunkStart := g.audioElapsed
	chunkEnd := chunkStart + chunkDuration
	if len(samples) == 0 {
		g.audioElapsed = chunkEnd
		if g.audioNextTime < g.audioElapsed {
			g.audioNextTime = g.audioElapsed
		}
		return
	}
	if g.audioNextTime < chunkStart {
		g.audioNextTime = chunkStart
	}
	g.audioPCM = g.audioPCM[:0]
	if math.Abs(sourceRate-float64(audioSampleRate)) < 1e-6 {
		startIndex := int(math.Round((g.audioNextTime - chunkStart) * sourceRate))
		if startIndex < 0 {
			startIndex = 0
		}
		if startIndex >= len(samples) {
			startIndex = len(samples) - 1
		}
		for idx := startIndex; idx < len(samples) && g.audioNextTime < chunkEnd; idx++ {
			sample := int(samples[idx])
			if sample > pcm16MaxValue {
				sample = pcm16MaxValue
			} else if sample < pcm16MinValue {
				sample = pcm16MinValue
			}
			v := int16(sample)
			g.audioPCM = append(g.audioPCM, v, v)
			g.audioNextTime += g.audioSampleDur
		}
	} else {
		for g.audioNextTime < chunkEnd {
			relTime := g.audioNextTime - chunkStart
			srcPos := relTime * sourceRate
			idx := int(srcPos)
			if idx < 0 {
				idx = 0
			}
			if idx >= len(samples) {
				idx = len(samples) - 1
			}
			sample := float64(samples[idx])
			if idx+1 < len(samples) {
				frac := srcPos - float64(idx)
				nextSample := float64(samples[idx+1])
				sample += (nextSample - sample) * frac
			}
			if sample > pcm16MaxValue {
				sample = pcm16MaxValue
			} else if sample < pcm16MinValue {
				sample = pcm16MinValue
			}
			v := int16(math.Round(sample))
			g.audioPCM = append(g.audioPCM, v, v)
			g.audioNextTime += g.audioSampleDur
		}
	}
	g.audioElapsed = chunkEnd
	if g.audioPipe == nil {
		g.audioPCM = g.audioPCM[:0]
		if g.audioNextTime < g.audioElapsed {
			g.audioNextTime = g.audioElapsed
		}
		return
	}
	g.flushAudioBuffer()
	if g.audioNextTime < g.audioElapsed {
		g.audioNextTime = g.audioElapsed
	}
}

func (g *Game) samplePressureAtIndex() int16 {
	if g.pressureSampleIndex < 0 || g.pressureSampleIndex >= len(g.field.curr) {
		return 0
	}
	return floatToPCM16(g.field.curr[g.pressureSampleIndex])
}

func (g *Game) rebuildInteriorMask() {
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	rows := make([]rowMask, 0, h-2)
	for y := 1; y < h-1; y++ {
		base := y * w
		spans := make([]span, 0, 8)
		in := false
		start := 0
		for x := 1; x < w-1; x++ {
			blocked := g.walls[base+x]
			if !blocked && !in {
				in = true
				start = x
			}
			if (blocked || x == w-2) && in {
				end := x - 1
				if x == w-2 && !blocked {
					end = x
				}
				if end >= start {
					spans = append(spans, span{start: start, end: end})
				}
				in = false
			}
		}
		if len(spans) == 0 {
			continue
		}
		rows = append(rows, rowMask{y: y, spans: spans})
	}
	g.workerMasks = assignRowMasks(g.workerCount, rows)
	g.maskDirty = false
}

func (g *Game) ensureInteriorMask() {
	if g.maskDirty || len(g.workerMasks) != g.workerCount {
		g.rebuildInteriorMask()
	}
}

func (g *Game) enableAutoWalk(duration time.Duration) {
	g.autoWalk = true
	g.autoWalkDeadline = time.Now().Add(duration)
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 3))
	}
	g.autoWalkFrameCount = 0
}

func (g *Game) movementVector() (float64, float64) {
	if g.autoWalk {
		if time.Now().After(g.autoWalkDeadline) {
			g.autoWalk = false
			return 0, 0
		}
		return g.autoWalkVector()
	}
	return g.manualMovementVector()
}

func (g *Game) manualMovementVector() (float64, float64) {
	dx, dy := 0.0, 0.0
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		dy -= moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) {
		dy += moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) {
		dx -= moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) {
		dx += moveSpeed
	}
	if dx != 0 && dy != 0 {
		dx *= 0.7071
		dy *= 0.7071
	}
	return dx, dy
}

func (g *Game) autoWalkVector() (float64, float64) {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 4))
	}
	for attempts := 0; attempts < 5; attempts++ {
		if g.autoWalkFrameCount <= 0 {
			g.randomizeAutoWalkDirection()
		}
		nextX := g.ex + g.autoWalkDirX*moveSpeed
		nextY := g.ey + g.autoWalkDirY*moveSpeed
		if nextX > float64(emitterRad) && nextX < float64(w-emitterRad-1) &&
			nextY > float64(emitterRad) && nextY < float64(h-emitterRad-1) &&
			!g.isWall(int(nextX), int(nextY)) {
			g.autoWalkFrameCount--
			return g.autoWalkDirX * moveSpeed, g.autoWalkDirY * moveSpeed
		}
		g.autoWalkFrameCount = 0
	}
	return 0, 0
}

func (g *Game) randomizeAutoWalkDirection() {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 5))
	}
	angle := g.autoWalkRand.Float64() * 2 * math.Pi
	g.autoWalkDirX = math.Cos(angle)
	g.autoWalkDirY = math.Sin(angle)
	g.autoWalkFrameCount = 20 + g.autoWalkRand.Intn(50)
}

func (g *Game) handleDebugControls() {
	if !*debugFlag {
		return
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyMinus) || inpututil.IsKeyJustPressed(ebiten.KeyKPSubtract) {
		g.adjustSimMultiplier(-simMultiplierStep)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEqual) || inpututil.IsKeyJustPressed(ebiten.KeyKPAdd) {
		g.adjustSimMultiplier(simMultiplierStep)
	}
}

func (g *Game) adjustSimMultiplier(delta int) {
	g.simStepMultiplier += delta
	if g.simStepMultiplier < minSimMultiplier {
		g.simStepMultiplier = minSimMultiplier
	} else if g.simStepMultiplier > maxSimMultiplier {
		g.simStepMultiplier = maxSimMultiplier
	}
}

func (g *Game) simStepsPerSecond() float64 {
	return defaultTPS * float64(g.simStepMultiplier)
}

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
		wallsDirty := g.maskDirty
		samples, err := g.gpuSolver.Step(g.field, g.walls, steps, wallsDirty)
		if err != nil {
			log.Printf("OpenCL solver error: %v; falling back to CPU", err)
			g.gpuSolver.Close()
			g.gpuSolver = nil
			g.startWorkers()
			g.stepWaveCPUBatch(steps)
		} else {
			if wallsDirty {
				g.rebuildInteriorMask()
			}
			g.setPressureSamples(samples)
		}
	} else {
		g.stepWaveCPUBatch(steps)
	}
	producedSamples = g.latestPressureSamples
	g.lastSimDuration = time.Since(simStart)

	if producedSamples != nil {
		sourceRate := g.simStepsPerSecond()
		if g.adaptiveStepScaling {
			sourceRate = float64(steps) * actualTPS
		}
		g.streamAudioSamples(producedSamples, sourceRate)
	}

	// Keep LOS visibility mask updated here to avoid work in Draw.
	if *occludeLineOfSightFlag {
		g.refreshVisibleMask()
	}

	return nil
}

func (g *Game) generateWalls() {
	if len(g.walls) != w*h {
		g.walls = make([]bool, w*h)
	} else {
		for i := range g.walls {
			g.walls[i] = false
		}
	}
	if g.levelRand == nil {
		g.levelRand = rand.New(rand.NewSource(time.Now().UnixNano() + 1))
	}
	for s := 0; s < wallSegments; s++ {
		lengthRange := wallMaxLen - wallMinLen + 1
		if lengthRange <= 0 {
			lengthRange = 1
		}
		length := wallMinLen + g.levelRand.Intn(lengthRange)
		thickness := 1
		if wallThicknessVariance > 0 {
			thickness += g.levelRand.Intn(wallThicknessVariance + 1)
		}
		horizontal := g.levelRand.Intn(2) == 0
		x := g.levelRand.Intn(w-4) + 2
		y := g.levelRand.Intn(h-4) + 2
		dx, dy := 0, 1
		if horizontal {
			dx, dy = 1, 0
		}
		perpX, perpY := dy, dx
		cx, cy := x, y
		for l := 0; l < length; l++ {
			if cx <= 1 || cx >= w-1 || cy <= 1 || cy >= h-1 {
				break
			}
			for t := -thickness; t <= thickness; t++ {
				tx := cx + perpX*t
				ty := cy + perpY*t
				g.trySetWall(tx, ty)
			}
			cx += dx
			cy += dy
		}
	}
	g.maskDirty = true
	// force LOS recomputation next draw
	g.lastVisCX, g.lastVisCY = -1, -1
}

func (g *Game) trySetWall(x, y int) {
	if x <= 1 || x >= w-1 || y <= 1 || y >= h-1 {
		return
	}
	dx := float64(x) - g.ex
	dy := float64(y) - g.ey
	if dx*dx+dy*dy < float64(wallExclusionRadius*wallExclusionRadius) {
		return
	}
	idx := y*w + x
	g.walls[idx] = true
	g.field.zeroCell(x, y)
	g.maskDirty = true
}

func (g *Game) isWall(x, y int) bool {
	if x < 0 || x >= w || y < 0 || y >= h {
		return true
	}
	if len(g.walls) == 0 {
		return false
	}
	return g.walls[y*w+x]
}

func (g *Game) earOffsets() (int, int) {
	fx, fy := g.listenerForwardX, g.listenerForwardY
	if fx == 0 && fy == 0 {
		fy = -1
	}
	earVecX := -fy
	earVecY := fx
	length := math.Hypot(earVecX, earVecY)
	if length == 0 {
		return earOffsetCells, 0
	}
	scale := float64(earOffsetCells) / length
	ox := int(math.Round(earVecX * scale))
	oy := int(math.Round(earVecY * scale))
	if ox == 0 && oy == 0 {
		if math.Abs(earVecX) >= math.Abs(earVecY) {
			if earVecX >= 0 {
				ox = earOffsetCells
			} else {
				ox = -earOffsetCells
			}
		} else {
			if earVecY >= 0 {
				oy = earOffsetCells
			} else {
				oy = -earOffsetCells
			}
		}
	}
	return ox, oy
}

func clampCoord(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func (g *Game) refreshVisibleMask() {
	if len(g.visibleStamp) != w*h {
		g.visibleStamp = make([]uint32, w*h)
	}
	cx := clampCoord(int(math.Round(g.ex)), 0, w-1)
	cy := clampCoord(int(math.Round(g.ey)), 0, h-1)
	// If the integer listener position hasn't changed, keep prior mask
	if g.lastVisCX == cx && g.lastVisCY == cy {
		return
	}
	// Bump generation to avoid clearing the whole mask each time.
	if g.visibleGen == ^uint32(0) {
		// Rare wraparound: reset stamps to zero.
		for i := range g.visibleStamp {
			g.visibleStamp[i] = 0
		}
		g.visibleGen = 1
	} else {
		g.visibleGen++
	}
	g.visibleStamp[cy*w+cx] = g.visibleGen
	// Compute forward vector and cone parameters
	fx, fy := g.listenerForwardX, g.listenerForwardY
	if fx == 0 && fy == 0 {
		fx, fy = 0, -1 // default up if no movement
	}
	mag := math.Hypot(fx, fy)
	if mag == 0 {
		fx, fy = 0, -1
		mag = 1
	}
	fx /= mag
	fy /= mag
	// Clamp FOV to [1,180] degrees to keep the cone well-defined with our
	// cosine-squared check (values > 180 would require a different predicate).
	fovDeg := *fovDegreesFlag
	if fovDeg < 1 {
		fovDeg = 1
	} else if fovDeg > 180 {
		fovDeg = 180
	}
	halfAngleRad := fovDeg * math.Pi / 180.0 / 2.0
	cosHalf := math.Cos(halfAngleRad)
	cosHalfSq := cosHalf * cosHalf
	// Use symmetrical shadowcasting for efficient FOV computation with directional cone filter.
	// Limit radius to the farthest boundary from the listener to avoid extra work.
	maxLeft := cx
	maxRight := (w - 1) - cx
	maxUp := cy
	maxDown := (h - 1) - cy
	radius := maxLeft
	if maxRight > radius {
		radius = maxRight
	}
	if maxUp > radius {
		radius = maxUp
	}
	if maxDown > radius {
		radius = maxDown
	}
	g.computeFOVShadow(cx, cy, radius, fx, fy, cosHalfSq)
	// Safety fallback: if the new algorithm produced suspiciously few visible
	// cells (e.g., due to a bug or extreme occlusion), fall back to perimeter rays.
	visCount := 0
	// Count within screen to avoid allocation; early exit when large enough
	for i := 0; i < w*h; i++ {
		if g.visibleStamp[i] == g.visibleGen {
			visCount++
			if visCount > 128 {
				break
			}
		}
	}
	if visCount <= 1 { // only the origin was set
		// Fallback perimeter rays constrained to the cone
		for _, target := range losPerimeterTargets {
			vx := float64(target.x - cx)
			vy := float64(target.y - cy)
			dot := vx*fx + vy*fy
			// Forward-only and within cone using squared comparison
			if dot <= 0 || dot*dot < (vx*vx+vy*vy)*cosHalfSq {
				continue
			}
			g.castVisibilityRay(cx, cy, target.x, target.y)
		}
	}
	g.lastVisCX, g.lastVisCY = cx, cy
}

// computeFOVShadow computes visible tiles using symmetrical shadowcasting.
func (g *Game) computeFOVShadow(cx, cy, radius int, fx, fy float64, cosHalfSq float64) {
	// Octant transforms for symmetrical shadowcasting
	// Standard octant transforms from RogueBasin symmetrical shadowcasting
	oct := [8][4]int{
		{1, 0, 0, 1},   // E-SE
		{0, 1, 1, 0},   // SE-S
		{-1, 0, 0, 1},  // W-SW
		{0, 1, -1, 0},  // SW-S
		{-1, 0, 0, -1}, // W-NW
		{0, -1, -1, 0}, // NW-N
		{1, 0, 0, -1},  // E-NE
		{0, -1, 1, 0},  // NE-E
	}
	for i := 0; i < 8; i++ {
		g.castLight(cx, cy, 1, 1.0, 0.0, radius, oct[i][0], oct[i][1], oct[i][2], oct[i][3], fx, fy, cosHalfSq)
	}
}

// castLight recursively scans one octant.
func (g *Game) castLight(cx, cy, row int, startSlope, endSlope float64, radius int, xx, xy, yx, yy int, fx, fy float64, cosHalfSq float64) {
	if startSlope < endSlope {
		return
	}
	radiusSq := radius * radius
	for i := row; i <= radius; i++ {
		blocked := false
		newStart := 0.0
		// For each cell in the row from left to right within the octant
		for dx := -i; dx <= 0; dx++ {
			dy := -i
			lSlope := (float64(dx) - 0.5) / (float64(dy) + 0.5)
			rSlope := (float64(dx) + 0.5) / (float64(dy) - 0.5)
			if rSlope > startSlope {
				continue
			}
			if lSlope < endSlope {
				break
			}
			X := cx + dx*xx + dy*xy
			Y := cy + dx*yx + dy*yy
			if X < 0 || X >= w || Y < 0 || Y >= h {
				continue
			}
			distSq := dx*dx + dy*dy
			if distSq <= radiusSq {
				// Directional cone filter: forward-only and within half-angle
				vx := float64(X - cx)
				vy := float64(Y - cy)
				dot := vx*fx + vy*fy
				r2 := (vx*vx + vy*vy)
				if dot > 0 && dot*dot >= r2*cosHalfSq {
					g.visibleStamp[Y*w+X] = g.visibleGen
				}
			}
			wall := g.isWall(X, Y)
			if blocked {
				if wall {
					// still in shadow
					newStart = rSlope
					continue
				} else {
					// shadow ends
					blocked = false
					startSlope = newStart
				}
			} else {
				if wall && i < radius {
					// enter a shadow; recurse for the part before the wall
					blocked = true
					g.castLight(cx, cy, i+1, startSlope, lSlope, radius, xx, xy, yx, yy, fx, fy, cosHalfSq)
					newStart = rSlope
				}
			}
		}
		if blocked {
			// entire row is blocked in this octant
			break
		}
	}
}

func (g *Game) castVisibilityRay(x0, y0, x1, y1 int) {
	dx := int(math.Abs(float64(x1 - x0)))
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -int(math.Abs(float64(y1 - y0)))
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		if x0 < 0 || x0 >= w || y0 < 0 || y0 >= h {
			break
		}
		idx := y0*w + x0
		g.visibleStamp[idx] = g.visibleGen
		if g.isWall(x0, y0) && !(x0 == x1 && y0 == y1) {
			break
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func (g *Game) stepWaveCPU() {
	g.ensureInteriorMask()
	g.workerMu.Lock()
	g.workerPending = g.workerCount
	g.workerStep++
	g.workerCond.Broadcast()
	for g.workerPending > 0 {
		g.workerCond.Wait()
	}
	g.workerMu.Unlock()
	g.field.zeroBoundaries()
	g.field.swap()
}

func (g *Game) stepWaveCPUBatch(steps int) {
	if steps <= 0 {
		g.ensurePressureSampleCapacity(0)
		return
	}
	g.ensurePressureSampleCapacity(steps)
	for i := 0; i < steps; i++ {
		g.stepWaveCPU()
		g.latestPressureSamples[i] = g.samplePressureAtIndex()
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	if len(g.pixelBuf) != w*h*4 {
		g.pixelBuf = make([]byte, w*h*4)
	}
	img := g.pixelBuf
	showWalls := *showWallsFlag
	occludeLOS := *occludeLineOfSightFlag
	for i := 0; i < w*h; i++ {
		base := i * 4
		if occludeLOS && (len(g.visibleStamp) == w*h) && !(g.visibleStamp[i] == g.visibleGen) {
			img[base] = 0
			img[base+1] = 0
			img[base+2] = 0
			img[base+3] = 255
			continue
		}
		if showWalls && len(g.walls) > 0 && g.walls[i] {
			img[base] = 30
			img[base+1] = 40
			img[base+2] = 80
			img[base+3] = 255
			continue
		}
		x := i % w
		y := i / w
		v := g.field.readCurr(x, y)
		v = float32(math.Max(-1, math.Min(1, float64(v))))
		intensity := byte(math.Abs(float64(v)) * 255)
		img[base] = intensity
		img[base+1] = intensity
		img[base+2] = intensity
		img[base+3] = 255
	}
	screen.WritePixels(img)

	for y := -emitterRad; y <= emitterRad; y++ {
		for x := -emitterRad; x <= emitterRad; x++ {
			cx := int(g.ex) + x
			cy := int(g.ey) + y
			if cx >= 0 && cx < w && cy >= 0 && cy < h {
				screen.Set(cx, cy, color.RGBA{255, 0, 0, 255})
			}
		}
	}
	g.drawEarIndicators(screen, int(g.ex), int(g.ey))

	if *debugFlag {
		fps := ebiten.ActualFPS()
		tps := ebiten.ActualTPS()
		if tps < 0 {
			tps = 0
		}
		simMultiplier := 0.0
		if defaultTPS > 0 {
			simMultiplier = tps / defaultTPS
		}
		simMS := g.lastSimDuration.Seconds() * 1000
		simSteps := g.simStepsPerSecond()
		debugMsg := fmt.Sprintf("FPS: %.1f\nSim speed: %.2fx (%.1f TPS)\nSim steps: %.1f/s (mult %dx, +/-)\nSim: %.2f ms", fps, simMultiplier, tps, simSteps, g.simStepMultiplier, simMS)
		ebitenutil.DebugPrint(screen, debugMsg)
	}
}

func (g *Game) Layout(_, _ int) (int, int) { return w, h }

func (g *Game) drawEarIndicators(screen *ebiten.Image, cx, cy int) {
	ox, oy := g.earOffsets()
	leftX := clampCoord(cx-ox, 0, w-1)
	leftY := clampCoord(cy-oy, 0, h-1)
	rightX := clampCoord(cx+ox, 0, w-1)
	rightY := clampCoord(cy+oy, 0, h-1)
	drawLine(screen, cx, cy, leftX, leftY, color.RGBA{0, 255, 200, 200})
	drawLine(screen, cx, cy, rightX, rightY, color.RGBA{0, 200, 255, 200})
	if leftX >= 0 && leftX < w && leftY >= 0 && leftY < h {
		screen.Set(leftX, leftY, color.RGBA{0, 255, 200, 255})
	}
	if rightX >= 0 && rightX < w && rightY >= 0 && rightY < h {
		screen.Set(rightX, rightY, color.RGBA{0, 200, 255, 255})
	}
}

func drawLine(screen *ebiten.Image, x0, y0, x1, y1 int, clr color.Color) {
	dx := int(math.Abs(float64(x1 - x0)))
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -int(math.Abs(float64(y1 - y0)))
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
			screen.Set(x0, y0, clr)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func startDefaultPGORecording(path string) (func(), error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return nil, err
	}
	var once sync.Once
	stop := func() {
		once.Do(func() {
			pprof.StopCPUProfile()
			_ = f.Close()
		})
	}
	return stop, nil
}

func main() {
	flag.Parse()
	workerCount := *threadCountFlag
	if workerCount <= 0 {
		workerCount = runtime.NumCPU()
	}
	if workerCount < 1 {
		workerCount = 1
	}
	runtime.GOMAXPROCS(workerCount)

	var stopProfile func()
	if *recordDefaultPGO {
		var err error
		stopProfile, err = startDefaultPGORecording("default.pgo")
		if err != nil {
			log.Fatalf("unable to start PGO recording: %v", err)
		}
		defer stopProfile()
	}

	g := newGame(workerCount, *useOpenCLFlag)
	if *recordDefaultPGO {
		g.enableAutoWalk(pgoRecordDuration)
		go func(stop func()) {
			timer := time.NewTimer(pgoRecordDuration)
			<-timer.C
			stop()
			log.Printf("default.pgo captured after %s; exiting", pgoRecordDuration)
			os.Exit(0)
		}(stopProfile)
	}

	ebiten.SetWindowSize(w*windowScale, h*windowScale)
	ebiten.SetWindowTitle("Acoustic Steps")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
