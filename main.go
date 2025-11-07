package main

import (
	"flag"
	"image/color"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
)

const (
	w, h                   = 256, 256
	damp                   = 0.9998
	speed                  = 0.5
	waveDamp32             = float32(damp)
	waveSpeed32            = float32(speed)
	emitterRad             = 3
	moveSpeed              = 2
	stepDelay              = 15
	sampleRate             = 44100
	defaultTPS             = 60.0
	simStepsPerSecond      = defaultTPS * 100
	audioTicksPerSecond    = simStepsPerSecond
	controlDownsampleSteps = 6
	earOffsetCells         = 5
	brownStep              = 0.02
	pinkSmoothing          = 0.05
	brightSmoothing        = 0.2
	ampSmoothing           = 0.15
	pressureMix            = 0.08
	gradientMix            = 0.04
	maxAudioLatencySec     = 0.2
	minAudioBufferChunk    = 4096
	minNoiseFloor          = 0.02
	minSamplesPerPush      = 128
	maxSamplesPerPush      = 2048
	audioChannels          = 2
	audioBytesPerSample    = 2
	audioFrameBytes        = audioChannels * audioBytesPerSample
	wallSegments           = 22
	wallMinLen             = 12
	wallMaxLen             = 42
	wallExclusionRadius    = 12
	wallThicknessVariance  = 2
	lowBandSmoothing       = 0.01
	midBandSmoothing       = 0.04
	highBandSmoothing      = 0.12
	bandResponseSmoothing  = 0.2
	bandWeightFloor        = 0.0001
)

var amplitudeOnlyFlag = flag.Bool("am", true, "output direct wave amplitude instead of noise texture")

var maxAudioSamples = int(float64(sampleRate) * maxAudioLatencySec)

func init() {
	if maxAudioSamples < minAudioBufferChunk {
		maxAudioSamples = minAudioBufferChunk
		listenerForwardX: 0,
		listenerForwardY: -1,
	}
}

type half uint16

func float32ToHalf(v float32) half {
	bits := math.Float32bits(v)
	sign := (bits >> 16) & 0x8000
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	switch {
	case exp <= 0:
		if exp < -10 {
			return half(sign)
		}
		mant |= 0x800000
		shift := uint32(1 - exp)
		mant16 := mant >> (shift + 13)
		if (mant>>(shift+12))&1 == 1 {
			mant16++
		}
		return half(sign | mant16)
	case exp >= 0x1F:
		return half(sign | 0x7C00)
	default:
		mant16 := mant >> 13
		if mant&0x1FFF > 0x1000 || (mant&0x1FFF == 0x1000 && mant16&1 == 1) {
			mant16++
			if mant16 == 0x0400 {
				mant16 = 0
				exp++
				if exp == 0x1F {
					return half(sign | 0x7C00)
				}
			}
		}
		return half(sign | uint32(exp)<<10 | mant16)
	}
}

func halfToFloat32(h half) float32 {
	bits := uint16(h)
	sign := uint32(bits&0x8000) << 16
	exp := (bits >> 10) & 0x1F
	mant := uint32(bits & 0x3FF)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3FF
		fbits := sign | ((uint32(exp - 15 + 127)) << 23) | (mant << 13)
		return math.Float32frombits(fbits)
	case 0x1F:
		fbits := sign | 0x7F800000 | (mant << 13)
		return math.Float32frombits(fbits)
	default:
		fbits := sign | ((uint32(exp - 15 + 127)) << 23) | (mant << 13)
		return math.Float32frombits(fbits)
	}
}

type wavePlane [][]half

type waveField struct {
	width, height int
	curr          wavePlane
	prev          wavePlane
	next          wavePlane
}

func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr: makePlane(width, height),
		prev: makePlane(width, height),
		next: makePlane(width, height),
	}
}

func makePlane(width, height int) wavePlane {
	p := make(wavePlane, height)
	for y := range p {
		p[y] = make([]half, width)
	}
	return p
}

func (f *waveField) setCurr(x, y int, value float32) {
	f.curr[y][x] = float32ToHalf(value)
}

func (f *waveField) zeroCell(x, y int) {
	f.curr[y][x] = 0
	f.prev[y][x] = 0
	f.next[y][x] = 0
}

func (f *waveField) readCurr(x, y int) float32 {
	return halfToFloat32(f.curr[y][x])
}

func (f *waveField) swap() {
	f.prev, f.curr, f.next = f.curr, f.next, f.prev
}

func (f *waveField) zeroBoundaries() {
	lastRow := f.height - 1
	lastCol := f.width - 1
	for x := 0; x < f.width; x++ {
		f.next[0][x] = 0
		f.next[lastRow][x] = 0
	}
	for y := 1; y < lastRow; y++ {
		f.next[y][0] = 0
		f.next[y][lastCol] = 0
	}
}

type rowMask struct {
	y  int
	xs []int
}

type workerMask struct {
	rows []rowMask
}

type workerJob struct {
	field *waveField
	mask  *workerMask
	wg    *sync.WaitGroup
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

func runWaveWorker(jobs <-chan workerJob, width int) {
	cache := newRowCache(width)
	for job := range jobs {
		if job.mask == nil || len(job.mask.rows) == 0 {
			if job.wg != nil {
				job.wg.Done()
			}
			continue
		}
		processMask(job.field, job.mask, cache)
		if job.wg != nil {
			job.wg.Done()
		}
	}
}

func processMask(field *waveField, mask *workerMask, cache *rowCache) {
	for _, row := range mask.rows {
		y := row.y
		convertRow(field.curr[y], cache.center)
		convertRow(field.prev[y], cache.prev)
		convertRow(field.curr[y-1], cache.top)
		convertRow(field.curr[y+1], cache.bottom)

		nextRow := field.next[y]
		nextRow[0] = 0
		nextRow[len(nextRow)-1] = 0
		for _, x := range row.xs {
			lap := cache.center[x-1] + cache.center[x+1] + cache.top[x] + cache.bottom[x] - 4*cache.center[x]
			val := (2*cache.center[x] - cache.prev[x]) + waveSpeed32*lap
			val *= waveDamp32
			nextRow[x] = float32ToHalf(val)
		}
	}
}

func convertRow(src []half, dst []float32) {
	for i := range src {
		dst[i] = halfToFloat32(src[i])
	}
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

type Game struct {
	field              *waveField
	ex, ey             float64
	stepTimer          int
	audioStream        *WaveStream
	sampleAccumulator  float64
	noiseRand          *rand.Rand
	brownState         float32
	pinkState          float32
	brightState        float32
	audioAmp           float32
	physicsAccumulator float64
	noiseBuf           []float32
	lastAudioTime      time.Time
	walls              []bool
	levelRand          *rand.Rand
	amplitudeOnly      bool
	workerJobs         chan workerJob
	workerCount        int
	workerMasks        []workerMask
	maskDirty          bool
	controlPressureSum float64
	controlEnergySum   float64
	controlGradientSum float64
	controlDurationSum float64
	controlSamples     int
	lowEnergyEnv       float32
	midEnergyEnv       float32
	highEnergyEnv      float32
	bandLowEnergy      float32
	bandMidEnergy      float32
	bandHighEnergy     float32
	listenerForwardX   float64
	listenerForwardY   float64
	leftEarPosX        int
	leftEarPosY        int
	rightEarPosX       int
	rightEarPosY       int
	controlLeftPressureSum   float64
	controlLeftEnergySum     float64
	controlLeftGradientSum   float64
	controlRightPressureSum  float64
	controlRightEnergySum    float64
	controlRightGradientSum  float64
}

func newGame(stream *WaveStream, amplitudeOnly bool) *Game {
	workerCount := runtime.NumCPU()
	if workerCount < 1 {
		workerCount = 1
	}
	g := &Game{
		field:             newWaveField(w, h),
		ex:                float64(w / 2),
		ey:                float64(h / 2),
		audioStream:       stream,
		noiseRand:         rand.New(rand.NewSource(time.Now().UnixNano())),
		levelRand:         rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:             make([]bool, w*h),
		amplitudeOnly:     amplitudeOnly,
		workerCount:       workerCount,
		workerJobs:        make(chan workerJob, workerCount),
		maskDirty:         true,
		lastAudioTime:     time.Time{},
		noiseBuf:          nil,
		sampleAccumulator: 0,
	}
	for i := 0; i < workerCount; i++ {
		go runWaveWorker(g.workerJobs, g.field.width)
	}
	g.generateWalls()
	g.rebuildInteriorMask()
	return g
}

func (g *Game) rebuildInteriorMask() {
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	rows := make([]rowMask, 0, h-2)
	for y := 1; y < h-1; y++ {
		base := y * w
		xs := make([]int, 0, w-2)
		for x := 1; x < w-1; x++ {
			if g.walls[base+x] {
				continue
			}
			xs = append(xs, x)
		}
		if len(xs) == 0 {
			continue
		}
		rows = append(rows, rowMask{y: y, xs: xs})
	}
	g.workerMasks = assignRowMasks(g.workerCount, rows)
	g.maskDirty = false
}

func (g *Game) ensureInteriorMask() {
	if g.maskDirty || len(g.workerMasks) != g.workerCount {
		g.rebuildInteriorMask()
	}
}

func (g *Game) Update() error {
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
	oldX, oldY := g.ex, g.ey
	g.ex = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.ex+dx))
	g.ey = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.ey+dy))
	if g.isWall(int(g.ex), int(g.ey)) {
		g.ex, g.ey = oldX, oldY
	}

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
						g.field.setCurr(cx, cy, 1.0)
					}
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

	cx, cy := int(g.ex), int(g.ey)
	now := time.Now()
	if g.lastAudioTime.IsZero() {
		g.lastAudioTime = now
	}
	frameElapsed := now.Sub(g.lastAudioTime)
	frameSeconds := frameElapsed.Seconds()
	if frameSeconds <= 0 {
		frameSeconds = 1.0 / simStepsPerSecond
	}
	g.lastAudioTime = now
	actualTPS := ebiten.ActualTPS()
	if actualTPS < 1 {
		actualTPS = defaultTPS
	}
	g.physicsAccumulator += simStepsPerSecond / actualTPS
	steps := int(g.physicsAccumulator)
	if steps < 1 {
		steps = 1
	}
	stepDuration := frameSeconds / float64(steps)
	if stepDuration <= 0 {
		stepDuration = 1.0 / simStepsPerSecond
	}
	for i := 0; i < steps; i++ {
		g.stepWave(cx, cy, stepDuration)
	}
	g.flushControlAccumulator(true)
	g.physicsAccumulator -= float64(steps)

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

func (g *Game) stepWave(cx, cy int, stepDuration float64) {
	g.ensureInteriorMask()
	var wg sync.WaitGroup
	for i := range g.workerMasks {
		mask := &g.workerMasks[i]
		if len(mask.rows) == 0 {
			continue
		}
		wg.Add(1)
		g.workerJobs <- workerJob{
			field: g.field,
			mask:  mask,
			wg:    &wg,
		}
	}
	wg.Wait()
	g.field.zeroBoundaries()
	g.field.swap()

	if cx >= 1 && cx < w-1 && cy >= 1 && cy < h-1 {
		pressure, energy, gradient := g.samplePressureEnergy(cx, cy)
		g.updateBandEnergies(energy)
		g.accumulateControlSample(pressure, energy, gradient, stepDuration)
	} else {
		g.updateBandEnergies(0)
		g.accumulateControlSample(0, 0, 0, stepDuration)
	}
}

func (g *Game) samplePressureEnergy(cx, cy int) (float32, float32, float32) {
	var sum float32
	var energy float64
	var gx, gy float64
	count := 0
	for oy := -2; oy <= 2; oy++ {
		y := cy + oy
		if y < 1 || y >= h-1 {
			continue
		}
		for ox := -2; ox <= 2; ox++ {
			x := cx + ox
			if x < 1 || x >= w-1 {
				continue
			}
			if g.isWall(x, y) {
				continue
			}
			v := g.field.readCurr(x, y)
			sum += v
			energy += math.Abs(float64(v))
			count++
			if x > 0 && x < w-1 {
				if !g.isWall(x+1, y) && !g.isWall(x-1, y) {
					gx += float64(g.field.readCurr(x+1, y) - g.field.readCurr(x-1, y))
				}
			}
			if y > 0 && y < h-1 {
				if !g.isWall(x, y+1) && !g.isWall(x, y-1) {
					gy += float64(g.field.readCurr(x, y+1) - g.field.readCurr(x, y-1))
				}
			}
		}
	}
	if count == 0 {
		return 0, 0, 0
	}
	gradMag := float32(math.Sqrt(gx*gx+gy*gy) / float64(count))
	return sum / float32(count), float32(energy / float64(count)), gradMag
}

func (g *Game) updateBandEnergies(localEnergy float32) {
	if localEnergy < 0 {
		localEnergy = -localEnergy
	}
	g.lowEnergyEnv += (localEnergy - g.lowEnergyEnv) * lowBandSmoothing
	g.midEnergyEnv += (localEnergy - g.midEnergyEnv) * midBandSmoothing
	g.highEnergyEnv += (localEnergy - g.highEnergyEnv) * highBandSmoothing

	lowBand := g.lowEnergyEnv
	midBand := g.midEnergyEnv - g.lowEnergyEnv
	highBand := g.highEnergyEnv - g.midEnergyEnv
	if midBand < 0 {
		midBand = 0
	}
	if highBand < 0 {
		highBand = 0
	}

	g.bandLowEnergy += (lowBand - g.bandLowEnergy) * bandResponseSmoothing
	g.bandMidEnergy += (midBand - g.bandMidEnergy) * bandResponseSmoothing
	g.bandHighEnergy += (highBand - g.bandHighEnergy) * bandResponseSmoothing
}

func (g *Game) accumulateControlSample(pressure, energy, gradient float32, stepDuration float64) {
	g.controlPressureSum += float64(pressure)
	g.controlEnergySum += float64(energy)
	g.controlGradientSum += float64(gradient)
	g.controlDurationSum += stepDuration
	g.controlSamples++
	g.flushControlAccumulator(false)
}

func (g *Game) flushControlAccumulator(force bool) {
	if g.controlSamples == 0 {
		return
	}
	if !force && g.controlSamples < controlDownsampleSteps {
		return
	}
	scale := 1.0 / float64(g.controlSamples)
	avgPressure := float32(g.controlPressureSum * scale)
	avgEnergy := float32(g.controlEnergySum * scale)
	avgGradient := float32(g.controlGradientSum * scale)
	duration := g.controlDurationSum

	g.controlPressureSum = 0
	g.controlEnergySum = 0
	g.controlGradientSum = 0
	g.controlDurationSum = 0
	g.controlSamples = 0

	if duration <= 0 {
		duration = float64(controlDownsampleSteps) / simStepsPerSecond
	}
	g.pushAudioSample(avgPressure, avgEnergy, avgGradient, duration)
}

func (g *Game) bandMixWeights() (float32, float32, float32) {
	total := g.bandLowEnergy + g.bandMidEnergy + g.bandHighEnergy
	if total < bandWeightFloor {
		return 0.55, 0.3, 0.15
	}
	invTotal := float32(1.0 / total)
	return g.bandLowEnergy * invTotal, g.bandMidEnergy * invTotal, g.bandHighEnergy * invTotal
}

func (g *Game) pushAudioSample(pressure, energy, gradient float32, stepDuration float64) {
	g.sampleAccumulator += stepDuration * sampleRate
	for {
		samples := int(g.sampleAccumulator)
		if samples < minSamplesPerPush {
			break
		}
		if samples > maxSamplesPerPush {
			samples = maxSamplesPerPush
		}
		g.produceAudioChunk(samples, pressure, energy, gradient)
		g.sampleAccumulator -= float64(samples)
	}
}

func (g *Game) produceAudioChunk(samples int, pressure, energy, gradient float32) {
	// Scale noise amplitude by local energy plus gradient cue and smooth changes.
	targetAmp := float32(math.Min(1, math.Max(float64(energy)*3+float64(gradient)*2, float64(minNoiseFloor))))
	g.audioAmp += (targetAmp - g.audioAmp) * ampSmoothing
	noise := g.ensureNoiseBuffer(samples)
	gradientAmt := float32(math.Min(1, float64(gradient)*4))
	lowWeight, midWeight, highWeight := g.bandMixWeights()
	lowShape := 0.55 * (0.6 + 0.8*lowWeight)
	midShape := 0.35 * (0.6 + 0.8*midWeight)
	highShape := 0.1 * (0.7 + 1.3*highWeight)
	shapeSum := lowShape + midShape + highShape
	var lowBase, midBase, highBase float32
	if shapeSum > 0 {
		scale := g.audioAmp / float32(shapeSum)
		lowBase = float32(lowShape) * scale
		midBase = float32(midShape) * scale
		highBase = float32(highShape)*scale + gradientAmt*0.25
	} else {
		lowBase = g.audioAmp * 0.55
		midBase = g.audioAmp * 0.35
		highBase = g.audioAmp*0.1 + gradientAmt*0.25
	}
	if g.amplitudeOnly {
		totalBandEnergy := g.bandLowEnergy + g.bandMidEnergy + g.bandHighEnergy
		gain := g.audioAmp*4 + 0.05 + totalBandEnergy*0.5
		baseSample := pressure*gain + gradientAmt*gradientMix
		if baseSample > 1 {
			baseSample = 1
		} else if baseSample < -1 {
			baseSample = -1
		}
		for i := 0; i < samples; i++ {
			noise[i] = baseSample
		}
		g.audioStream.PushSamples(noise)
		return
	}

	for i := 0; i < samples; i++ {
		white := g.noiseRand.Float32()*2 - 1
		g.brownState += white * brownStep
		if g.brownState > 1 {
			g.brownState = 1
		} else if g.brownState < -1 {
			g.brownState = -1
		}
		g.pinkState += (white - g.pinkState) * pinkSmoothing
		g.brightState += (white - g.brightState) * brightSmoothing
		low := g.brownState
		mid := g.pinkState
		high := white - g.brightState
		sample := low*lowBase + mid*midBase + high*highBase
		sample += pressure * pressureMix
		sample += gradientAmt * gradientMix
		if sample > 1 {
			sample = 1
		} else if sample < -1 {
			sample = -1
		}
		noise[i] = sample
	}
	g.audioStream.PushSamples(noise)
}

func (g *Game) ensureNoiseBuffer(n int) []float32 {
	if cap(g.noiseBuf) < n {
		g.noiseBuf = make([]float32, n)
	}
	g.noiseBuf = g.noiseBuf[:n]
	return g.noiseBuf
}

func (g *Game) Draw(screen *ebiten.Image) {
	img := make([]byte, w*h*4)
	for i := 0; i < w*h; i++ {
		if len(g.walls) > 0 && g.walls[i] {
			img[i*4] = 30
			img[i*4+1] = 40
			img[i*4+2] = 80
			img[i*4+3] = 255
			continue
		}
		x := i % w
		y := i / w
		v := g.field.readCurr(x, y)
		v = float32(math.Max(-1, math.Min(1, float64(v))))
		intensity := byte(math.Abs(float64(v)) * 255)
		img[i*4] = intensity
		img[i*4+1] = intensity
		img[i*4+2] = intensity
		img[i*4+3] = 255
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
}

func (g *Game) Layout(_, _ int) (int, int) { return w, h }

// WaveStream implements io.Read for Ebiten's audio player
type WaveStream struct {
	buf        []float32
	mutex      sync.Mutex
	cond       *sync.Cond
	lastSample float32
}

func NewWaveStream() *WaveStream {
	ws := &WaveStream{}
	ws.cond = sync.NewCond(&ws.mutex)
	return ws
}

func (s *WaveStream) PushSamples(samples []float32) {
	if len(samples) == 0 {
		return
	}
	s.mutex.Lock()
	s.buf = append(s.buf, samples...)
	if len(s.buf) > maxAudioSamples {
		s.buf = s.buf[len(s.buf)-maxAudioSamples:] // keep most recent window
	}
	s.lastSample = s.buf[len(s.buf)-1]
	s.cond.Broadcast()
	s.mutex.Unlock()
}

func (s *WaveStream) Read(p []byte) (int, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	frameCount := len(p) / audioFrameBytes
	if frameCount == 0 {
		return 0, nil
	}
	for len(s.buf) == 0 {
		s.cond.Wait()
	}
	if frameCount > len(s.buf) {
		frameCount = len(s.buf)
	}
	for i := 0; i < frameCount; i++ {
		sampleValue := s.buf[i]
		s.lastSample = sampleValue
		sample := int16(sampleValue * 20000)
		for ch := 0; ch < audioChannels; ch++ {
			base := i*audioFrameBytes + ch*audioBytesPerSample
			p[base] = byte(sample)
			p[base+1] = byte(sample >> 8)
		}
	}
	if frameCount < len(s.buf) {
		s.buf = s.buf[frameCount:]
	} else {
		s.buf = s.buf[:0]
	}
	return frameCount * audioFrameBytes, nil
}

func (s *WaveStream) Close() error { return nil }

func (s *WaveStream) Seek(offset int64, whence int) (int64, error) {
	// Ebiten's audio player probes the stream with Seek(0, io.SeekStart) and similar no-op requests.
	if offset == 0 {
		switch whence {
		case io.SeekStart, io.SeekCurrent, io.SeekEnd:
			return 0, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())
	audioCtx := audio.NewContext(sampleRate)
	stream := NewWaveStream()
	player, _ := audioCtx.NewPlayer(stream)
	player.Play()

	g := newGame(stream, *amplitudeOnlyFlag)

	ebiten.SetWindowSize(w*2, h*2)
	ebiten.SetWindowTitle("Acoustic Steps with Live Sound")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
