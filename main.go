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
	w, h                  = 256, 256
	damp                  = 0.9995
	speed                 = 0.5
	emitterRad            = 3
	moveSpeed             = 2
	stepDelay             = 15
	sampleRate            = 44100
	defaultTPS            = 60.0
	simStepsPerSecond     = defaultTPS * 100
	audioTicksPerSecond   = 960.0
	brownStep             = 0.02
	pinkSmoothing         = 0.05
	brightSmoothing       = 0.2
	ampSmoothing          = 0.15
	pressureMix           = 0.08
	gradientMix           = 0.04
	maxAudioLatencySec    = 0.2
	minAudioBufferChunk   = 4096
	minNoiseFloor         = 0.02
	minSamplesPerPush     = 128
	maxSamplesPerPush     = 2048
	audioChannels         = 2
	audioBytesPerSample   = 2
	audioFrameBytes       = audioChannels * audioBytesPerSample
	wallSegments          = 22
	wallMinLen            = 12
	wallMaxLen            = 42
	wallExclusionRadius   = 12
	wallThicknessVariance = 2
)

var amplitudeOnlyFlag = flag.Bool("amplitude-only", true, "output direct wave amplitude instead of noise texture")

var maxAudioSamples = int(float64(sampleRate) * maxAudioLatencySec)

func init() {
	if maxAudioSamples < minAudioBufferChunk {
		maxAudioSamples = minAudioBufferChunk
	}
}

type Game struct {
	curr, prev, next   []float32
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
						g.curr[cy*w+cx] = 1.0
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
}

func (g *Game) trySetWall(x, y int) {
	if x <= 1 || x >= w-1 || y <= 1 || y >= h-1 {
		return
	}
	dist := math.Hypot(float64(x)-g.ex, float64(y)-g.ey)
	if dist < float64(wallExclusionRadius) {
		return
	}
	idx := y*w + x
	g.walls[idx] = true
	g.curr[idx] = 0
	g.prev[idx] = 0
	g.next[idx] = 0
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
	hasWalls := len(g.walls) > 0
	if hasWalls {
		for idx, wall := range g.walls {
			if wall {
				g.curr[idx] = 0
				g.prev[idx] = 0
			}
		}
	}
	numCPU := runtime.NumCPU()
	rowsPer := (h + numCPU - 1) / numCPU
	var wg sync.WaitGroup
	for i := 0; i < numCPU; i++ {
		yStart := i * rowsPer
		if yStart >= h {
			break
		}
		yEnd := yStart + rowsPer
		if yEnd > h {
			yEnd = h
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				if y == 0 || y == h-1 {
					continue
				}
				for x := 1; x < w-1; x++ {
					i := y*w + x
					if hasWalls && g.walls[i] {
						g.next[i] = 0
						continue
					}
					lap := g.curr[i-1] + g.curr[i+1] + g.curr[i-w] + g.curr[i+w] - 4*g.curr[i]
					g.next[i] = (2*g.curr[i] - g.prev[i]) + float32(speed)*lap
					g.next[i] *= damp
				}
			}
		}(yStart, yEnd)
	}
	wg.Wait()
	g.prev, g.curr, g.next = g.curr, g.next, g.prev

	if cx >= 1 && cx < w-1 && cy >= 1 && cy < h-1 {
		pressure, energy, gradient := g.samplePressureEnergy(cx, cy)
		g.pushAudioSample(pressure, energy, gradient, stepDuration)
	} else {
		g.pushAudioSample(0, 0, 0, stepDuration)
	}
}

func (g *Game) samplePressureEnergy(cx, cy int) (float32, float32, float32) {
	var sum float32
	var energy float64
	var gx, gy float64
	count := 0
	for oy := -1; oy <= 1; oy++ {
		y := cy + oy
		if y < 1 || y >= h-1 {
			continue
		}
		for ox := -1; ox <= 1; ox++ {
			x := cx + ox
			if x < 1 || x >= w-1 {
				continue
			}
			if g.isWall(x, y) {
				continue
			}
			v := g.curr[y*w+x]
			sum += v
			energy += math.Abs(float64(v))
			count++
			if x > 0 && x < w-1 {
				if !g.isWall(x+1, y) && !g.isWall(x-1, y) {
					gx += float64(g.curr[y*w+x+1] - g.curr[y*w+x-1])
				}
			}
			if y > 0 && y < h-1 {
				if !g.isWall(x, y+1) && !g.isWall(x, y-1) {
					gy += float64(g.curr[(y+1)*w+x] - g.curr[(y-1)*w+x])
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
	lowBase := g.audioAmp * 0.55
	midBase := g.audioAmp * 0.35
	highBase := g.audioAmp*0.1 + gradientAmt*0.25
	if g.amplitudeOnly {
		gain := g.audioAmp*4 + 0.05
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
		v := g.curr[i]
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

	g := &Game{
		curr:          make([]float32, w*h),
		prev:          make([]float32, w*h),
		next:          make([]float32, w*h),
		ex:            float64(w / 2),
		ey:            float64(h / 2),
		audioStream:   stream,
		noiseRand:     rand.New(rand.NewSource(time.Now().UnixNano())),
		levelRand:     rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:         make([]bool, w*h),
		amplitudeOnly: *amplitudeOnlyFlag,
	}
	g.generateWalls()

	ebiten.SetWindowSize(w*2, h*2)
	ebiten.SetWindowTitle("Acoustic Steps with Live Sound")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
