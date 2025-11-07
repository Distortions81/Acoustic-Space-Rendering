package main

import (
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
	w, h                = 256, 256
	damp                = 0.996
	speed               = 0.5
	emitterRad          = 3
	moveSpeed           = 2
	stepDelay           = 15
	sampleRate          = 44100
	defaultTPS          = 60.0
	simStepsPerSecond   = defaultTPS * 8
	brownStep           = 0.02
	ampSmoothing        = 0.15
	maxAudioLatencySec  = 0.5
	minAudioBufferChunk = 1024
	minNoiseFloor       = 0.02
)

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
	audioAmp           float32
	physicsAccumulator float64
	noiseBuf           []float32
	lastAudioTime      time.Time
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
	g.ex = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.ex+dx))
	g.ey = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.ey+dy))

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

func (g *Game) stepWave(cx, cy int, stepDuration float64) {
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
		pressure, energy := g.samplePressureEnergy(cx, cy)
		g.pushAudioSample(pressure, energy, stepDuration)
	} else {
		g.pushAudioSample(0, 0, stepDuration)
	}
}

func (g *Game) samplePressureEnergy(cx, cy int) (float32, float32) {
	var sum float32
	var energy float64
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
			v := g.curr[y*w+x]
			sum += v
			energy += math.Abs(float64(v))
			count++
		}
	}
	if count == 0 {
		return 0, 0
	}
	return sum / float32(count), float32(energy / float64(count))
}

func (g *Game) pushAudioSample(pressure, energy float32, stepDuration float64) {
	g.sampleAccumulator += stepDuration * sampleRate
	samples := int(g.sampleAccumulator)
	if samples <= 0 {
		return
	}
	// Scale brown noise amplitude by local wave magnitude and smooth changes.
	targetAmp := float32(math.Min(1, math.Max(float64(energy)*4, float64(minNoiseFloor))))
	g.audioAmp += (targetAmp - g.audioAmp) * ampSmoothing
	noise := g.ensureNoiseBuffer(samples)
	for i := 0; i < samples; i++ {
		g.brownState += (g.noiseRand.Float32()*2 - 1) * brownStep
		if g.brownState > 1 {
			g.brownState = 1
		} else if g.brownState < -1 {
			g.brownState = -1
		}
		sample := g.brownState*g.audioAmp + pressure*0.05
		if sample > 1 {
			sample = 1
		} else if sample < -1 {
			sample = -1
		}
		noise[i] = sample
	}
	g.audioStream.PushSamples(noise)
	g.sampleAccumulator -= float64(samples)
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
	lastSample float32
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
	s.mutex.Unlock()
}

func (s *WaveStream) Read(p []byte) (int, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	n := len(p) / 4 // stereo 16-bit
	if n == 0 {
		return 0, nil
	}
	readable := len(s.buf)
	for i := 0; i < n; i++ {
		var sampleValue float32
		if i < readable {
			sampleValue = s.buf[i]
		} else {
			sampleValue = s.lastSample
		}
		s.lastSample = sampleValue
		sample := int16(sampleValue * 20000)
		p[4*i] = byte(sample)
		p[4*i+1] = byte(sample >> 8)
		p[4*i+2] = p[4*i]
		p[4*i+3] = p[4*i+1]
	}
	consumed := n
	if consumed > readable {
		consumed = readable
	}
	if consumed < len(s.buf) {
		s.buf = s.buf[consumed:]
	} else {
		s.buf = s.buf[:0]
	}
	return len(p), nil
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
	runtime.GOMAXPROCS(runtime.NumCPU())
	audioCtx := audio.NewContext(sampleRate)
	stream := &WaveStream{}
	player, _ := audioCtx.NewPlayer(stream)
	player.Play()

	g := &Game{
		curr:        make([]float32, w*h),
		prev:        make([]float32, w*h),
		next:        make([]float32, w*h),
		ex:          float64(w / 2),
		ey:          float64(h / 2),
		audioStream: stream,
		noiseRand:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	ebiten.SetWindowSize(w*2, h*2)
	ebiten.SetWindowTitle("Acoustic Steps with Live Sound")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
