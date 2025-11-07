package main

import (
	"image/color"
	"math"
	"runtime"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	w, h       = 256, 256
	damp       = 0.996
	speed      = 0.3
	emitterRad = 3
	moveSpeed  = 2
)

type Game struct {
	curr, prev, next []float32
	ex, ey           float64
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

	if dx != 0 || dy != 0 {
		for y := -emitterRad; y <= emitterRad; y++ {
			for x := -emitterRad; x <= emitterRad; x++ {
				if x*x+y*y <= emitterRad*emitterRad {
					cx := int(g.ex) + x
					cy := int(g.ey) + y
					g.curr[cy*w+cx] = 1.0
				}
			}
		}
	}

	numCPU := runtime.NumCPU()
	var wg sync.WaitGroup
	rowsPer := h / numCPU

	for i := 0; i < numCPU; i++ {
		yStart := i * rowsPer
		yEnd := yStart + rowsPer
		if i == numCPU-1 {
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
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	img := make([]byte, w*h*4)
	for i := 0; i < w*h; i++ {
		v := g.curr[i]
		v = float32(math.Max(-1, math.Min(1, float64(v))))
		intensity := byte(math.Abs(float64(v)) * 255)
		if v > 0 {
			img[i*4] = intensity // white on black
			img[i*4+1] = intensity
			img[i*4+2] = intensity
		} else {
			img[i*4] = 0
			img[i*4+1] = 0
			img[i*4+2] = 0
		}
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

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	ebiten.SetWindowSize(w*2, h*2)
	ebiten.SetWindowTitle("Acoustic Footsteps - Optimized BW")
	g := &Game{
		curr: make([]float32, w*h),
		prev: make([]float32, w*h),
		next: make([]float32, w*h),
		ex:   float64(w / 2),
		ey:   float64(h / 2),
	}
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
