package main

import (
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Draw renders the current wave field, ear indicators, and optional overlays.
func (g *Game) Draw(screen *ebiten.Image) {
	if g.gpuSolver != nil {
		pixels := g.gpuSolver.PixelBytes()
		if len(pixels) == w*h*4 {
			if *occludeLineOfSightFlag && len(g.visibleStamp) == w*h {
				for i := range g.visibleStamp {
					if g.visibleStamp[i] == g.visibleGen {
						continue
					}
					base := i * 4
					pixels[base] = 0
					pixels[base+1] = 0
					pixels[base+2] = 0
					pixels[base+3] = 255
				}
			}
			if *showWallsFlag && len(g.walls) == w*h {
				for i, wall := range g.walls {
					if !wall {
						continue
					}
					base := i * 4
					pixels[base] = 30
					pixels[base+1] = 40
					pixels[base+2] = 80
					pixels[base+3] = 255
				}
			}
			screen.WritePixels(pixels)
		}
	}

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
		debugMsg := fmt.Sprintf("FPS: %.1f\nSim speed: %.2fx (%.1f TPS)\nSim steps: %.1f/s (mult %dx, +/-)\nSim: %.2f ms",
			fps, simMultiplier, tps, simSteps, g.simStepMultiplier, simMS)
		ebitenutil.DebugPrint(screen, debugMsg)
	}
}

// Layout reports the logical screen size used by Ebiten.
func (g *Game) Layout(_, _ int) (int, int) { return w, h }

// drawEarIndicators renders the listener's ear offset visualization.
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

// drawLine plots a line segment using Bresenham's integer algorithm.
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
