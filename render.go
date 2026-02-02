package main

import (
	"fmt"
	"image/color"
	"math"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Draw renders the current wave field, ear indicators, and optional overlays.
func (g *Game) Draw(screen *ebiten.Image) {
	if g.gpuSolver != nil {
		pixels := g.gpuSolver.PixelBytes()
		if len(pixels) == w*h*4 {
			screen.WritePixels(pixels)
		}
	}

	baseX := int(g.emitterX)
	baseY := int(g.emitterY)
	for _, offset := range emitterFootprint {
		cx := baseX + offset.dx
		cy := baseY + offset.dy
		if cx >= 0 && cx < w && cy >= 0 && cy < h {
			screen.Set(cx, cy, color.RGBA{255, 0, 0, 255})
		}
	}
	g.drawEarIndicators(screen, int(g.listenerX), int(g.listenerY))
	g.drawAudioSampleMarker(screen)
	g.drawScaleBar(screen)

	if *debugFlag {
		now := time.Now()
		if g.debugOverlayMessage == "" || !now.Before(g.nextDebugOverlayUpdate) {
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
			worldWidthFeet := float64(w) * cellSizeMeters() * 3.28084
			g.debugOverlayMessage = fmt.Sprintf("FPS: %.1f\nSim speed: %.2fx (%.1f TPS)\nSim steps: %.1f/s (mult %dx, +/-)\nWorld: %.1f ft\nControl: %s (Tab)\nSim: %.2f ms",
				fps, simMultiplier, tps, simSteps, g.simStepMultiplier, worldWidthFeet, g.controlModeLabel(), simMS)
			g.nextDebugOverlayUpdate = now.Add(time.Second)
		}
		ebitenutil.DebugPrint(screen, g.debugOverlayMessage)
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

func (g *Game) drawAudioSampleMarker(screen *ebiten.Image) {
	centerX := clampCoord(int(g.listenerX), 0, w-1)
	centerY := clampCoord(int(g.listenerY), 0, h-1)
	dotColor := color.RGBA{255, 40, 40, 255}
	for dy := -1; dy <= 1; dy++ {
		y := centerY + dy
		if y < 0 || y >= h {
			continue
		}
		for dx := -1; dx <= 1; dx++ {
			x := centerX + dx
			if x < 0 || x >= w {
				continue
			}
			screen.Set(x, y, dotColor)
		}
	}
}

func (g *Game) drawScaleBar(screen *ebiten.Image) {
	dxM := cellSizeMeters()
	if dxM <= 0 {
		return
	}
	const metersPerFoot = 0.3048
	feetPerCell := dxM / metersPerFoot
	if feetPerCell <= 0 {
		return
	}

	// Choose a "nice" bar length that fits comfortably on screen.
	maxBarPixels := float64(w) * 0.18
	if maxBarPixels > 140 {
		maxBarPixels = 140
	}
	if maxBarPixels < 60 {
		maxBarPixels = 60
	}
	candidatesFeet := []float64{1, 2, 5, 10, 20, 50}
	chosenFeet := candidatesFeet[0]
	chosenPixels := chosenFeet / feetPerCell
	for _, ft := range candidatesFeet {
		pixels := ft / feetPerCell
		if pixels <= maxBarPixels && pixels >= 35 {
			chosenFeet = ft
			chosenPixels = pixels
		}
	}
	barLen := int(math.Round(chosenPixels))
	if barLen < 10 {
		return
	}

	label := fmt.Sprintf("%.0f ft", chosenFeet)
	labelWidth := 7 * len(label)

	margin := 6
	x1 := w - margin
	x0 := x1 - barLen
	y := h - margin - 2
	labelX := x1 - labelWidth
	labelY := y - 14

	if x0 < margin {
		x0 = margin
	}
	if labelX < margin {
		labelX = margin
	}
	if labelY < margin {
		labelY = margin
	}

	lineColor := color.RGBA{240, 240, 240, 255}
	shadowColor := color.RGBA{0, 0, 0, 180}
	drawLine(screen, x0+1, y+1, x1+1, y+1, shadowColor)
	drawLine(screen, x0, y, x1, y, lineColor)
	drawLine(screen, x0, y-3, x0, y+3, lineColor)
	drawLine(screen, x1, y-3, x1, y+3, lineColor)
	ebitenutil.DebugPrintAt(screen, label, labelX, labelY)
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
