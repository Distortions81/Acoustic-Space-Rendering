package main

import (
	"fmt"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Draw renders the current wave field, ear indicators, and optional overlays.
func (g *Game) Draw(screen *ebiten.Image) {
	if g.gpuSolver != nil {
		pixels := g.gpuSolver.PixelBytes()
		if view := g.viewportPixels(pixels); len(view) == viewportWidth*viewportHeight*4 {
			screen.WritePixels(view)
		}
	}

	baseX := viewportWidth / 2
	baseY := viewportHeight / 2
	for _, offset := range emitterFootprint {
		cx := baseX + offset.dx
		cy := baseY + offset.dy
		if cx >= 0 && cx < viewportWidth && cy >= 0 && cy < viewportHeight {
			screen.Set(cx, cy, color.RGBA{255, 0, 0, 255})
		}
	}

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
func (g *Game) Layout(_, _ int) (int, int) { return viewportWidth, viewportHeight }

func (g *Game) viewportPixels(world []byte) []byte {
	if viewportWidth <= 0 || viewportHeight <= 0 {
		return nil
	}
	viewSize := viewportWidth * viewportHeight * 4
	if len(g.viewPixels) != viewSize {
		g.viewPixels = make([]byte, viewSize)
	}
	if len(world) != w*h*4 || viewSize == 0 {
		for i := range g.viewPixels {
			g.viewPixels[i] = 0
		}
		return g.viewPixels
	}
	rowWidth := viewportWidth * 4
	worldRowSize := w * 4
	for row := 0; row < viewportHeight; row++ {
		dstStart := row * rowWidth
		srcY := g.viewY + row
		if srcY < 0 || srcY >= h {
			for i := 0; i < rowWidth; i++ {
				g.viewPixels[dstStart+i] = 0
			}
			continue
		}
		srcStart := srcY * worldRowSize
		for col := 0; col < viewportWidth; col++ {
			dstIdx := dstStart + col*4
			srcX := g.viewX + col
			if srcX < 0 || srcX >= w {
				g.viewPixels[dstIdx] = 0
				g.viewPixels[dstIdx+1] = 0
				g.viewPixels[dstIdx+2] = 0
				g.viewPixels[dstIdx+3] = 0
				continue
			}
			srcIdx := srcStart + srcX*4
			copy(g.viewPixels[dstIdx:dstIdx+4], world[srcIdx:srcIdx+4])
		}
	}
	return g.viewPixels
}
