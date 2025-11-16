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
		if len(pixels) == w*h*4 {
			screen.WritePixels(pixels)
		}
	}

	baseX := int(g.ex)
	baseY := int(g.ey)
	for _, offset := range emitterFootprint {
		cx := baseX + offset.dx
		cy := baseY + offset.dy
		if cx >= 0 && cx < w && cy >= 0 && cy < h {
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
func (g *Game) Layout(_, _ int) (int, int) { return w, h }
