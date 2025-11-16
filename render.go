package main

import (
	"fmt"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

type renderState struct {
	simImage *ebiten.Image
}

// Draw renders the current wave field, ear indicators, and optional overlays.
func (g *Game) Draw(screen *ebiten.Image) {
	g.ensureSimImage()
	if g.gpuSolver != nil && g.simImage != nil {
		pixels := g.gpuSolver.PixelBytes()
		if view := g.viewportPixels(pixels); len(view) == viewportWidth*viewportHeight*4 {
			g.simImage.WritePixels(view)
		}
	}
	if g.simImage != nil && viewportWidth > 0 && viewportHeight > 0 {
		opts := &ebiten.DrawImageOptions{}
		if viewportWidth > 0 && viewportHeight > 0 {
			opts.GeoM.Scale(float64(windowWidth)/float64(viewportWidth), float64(windowHeight)/float64(viewportHeight))
		}
		opts.Filter = ebiten.FilterLinear
		screen.DrawImage(g.simImage, opts)
	}

	g.drawWallsOverlay(screen)
	g.drawEmitterIndicator(screen)

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
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	if outsideWidth <= 0 {
		outsideWidth = referenceWidth
	}
	if outsideHeight <= 0 {
		outsideHeight = referenceHeight
	}
	windowWidth = outsideWidth
	windowHeight = outsideHeight
	return outsideWidth, outsideHeight
}

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

func (g *Game) ensureSimImage() {
	if viewportWidth <= 0 || viewportHeight <= 0 {
		g.simImage = nil
		return
	}
	if g.simImage == nil || g.simImage.Bounds().Dx() != viewportWidth || g.simImage.Bounds().Dy() != viewportHeight {
		g.simImage = ebiten.NewImage(viewportWidth, viewportHeight)
	}
}

func (g *Game) drawEmitterIndicator(screen *ebiten.Image) {
	if screen == nil {
		return
	}
	red := color.RGBA{255, 0, 0, 255}
	centerX := float64(windowWidth) / 2
	centerY := float64(windowHeight) / 2
	size := 3.0
	offset := size / 2
	ebitenutil.DrawRect(screen, centerX-offset, centerY-offset, size, size, red)
}

func (g *Game) drawWallsOverlay(screen *ebiten.Image) {
	if !*showWallsFlag || viewportWidth <= 0 || viewportHeight <= 0 || screen == nil || len(g.walls) != w*h {
		return
	}
	scaleX := float64(windowWidth) / float64(viewportWidth)
	scaleY := float64(windowHeight) / float64(viewportHeight)
	wallColor := color.RGBA{30, 40, 80, 255}
	startX := g.viewX
	startY := g.viewY
	for localY := 0; localY < viewportHeight; localY++ {
		worldY := startY + localY
		if worldY < 0 || worldY >= h {
			continue
		}
		rowBase := worldY * w
		for localX := 0; localX < viewportWidth; localX++ {
			worldX := startX + localX
			if worldX < 0 || worldX >= w {
				continue
			}
			if !g.walls[rowBase+worldX] {
				continue
			}
			x := float64(localX) * scaleX
			y := float64(localY) * scaleY
			x1 := float64(localX+1) * scaleX
			y1 := float64(localY+1) * scaleY
			width := x1 - x
			height := y1 - y
			if width < 1 {
				width = 1
			}
			if height < 1 {
				height = 1
			}
			ebitenutil.DrawRect(screen, x, y, width, height, wallColor)
		}
	}
}
