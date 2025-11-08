package main

import (
	"math"
	"math/rand"
	"time"
)

// generateWalls procedurally creates wall segments within the grid.
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
	g.lastVisCX, g.lastVisCY = -1, -1
}

// trySetWall marks a grid cell as a wall while enforcing spacing from the emitter.
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

// isWall reports whether the coordinates reference a wall cell.
func (g *Game) isWall(x, y int) bool {
	if x < 0 || x >= w || y < 0 || y >= h {
		return true
	}
	if len(g.walls) == 0 {
		return false
	}
	return g.walls[y*w+x]
}

// earOffsets computes the ear indicator positions relative to the listener.
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
