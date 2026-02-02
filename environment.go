package main

import (
	"math"
	"math/rand"
	"time"
)

func cellSizeMeters() float64 {
	if worldScaleMetersPerCell > 0 {
		return worldScaleMetersPerCell
	}
	return defaultCellSizeM
}

func metersToCells(distanceM float64) int {
	dx := cellSizeMeters()
	if dx <= 0 {
		return 0
	}
	return int(math.Round(distanceM / dx))
}

func wallHalfThicknessCells() int {
	thicknessM := roomWallThicknessM
	if thicknessM < 0 {
		thicknessM = 0
	}
	widthCells := metersToCells(thicknessM)
	if widthCells < 1 {
		widthCells = 1
	}
	if widthCells%2 == 0 {
		widthCells++
	}
	return (widthCells - 1) / 2
}

func wallHalfThicknessJitterCells() int {
	jitterM := roomWallThicknessJitM
	if jitterM <= 0 {
		return 0
	}
	jitterCells := metersToCells(jitterM)
	if jitterCells < 0 {
		jitterCells = 0
	}
	// Jitter is applied to the half-thickness to keep the wall centered.
	if jitterCells%2 != 0 {
		jitterCells++
	}
	return jitterCells / 2
}

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

	segments := roomWallSegments
	minLenM := roomWallMinLenM
	maxLenM := roomWallMaxLenM
	minLenCells := metersToCells(minLenM)
	maxLenCells := metersToCells(maxLenM)
	if minLenCells < 1 {
		minLenCells = 1
	}
	if maxLenCells < 1 {
		maxLenCells = 1
	}
	if minLenCells > maxLenCells {
		minLenCells, maxLenCells = maxLenCells, minLenCells
	}
	baseHalfThickness := wallHalfThicknessCells()
	jitterHalfThickness := wallHalfThicknessJitterCells()

	for s := 0; s < segments; s++ {
		lengthRange := maxLenCells - minLenCells + 1
		if lengthRange <= 0 {
			lengthRange = 1
		}
		length := minLenCells + g.levelRand.Intn(lengthRange)
		thickness := baseHalfThickness
		if jitterHalfThickness > 0 {
			thickness += g.levelRand.Intn(jitterHalfThickness + 1)
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
	g.wallsDirty = true
}

// trySetWall marks a grid cell as a wall while enforcing spacing from the emitter.
func (g *Game) trySetWall(x, y int) {
	if x <= 1 || x >= w-1 || y <= 1 || y >= h-1 {
		return
	}
	exclusionM := roomWallExclusionM
	if exclusionM < 0 {
		exclusionM = 0
	}
	exclusionCells := metersToCells(exclusionM)
	if exclusionCells < 0 {
		exclusionCells = 0
	}
	dx := float64(x) - g.listenerX
	dy := float64(y) - g.listenerY
	if dx*dx+dy*dy < float64(exclusionCells*exclusionCells) {
		return
	}
	idx := y*w + x
	g.walls[idx] = true
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
	// Convert desired ear spacing (meters) into a half-offset in grid cells.
	earHalfSpacingM := 0.5 * defaultEarSpacingM
	earOffsetCells := int(math.Round(earHalfSpacingM / cellSizeMeters()))
	if earOffsetCells < 1 {
		earOffsetCells = 1
	}
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
