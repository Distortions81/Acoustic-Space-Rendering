package main

import "math"

// refreshVisibleMask recomputes line-of-sight occlusion around the listener.
func (g *Game) refreshVisibleMask() {
	if len(g.visibleStamp) != w*h {
		g.visibleStamp = make([]uint32, w*h)
	}
	cx := clampCoord(int(math.Round(g.ex)), 0, w-1)
	cy := clampCoord(int(math.Round(g.ey)), 0, h-1)
	if g.lastVisCX == cx && g.lastVisCY == cy {
		return
	}
	if g.visibleGen == ^uint32(0) {
		for i := range g.visibleStamp {
			g.visibleStamp[i] = 0
		}
		g.visibleGen = 1
	} else {
		g.visibleGen++
	}
	g.visibleStamp[cy*w+cx] = g.visibleGen
	fx, fy := g.listenerForwardX, g.listenerForwardY
	if fx == 0 && fy == 0 {
		fx, fy = 0, -1
	}
	mag := math.Hypot(fx, fy)
	if mag == 0 {
		fx, fy = 0, -1
		mag = 1
	}
	fx /= mag
	fy /= mag
	fovDeg := *fovDegreesFlag
	if fovDeg < 1 {
		fovDeg = 1
	} else if fovDeg > 180 {
		fovDeg = 180
	}
	halfAngleRad := fovDeg * math.Pi / 180.0 / 2.0
	cosHalf := math.Cos(halfAngleRad)
	cosHalfSq := cosHalf * cosHalf
	maxLeft := cx
	maxRight := (w - 1) - cx
	maxUp := cy
	maxDown := (h - 1) - cy
	radius := maxLeft
	if maxRight > radius {
		radius = maxRight
	}
	if maxUp > radius {
		radius = maxUp
	}
	if maxDown > radius {
		radius = maxDown
	}
	g.computeFOVShadow(cx, cy, radius, fx, fy, cosHalfSq)
	visCount := 0
	for i := 0; i < w*h; i++ {
		if g.visibleStamp[i] == g.visibleGen {
			visCount++
			if visCount > 128 {
				break
			}
		}
	}
	if visCount <= 1 {
		for _, target := range losPerimeterTargets {
			vx := float64(target.x - cx)
			vy := float64(target.y - cy)
			dot := vx*fx + vy*fy
			if dot <= 0 || dot*dot < (vx*vx+vy*vy)*cosHalfSq {
				continue
			}
			g.castVisibilityRay(cx, cy, target.x, target.y)
		}
	}
	g.lastVisCX, g.lastVisCY = cx, cy
}

// computeFOVShadow performs symmetrical shadowcasting for a limited FOV cone.
func (g *Game) computeFOVShadow(cx, cy, radius int, fx, fy float64, cosHalfSq float64) {
	oct := [8][4]int{
		{1, 0, 0, 1},
		{0, 1, 1, 0},
		{-1, 0, 0, 1},
		{0, 1, -1, 0},
		{-1, 0, 0, -1},
		{0, -1, -1, 0},
		{1, 0, 0, -1},
		{0, -1, 1, 0},
	}
	for i := 0; i < 8; i++ {
		g.castLight(cx, cy, 1, 1.0, 0.0, radius, oct[i][0], oct[i][1], oct[i][2], oct[i][3], fx, fy, cosHalfSq)
	}
}

// castLight recursively explores an octant collecting visible cells.
func (g *Game) castLight(cx, cy, row int, startSlope, endSlope float64, radius int, xx, xy, yx, yy int, fx, fy float64, cosHalfSq float64) {
	if startSlope < endSlope {
		return
	}
	radiusSq := radius * radius
	for i := row; i <= radius; i++ {
		blocked := false
		newStart := 0.0
		for dx := -i; dx <= 0; dx++ {
			dy := -i
			lSlope := (float64(dx) - 0.5) / (float64(dy) + 0.5)
			rSlope := (float64(dx) + 0.5) / (float64(dy) - 0.5)
			if rSlope > startSlope {
				continue
			}
			if lSlope < endSlope {
				break
			}
			X := cx + dx*xx + dy*xy
			Y := cy + dx*yx + dy*yy
			if X < 0 || X >= w || Y < 0 || Y >= h {
				continue
			}
			distSq := dx*dx + dy*dy
			if distSq <= radiusSq {
				vx := float64(X - cx)
				vy := float64(Y - cy)
				dot := vx*fx + vy*fy
				r2 := vx*vx + vy*vy
				if dot > 0 && dot*dot >= r2*cosHalfSq {
					g.visibleStamp[Y*w+X] = g.visibleGen
				}
			}
			wall := g.isWall(X, Y)
			if blocked {
				if wall {
					newStart = rSlope
					continue
				}
				blocked = false
				startSlope = newStart
			} else if wall && i < radius {
				blocked = true
				g.castLight(cx, cy, i+1, startSlope, lSlope, radius, xx, xy, yx, yy, fx, fy, cosHalfSq)
				newStart = rSlope
			}
		}
		if blocked {
			break
		}
	}
}

// castVisibilityRay performs a Bresenham ray cast to mark visible cells.
func (g *Game) castVisibilityRay(x0, y0, x1, y1 int) {
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
		if x0 < 0 || x0 >= w || y0 < 0 || y0 >= h {
			break
		}
		idx := y0*w + x0
		g.visibleStamp[idx] = g.visibleGen
		if g.isWall(x0, y0) && !(x0 == x1 && y0 == y1) {
			break
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
