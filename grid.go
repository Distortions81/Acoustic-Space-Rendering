package main

// intPoint represents an integer coordinate on the simulation grid.
type intPoint struct {
	x int
	y int
}

// buildLOSPerimeterTargets precomputes the grid edge cells used when casting
// line-of-sight rays.
func buildLOSPerimeterTargets() []intPoint {
	points := make([]intPoint, 0, 2*(w+h))
	for x := 0; x < w; x++ {
		points = append(points, intPoint{x: x, y: 0})
		points = append(points, intPoint{x: x, y: h - 1})
	}
	for y := 1; y < h-1; y++ {
		points = append(points, intPoint{x: 0, y: y})
		points = append(points, intPoint{x: w - 1, y: y})
	}
	return points
}

// clampCoord constrains v to lie within the inclusive [min, max] range.
func clampCoord(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// losPerimeterTargets caches the perimeter cells used for visibility checks.
var losPerimeterTargets = buildLOSPerimeterTargets()
