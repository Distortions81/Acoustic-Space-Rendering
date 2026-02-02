package main

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
