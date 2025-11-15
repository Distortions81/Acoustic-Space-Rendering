package main

type gridOffset struct {
	dx int
	dy int
}

var emitterFootprint = precomputeEmitterFootprint(emitterRad)

func precomputeEmitterFootprint(radius int) []gridOffset {
	footprint := make([]gridOffset, 0, (2*radius+1)*(2*radius+1))
	r2 := radius * radius
	for y := -radius; y <= radius; y++ {
		for x := -radius; x <= radius; x++ {
			if x*x+y*y <= r2 {
				footprint = append(footprint, gridOffset{dx: x, dy: y})
			}
		}
	}
	return footprint
}
