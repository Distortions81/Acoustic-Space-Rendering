package main

// rebuildInteriorMask recalculates the worker masks describing non-wall cells.
func (g *Game) rebuildInteriorMask() {
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	rows := make([]rowMask, 0, h-2)
	for y := 1; y < h-1; y++ {
		base := y * w
		spans := make([]span, 0, 8)
		in := false
		start := 0
		for x := 1; x < w-1; x++ {
			blocked := g.walls[base+x]
			if !blocked && !in {
				in = true
				start = x
			}
			if (blocked || x == w-2) && in {
				end := x - 1
				if x == w-2 && !blocked {
					end = x
				}
				if end >= start {
					spans = append(spans, span{start: start, end: end})
				}
				in = false
			}
		}
		if len(spans) == 0 {
			continue
		}
		rows = append(rows, rowMask{y: y, spans: spans})
	}
	g.workerMasks = assignRowMasks(g.workerCount, rows)
}
