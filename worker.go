package main

import "sync"

// span represents an inclusive column range inside a row mask.
type span struct{ start, end int }

// rowMask groups contiguous spans for a single row that requires computation.
type rowMask struct {
	y     int
	spans []span
}

// workerMask collects the row masks assigned to a worker goroutine.
type workerMask struct {
	rows []rowMask
}

// rowCache stores scratch buffers reused by worker goroutines to limit
// allocations during wave propagation.
type rowCache struct {
	center []float32
	prev   []float32
	top    []float32
	bottom []float32
}

// newRowCache constructs a cache sized for a specific grid width.
func newRowCache(width int) *rowCache {
	return &rowCache{
		center: make([]float32, width),
		prev:   make([]float32, width),
		top:    make([]float32, width),
		bottom: make([]float32, width),
	}
}

// waveWorkerLoop executes CPU wave updates for rows assigned to the worker.
func (g *Game) waveWorkerLoop(index int) {
	cache := newRowCache(g.field.width)
	lastStep := 0
	g.workerMu.Lock()
	for {
		for g.workerStep == lastStep {
			g.workerCond.Wait()
		}
		lastStep = g.workerStep
		var mask workerMask
		if index < len(g.workerMasks) {
			mask = g.workerMasks[index]
		}
		g.workerMu.Unlock()

		if len(mask.rows) > 0 {
			processMask(g.field, &mask, cache)
		}

		g.workerMu.Lock()
		g.workerPending--
		if g.workerPending == 0 {
			g.workerCond.Broadcast()
		}
	}
}

// processMask steps the finite difference solver over the provided worker mask.
func processMask(field *waveField, mask *workerMask, _ *rowCache) {
	width := field.width
	wd := waveDamp32
	ws := waveSpeed32
	for _, row := range mask.rows {
		y := row.y
		rowBase := y * width
		topBase := (y - 1) * width
		bottomBase := (y + 1) * width
		field.next[rowBase+0] = 0
		field.next[rowBase+width-1] = 0

		center := field.curr[rowBase : rowBase+width]
		prev := field.prev[rowBase : rowBase+width]
		top := field.curr[topBase : topBase+width]
		bottom := field.curr[bottomBase : bottomBase+width]
		nextRow := field.next[rowBase : rowBase+width]

		for _, sp := range row.spans {
			start := sp.start
			if start < 1 {
				start = 1
			}
			end := sp.end
			if end > width-2 {
				end = width - 2
			}

			x := start
			for ; x+3 <= end; x += 4 {
				c0 := center[x]
				lap0 := center[x-1] + center[x+1] + top[x] + bottom[x] - 4*c0
				nextRow[x] = ((2*c0 - prev[x]) + ws*lap0) * wd

				x1 := x + 1
				c1 := center[x1]
				lap1 := center[x1-1] + center[x1+1] + top[x1] + bottom[x1] - 4*c1
				nextRow[x1] = ((2*c1 - prev[x1]) + ws*lap1) * wd

				x2 := x + 2
				c2 := center[x2]
				lap2 := center[x2-1] + center[x2+1] + top[x2] + bottom[x2] - 4*c2
				nextRow[x2] = ((2*c2 - prev[x2]) + ws*lap2) * wd

				x3 := x + 3
				c3 := center[x3]
				lap3 := center[x3-1] + center[x3+1] + top[x3] + bottom[x3] - 4*c3
				nextRow[x3] = ((2*c3 - prev[x3]) + ws*lap3) * wd
			}
			for ; x <= end; x++ {
				c := center[x]
				lap := center[x-1] + center[x+1] + top[x] + bottom[x] - 4*c
				nextRow[x] = ((2*c - prev[x]) + ws*lap) * wd
			}
		}
	}
}

// convertRow copies floating-point data between buffers; it remains available
// for potential future optimizations.
func convertRow(src []float32, dst []float32) {
	copy(dst, src)
}

// assignRowMasks distributes row masks across worker goroutines in round robin fashion.
func assignRowMasks(workerCount int, rows []rowMask) []workerMask {
	if workerCount < 1 {
		workerCount = 1
	}
	masks := make([]workerMask, workerCount)
	for idx, row := range rows {
		workerIdx := idx % workerCount
		masks[workerIdx].rows = append(masks[workerIdx].rows, row)
	}
	return masks
}

// startWorkers launches the background goroutines that execute CPU wave steps.
func (g *Game) startWorkers() {
	if g.workersStarted {
		return
	}
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	if g.workerCond == nil {
		g.workerCond = sync.NewCond(&g.workerMu)
	}
	g.workersStarted = true
	for i := 0; i < g.workerCount; i++ {
		go g.waveWorkerLoop(i)
	}
}
