package main

type wallClockAverager struct {
	window int
	buf    []float64
	sum    float64
	idx    int
	count  int
}

func (a *wallClockAverager) Init(window int) {
	a.window = window
	a.buf = nil
	a.sum = 0
	a.idx = 0
	a.count = 0
}

func (a *wallClockAverager) Add(sample float64) float64 {
	if a == nil || a.window <= 1 {
		return sample
	}
	if sample < 0 {
		sample = 0
	}
	if a.buf == nil || len(a.buf) != a.window {
		a.buf = make([]float64, a.window)
		a.sum = 0
		a.idx = 0
		a.count = 0
	}
	if a.count < a.window {
		a.buf[a.idx] = sample
		a.sum += sample
		a.count++
	} else {
		a.sum -= a.buf[a.idx]
		a.buf[a.idx] = sample
		a.sum += sample
	}
	a.idx++
	if a.idx >= a.window {
		a.idx = 0
	}
	if a.count <= 0 {
		return sample
	}
	return a.sum / float64(a.count)
}
