package main

// waveField stores the three wave simulation buffers required by the finite
// difference solver.
type waveField struct {
	width, height int
	curr          []float32
	prev          []float32
	next          []float32
	impulses      []waveImpulse
	shiftScratch  []float32
}

type waveImpulse struct {
	index     int32
	value     float32
	applyPrev bool
}

// audio removal: sample index and PCM helpers removed

// newWaveField allocates a waveField with properly sized buffers.
func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr: make([]float32, width*height),
		prev: make([]float32, width*height),
		next: make([]float32, width*height),
	}
}

// queueImpulse records an impulse to be applied to the device buffers. It
// updates the host-side current buffer for debug visibility and always reports
// that an impulse was enqueued.
func (f *waveField) queueImpulse(x, y int, value float32) bool {
	f.queueImpulseInternal(x, y, value, false)
	return true
}

func (f *waveField) queueImpulseInternal(x, y int, value float32, applyPrev bool) {
	idx := y*f.width + x
	f.curr[idx] = value
	if applyPrev {
		f.prev[idx] = value
	}
	f.impulses = append(f.impulses, waveImpulse{
		index:     int32(idx),
		value:     value,
		applyPrev: applyPrev,
	})
}

// zeroCell clears the current, previous, and next buffers at the given cell.
func (f *waveField) zeroCell(x, y int) {
	idx := y*f.width + x
	f.queueImpulseInternal(x, y, 0, true)
	f.next[idx] = 0
}

func (f *waveField) takeImpulses() []waveImpulse {
	if len(f.impulses) == 0 {
		return nil
	}
	batch := f.impulses
	f.impulses = f.impulses[:0]
	return batch
}

func (f *waveField) reset() {
	for i := range f.curr {
		f.curr[i] = 0
	}
	for i := range f.prev {
		f.prev[i] = 0
	}
	for i := range f.next {
		f.next[i] = 0
	}
	f.impulses = f.impulses[:0]
}

func (f *waveField) shift(dx, dy int) {
	if dx == 0 && dy == 0 {
		return
	}
	if len(f.shiftScratch) != len(f.curr) {
		f.shiftScratch = make([]float32, len(f.curr))
	}
	f.shiftBuffer(f.curr, dx, dy)
	f.shiftBuffer(f.prev, dx, dy)
	f.shiftBuffer(f.next, dx, dy)
	f.impulses = f.impulses[:0]
}

func (f *waveField) shiftBuffer(buf []float32, dx, dy int) {
	copy(f.shiftScratch, buf)
	width, height := f.width, f.height
	for y := 0; y < height; y++ {
		srcY := y + dy
		rowStart := y * width
		if srcY < 0 || srcY >= height {
			for x := 0; x < width; x++ {
				buf[rowStart+x] = 0
			}
			continue
		}
		srcRowStart := srcY * width
		for x := 0; x < width; x++ {
			srcX := x + dx
			if srcX < 0 || srcX >= width {
				buf[rowStart+x] = 0
				continue
			}
			buf[rowStart+x] = f.shiftScratch[srcRowStart+srcX]
		}
	}
}
