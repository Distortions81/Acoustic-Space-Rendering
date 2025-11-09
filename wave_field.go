package main

// waveField stores the three wave simulation buffers required by the finite
// difference solver.
type waveField struct {
	width, height int
	curr          []float32
	prev          []float32
	next          []float32
	impulses      []waveImpulse
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
