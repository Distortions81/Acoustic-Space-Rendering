package main

// waveField stores the three wave simulation buffers required by the finite
// difference solver.
type waveField struct {
	width, height int
	curr          []float32
	prev          []float32
	next          []float32
	currDirty     bool
}

// audio removal: sample index and PCM helpers removed

// newWaveField allocates a waveField with properly sized buffers.
func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr:      make([]float32, width*height),
		prev:      make([]float32, width*height),
		next:      make([]float32, width*height),
		currDirty: true,
	}
}

func (f *waveField) markCurrDirty() {
	f.currDirty = true
}

// setCurr writes a value into the current state buffer and reports whether the
// write changed the cell.
func (f *waveField) setCurr(x, y int, value float32) bool {
	idx := y*f.width + x
	if f.curr[idx] == value {
		return false
	}
	f.curr[idx] = value
	f.markCurrDirty()
	return true
}

// zeroCell clears the current, previous, and next buffers at the given cell.
func (f *waveField) zeroCell(x, y int) {
	idx := y*f.width + x
	if f.curr[idx] != 0 {
		f.markCurrDirty()
	}
	f.curr[idx] = 0
	f.prev[idx] = 0
	f.next[idx] = 0
}

// readCurr returns the value in the current buffer at the given coordinates.
func (f *waveField) readCurr(x, y int) float32 {
	return f.curr[y*f.width+x]
}

// swap rotates the triple buffers so that next becomes current and current becomes previous.
func (f *waveField) swap() {
	f.prev, f.curr, f.next = f.curr, f.next, f.prev
}

// currWasModified reports whether host-side writes have altered the current
// buffer since the last GPU upload.
func (f *waveField) currWasModified() bool {
	return f.currDirty
}

// clearCurrDirty marks the current buffer as synchronized with the device.
func (f *waveField) clearCurrDirty() {
	f.currDirty = false
}

// zeroBoundaries applies absorbing boundary conditions on the edges of the grid
// to prevent reflections back into the simulation domain.
func (f *waveField) zeroBoundaries() {
	lastRow := f.height - 1
	lastCol := f.width - 1
	reflect := float32(boundaryReflect)
	for x := 0; x < f.width; x++ {
		top := f.next[1*f.width+x]
		bottom := f.next[(lastRow-1)*f.width+x]
		f.next[0*f.width+x] = -top * reflect
		f.next[lastRow*f.width+x] = -bottom * reflect
	}
	for y := 1; y < lastRow; y++ {
		left := f.next[y*f.width+1]
		right := f.next[y*f.width+lastCol-1]
		f.next[y*f.width+0] = -left * reflect
		f.next[y*f.width+lastCol] = -right * reflect
	}
}
