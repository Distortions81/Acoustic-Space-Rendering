package main

import "math"

// waveField stores the three wave simulation buffers required by the finite
// difference solver.
type waveField struct {
	width, height int
	curr          []float32
	prev          []float32
	next          []float32
}

// defaultPressureSampleIndex selects the central cell used when exporting audio.
func defaultPressureSampleIndex(width, height int) int {
	return (height/2)*width + (width / 2)
}

// floatToPCM16 converts a normalized float sample into signed 16-bit PCM.
func floatToPCM16(v float32) int16 {
	if v > 1 {
		v = 1
	} else if v < -1 {
		v = -1
	}
	return int16(math.Round(float64(v) * float64(pcm16MaxValue)))
}

// newWaveField allocates a waveField with properly sized buffers.
func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr: make([]float32, width*height),
		prev: make([]float32, width*height),
		next: make([]float32, width*height),
	}
}

// setCurr writes a value into the current state buffer.
func (f *waveField) setCurr(x, y int, value float32) {
	f.curr[y*f.width+x] = value
}

// zeroCell clears the current, previous, and next buffers at the given cell.
func (f *waveField) zeroCell(x, y int) {
	idx := y*f.width + x
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
