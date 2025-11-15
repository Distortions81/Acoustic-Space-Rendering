package main

import "sync"

const (
	audioSampleRate      = 44100
	audioDCCouplingAlpha = 0.001
	stereoFrameByteWidth = 4 // two int16 samples per frame (stereo)
)

type centerAudioStream struct {
	mu      sync.Mutex
	pending []float32
	pos     int
	dc      float32
}

func newCenterAudioStream() *centerAudioStream {
	return &centerAudioStream{}
}

func (s *centerAudioStream) SetSample(v float32) {
	s.Enqueue([]float32{v})
}

func (s *centerAudioStream) Enqueue(samples []float32) {
	if len(samples) == 0 {
		return
	}
	s.mu.Lock()
	s.compactPendingLocked()
	for _, v := range samples {
		switch {
		case v > 1:
			v = 1
		case v < -1:
			v = -1
		}
		s.pending = append(s.pending, v)
	}
	s.mu.Unlock()
}

func (s *centerAudioStream) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	frameBytes := len(p) - len(p)%stereoFrameByteWidth
	if frameBytes == 0 {
		return 0, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	for i := 0; i < frameBytes; i += stereoFrameByteWidth {
		sample := s.nextSampleLocked()
		s.dc += audioDCCouplingAlpha * (sample - s.dc)
		filtered := sample - s.dc
		v := int16(filtered * 32767)
		p[i] = byte(v)
		p[i+1] = byte(v >> 8)
		p[i+2] = p[i]
		p[i+3] = p[i+1]
	}
	s.compactPendingLocked()
	return frameBytes, nil
}

func (s *centerAudioStream) Close() error {
	return nil
}

func (s *centerAudioStream) nextSampleLocked() float32 {
	if s.pos < len(s.pending) {
		val := s.pending[s.pos]
		s.pos++
		return val
	}
	return 0
}

func (s *centerAudioStream) compactPendingLocked() {
	if s.pos == 0 {
		return
	}
	if s.pos >= len(s.pending) {
		s.pending = s.pending[:0]
	} else {
		remaining := len(s.pending) - s.pos
		copy(s.pending, s.pending[s.pos:])
		s.pending = s.pending[:remaining]
	}
	s.pos = 0
}
