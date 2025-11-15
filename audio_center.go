package main

import (
	"sync"
)

const (
	audioSampleRate = 48000
)

type centerAudioStream struct {
	mu     sync.Mutex
	sample float32
	dc     float32
}

func newCenterAudioStream() *centerAudioStream {
	return &centerAudioStream{}
}

func (s *centerAudioStream) SetSample(v float32) {
	if v > 1 {
		v = 1
	} else if v < -1 {
		v = -1
	}
	s.mu.Lock()
	// Simple AC coupling: remove a slowly varying DC component.
	const alpha = 0.001
	s.dc += alpha * (v - s.dc)
	s.sample = v - s.dc
	s.mu.Unlock()
}

func (s *centerAudioStream) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	// Ensure we generate whole stereo frames (4 bytes per frame).
	frameBytes := len(p) - len(p)%4
	if frameBytes == 0 {
		return 0, nil
	}
	s.mu.Lock()
	sample := s.sample
	s.mu.Unlock()

	for i := 0; i < frameBytes; i += 4 {
		v := int16(sample * 32767)
		p[i] = byte(v)
		p[i+1] = byte(v >> 8)
		p[i+2] = p[i]
		p[i+3] = p[i+1]
	}
	return frameBytes, nil
}

func (s *centerAudioStream) Close() error {
	return nil
}
