package main

import (
	"math"
	"sync"
)

const (
	audioSampleRate      = 44100
	audioDCCouplingAlpha = 0.001
	stereoFrameByteWidth = 4 // two int16 samples per frame (stereo)
)

var (
	softClipDrive = float32(2.5)
	softClipNorm  = float32(1.0 / math.Tanh(float64(softClipDrive)))
)

type centerAudioStream struct {
	mu      sync.Mutex
	pending []float32
	pos     int
	dcL     float32
	dcR     float32
}

func newCenterAudioStream() *centerAudioStream {
	return &centerAudioStream{}
}

func (s *centerAudioStream) SetStereo(left, right float32) {
	s.EnqueueInterleaved([]float32{left, right})
}

func (s *centerAudioStream) EnqueueInterleaved(samples []float32) {
	if len(samples) == 0 {
		return
	}
	if len(samples)%2 != 0 {
		samples = samples[:len(samples)-1]
		if len(samples) == 0 {
			return
		}
	}
	s.mu.Lock()
	s.compactPendingLocked()
	for i := 0; i < len(samples); i += 2 {
		left := softClip(samples[i])
		right := softClip(samples[i+1])
		s.pending = append(s.pending, left, right)
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
		left, right := s.nextFrameLocked()
		s.dcL += audioDCCouplingAlpha * (left - s.dcL)
		s.dcR += audioDCCouplingAlpha * (right - s.dcR)
		left = left - s.dcL
		right = right - s.dcR
		lv := int16(left * 32767)
		rv := int16(right * 32767)
		p[i] = byte(lv)
		p[i+1] = byte(lv >> 8)
		p[i+2] = byte(rv)
		p[i+3] = byte(rv >> 8)
	}
	s.compactPendingLocked()
	return frameBytes, nil
}

func (s *centerAudioStream) Close() error {
	return nil
}

func (s *centerAudioStream) nextFrameLocked() (float32, float32) {
	if s.pos+1 < len(s.pending) {
		left := s.pending[s.pos]
		right := s.pending[s.pos+1]
		s.pos += 2
		return left, right
	}
	return 0, 0
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

func softClip(x float32) float32 {
	if x == 0 {
		return 0
	}
	y := float32(math.Tanh(float64(x*softClipDrive))) * softClipNorm
	if y > 1 {
		return 1
	}
	if y < -1 {
		return -1
	}
	return y
}
