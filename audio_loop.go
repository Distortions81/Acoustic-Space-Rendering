package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/hajimehoshi/ebiten/v2/audio/wav"
)

// loadLoopSamples decodes the WAV at path and returns stereo-averaged samples at sampleRate.
func loadLoopSamples(sampleRate int, path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	stream, err := wav.DecodeWithSampleRate(sampleRate, bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decoding %q: %w", path, err)
	}
	decoded, err := io.ReadAll(stream)
	if err != nil {
		return nil, fmt.Errorf("reading decoded %q: %w", path, err)
	}
	if len(decoded) == 0 {
		return nil, fmt.Errorf("wav %q has no audio data", path)
	}
	samples := decodeStereoI16ToFloat(decoded)
	if len(samples) == 0 {
		return nil, fmt.Errorf("wav %q has no usable samples", path)
	}
	return samples, nil
}

func decodeStereoI16ToFloat(pcm []byte) []float32 {
	frameCount := len(pcm) / 4
	if frameCount == 0 {
		return nil
	}
	samples := make([]float32, frameCount)
	for i := 0; i < frameCount; i++ {
		offset := i * 4
		left := int16(binary.LittleEndian.Uint16(pcm[offset : offset+2]))
		right := int16(binary.LittleEndian.Uint16(pcm[offset+2 : offset+4]))
		samples[i] = (float32(left) + float32(right)) * (0.5 / 32768.0)
	}
	return samples
}

type audioPressureSource struct {
	samples []float32
	pos     int
}

func newAudioPressureSource(samples []float32) *audioPressureSource {
	if len(samples) == 0 {
		return nil
	}
	return &audioPressureSource{samples: samples}
}

func (s *audioPressureSource) fillChunk(dst []float32) {
	if len(dst) == 0 {
		return
	}
	if len(s.samples) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	for i := range dst {
		dst[i] = s.samples[s.pos]
		s.pos++
		if s.pos >= len(s.samples) {
			s.pos = 0
		}
	}
}
