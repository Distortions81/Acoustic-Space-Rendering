//go:build !opencl

package main

import "errors"

type openCLWaveSolver struct{}

func newOpenCLWaveSolver(width, height int) (*openCLWaveSolver, error) {
	return nil, errors.New("OpenCL support is not enabled; rebuild with -tags opencl")
}

func (s *openCLWaveSolver) Step(field *waveField, walls []bool) error {
	return errors.New("OpenCL solver unavailable")
}

func (s *openCLWaveSolver) Close() {}

func (s *openCLWaveSolver) DeviceName() string { return "" }
