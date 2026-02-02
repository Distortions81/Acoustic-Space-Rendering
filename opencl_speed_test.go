package main

import (
	"math"
	"os"
	"testing"
)

func TestOpenCLWaveFrontSpeed_MaxStable(t *testing.T) {
	if os.Getenv("ASR_OPENCL_TEST") != "1" {
		t.Skip("set ASR_OPENCL_TEST=1 to run OpenCL integration tests")
	}

	// Keep this test stable and fast:
	// - Use explicit dx (avoid derived world scale differences)
	// - Disable damping/absorption so the wave is easy to detect.
	if rt60SecondsFlag != nil {
		*rt60SecondsFlag = 0
	}
	if airAbsDbPerMFlag != nil {
		*airAbsDbPerMFlag = 0
	}
	worldScaleMetersPerCell = 0.02 // dx = 2cm

	const (
		width  = 256
		height = 256
	)

	solver, err := newOpenCLWaveSolver(width, height)
	if err != nil {
		t.Skipf("OpenCL not available: %v", err)
	}
	defer solver.Close()

	field := newWaveField(width, height)
	walls := make([]bool, width*height)

	cx, cy := width/2, height/2
	field.queueImpulseInternal(cx, cy, 1.0, true)

	distanceCells := 48
	probeX := cx + distanceCells
	probeY := cy
	probeIdx := int32(probeY*width + probeX)

	stepsPerSecond := float64(audioSampleRate) // convenient, stable
	dt := 1.0 / stepsPerSecond
	steps := 240
	dtSecondsBatch := float64(steps) / stepsPerSecond

	if err := solver.Step(field, walls, steps, dtSecondsBatch, true, false, false, false, nil, 0, probeIdx, probeIdx, nil); err != nil {
		t.Fatalf("solver.Step: %v", err)
	}

	samples := solver.EarSamplesInterleaved()
	if len(samples) < steps*2 {
		t.Fatalf("expected %d ear samples, got %d", steps*2, len(samples))
	}

	series := make([]float64, steps)
	maxAbs := 0.0
	for i := 0; i < steps; i++ {
		v := float64(samples[i*2]) // left ear == probe index
		av := math.Abs(v)
		series[i] = av
		if av > maxAbs {
			maxAbs = av
		}
	}
	if maxAbs == 0 {
		t.Fatalf("probe never saw any signal (maxAbs=0)")
	}

	// Estimate arrival time as the first time the probe exceeds a fraction of the
	// observed peak. This is robust to impulse shape variations.
	threshold := maxAbs * 0.2
	arrival := -1
	for i, av := range series {
		if av >= threshold {
			arrival = i
			break
		}
	}
	if arrival < 0 {
		t.Fatalf("probe never crossed threshold (threshold=%.6g maxAbs=%.6g)", threshold, maxAbs)
	}

	dx := cellSizeMeters()
	if dx <= 0 {
		t.Fatalf("cellSizeMeters() must be > 0 (got %.6g)", dx)
	}
	distanceM := float64(distanceCells) * dx

	// The solver currently runs at the maximum stable stencil speedCoeff=0.5:
	//   speedCoeff = (c_eff*dt/dx)^2 => c_eff = sqrt(0.5)*dx/dt
	expectedC := math.Sqrt(maxStableSpeedCoeff) * dx / dt
	measuredC := distanceM / (float64(arrival+1) * dt)
	t.Logf("measured %.1f m/s vs expected %.1f m/s (arrival step %d, dx=%.4fm, dt=%.6fs)", measuredC, expectedC, arrival, dx, dt)

	// Allow a fairly wide tolerance because we're inferring speed from a single
	// time-series point and a non-ideal impulse.
	tol := 0.35
	if math.Abs(measuredC-expectedC) > expectedC*tol {
		t.Fatalf("unexpected propagation speed: measured %.1f m/s, expected %.1f m/s (arrival step %d, dx=%.4fm dt=%.6fs)",
			measuredC, expectedC, arrival, dx, dt)
	}
}
