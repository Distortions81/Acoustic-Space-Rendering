package main

import (
	"fmt"
	"math"
)

const (
	rt60AmplitudeDecayLn1000 = 6.907755278982137 // ln(1000) for -60 dB amplitude
	maxStableSpeedCoeff      = 0.5               // (c*dt/dx)^2 limit for 2D 5-point stencil
)

type waveCoefficients struct {
	DampPerStep  float32
	SpeedCoeff   float32
	StepsPerSec  float64
	DtSeconds    float64
	DxMeters     float64
	AirTempC     float64
	SpeedSoundMS float64
	Courant      float64
	Clamped      bool

	DispersionCenterHz float64
	DispersionFactor   float64
}

func speedOfSoundMS(tempC float64) float64 {
	// Approximation for dry air near room temperature.
	return 331.3 + 0.606*tempC
}

func computeWaveCoefficients(stepsPerSecond float64) (waveCoefficients, error) {
	dx := 0.0
	dx = cellSizeMeters()
	if dx <= 0 {
		return waveCoefficients{}, fmt.Errorf("cell size must be > 0 meters (got %.6f)", dx)
	}
	if stepsPerSecond <= 0 {
		return waveCoefficients{}, fmt.Errorf("steps per second must be > 0 (got %.6f)", stepsPerSecond)
	}
	dt := 1.0 / stepsPerSecond
	tempC := defaultAirTempC
	if airTempCFlag != nil {
		tempC = *airTempCFlag
	}
	c := speedOfSoundMS(tempC)
	if c <= 0 {
		return waveCoefficients{}, fmt.Errorf("computed speed of sound must be > 0 (got %.6f)", c)
	}

	courant := c * dt / dx
	speedCoeff := courant * courant
	clamped := false
	if speedCoeff > maxStableSpeedCoeff {
		speedCoeff = maxStableSpeedCoeff
		clamped = true
	}

	dispersionHz := 0.0
	dispersionFactor := 1.0
	if dispersionCenterHzFlag != nil && *dispersionCenterHzFlag > 0 {
		dispersionHz = *dispersionCenterHzFlag
		// Try to counter numerical dispersion by choosing an effective Courant number
		// such that the phase speed matches c at the requested frequency.
		//
		// For a 2D 5-point stencil, the discrete dispersion relation is:
		//   sin^2(ω dt / 2) = C^2 [sin^2(kx dx / 2) + sin^2(ky dx / 2)]
		// We choose a representative direction by averaging axis and diagonal travel.
		k := 2.0 * math.Pi * dispersionHz / c
		omegaTarget := c * k
		sinOmega := math.Sin(omegaTarget * dt / 2.0)
		sinAxis := math.Sin(k * dx / 2.0)
		sinDiag := math.Sin(k * dx / (2.0 * math.Sqrt2))
		denomAxis := math.Abs(sinAxis)
		denomDiag := math.Sqrt(2.0 * sinDiag * sinDiag)
		if denomAxis > 1e-8 && denomDiag > 1e-8 {
			cAxis := math.Abs(sinOmega) / denomAxis
			cDiag := math.Abs(sinOmega) / denomDiag
			cTarget := 0.5 * (cAxis + cDiag)
			targetCoeff := cTarget * cTarget
			if targetCoeff > 0 {
				dispersionFactor = targetCoeff / speedCoeff
				speedCoeff = targetCoeff
				if speedCoeff > maxStableSpeedCoeff {
					speedCoeff = maxStableSpeedCoeff
					clamped = true
				}
			}
		}
	}

	dampPerStep := 1.0
	if rt60SecondsFlag != nil && *rt60SecondsFlag > 0 {
		dampPerStep = math.Exp(-rt60AmplitudeDecayLn1000 * dt / *rt60SecondsFlag)
		if dampPerStep < 0 {
			dampPerStep = 0
		} else if dampPerStep > 1 {
			dampPerStep = 1
		}
	}
	if airAbsDbPerMFlag != nil && *airAbsDbPerMFlag > 0 {
		// Apply a simple, frequency-agnostic air absorption model as amplitude
		// attenuation per meter. We approximate traveled distance per solver step
		// as c*dt.
		dist := c * dt
		amp := math.Pow(10.0, -(*airAbsDbPerMFlag*dist)/20.0)
		dampPerStep *= amp
		if dampPerStep < 0 {
			dampPerStep = 0
		} else if dampPerStep > 1 {
			dampPerStep = 1
		}
	}

	return waveCoefficients{
		DampPerStep:        float32(dampPerStep),
		SpeedCoeff:         float32(speedCoeff),
		StepsPerSec:        stepsPerSecond,
		DtSeconds:          dt,
		DxMeters:           dx,
		AirTempC:           tempC,
		SpeedSoundMS:       c,
		Courant:            courant,
		Clamped:            clamped,
		DispersionCenterHz: dispersionHz,
		DispersionFactor:   dispersionFactor,
	}, nil
}
