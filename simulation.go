package main

// stepWaveCPU executes one CPU simulation tick, synchronizing worker goroutines
// and applying boundary conditions.
func (g *Game) stepWaveCPU() {
	g.workerMu.Lock()
	g.workerPending = g.workerCount
	g.workerStep++
	g.workerCond.Broadcast()
	for g.workerPending > 0 {
		g.workerCond.Wait()
	}
	g.workerMu.Unlock()
	g.field.zeroBoundaries()
	g.field.swap()
}

// stepWaveCPUBatch runs multiple simulation ticks and records audio samples when enabled.
func (g *Game) stepWaveCPUBatch(steps int) {
	if steps <= 0 {
		if !g.audioDisabled {
			g.ensurePressureSampleCapacity(0)
		}
		return
	}
	if g.audioDisabled {
		for i := 0; i < steps; i++ {
			g.stepWaveCPU()
		}
		return
	}
	g.ensurePressureSampleCapacity(steps)
	for i := 0; i < steps; i++ {
		g.stepWaveCPU()
		g.latestPressureSamples[i] = g.samplePressureAtIndex()
	}
}
