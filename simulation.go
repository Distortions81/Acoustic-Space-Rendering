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

// stepWaveCPUBatch runs multiple simulation ticks.
func (g *Game) stepWaveCPUBatch(steps int) {
    if steps <= 0 {
        return
    }
    for i := 0; i < steps; i++ {
        g.stepWaveCPU()
    }
}
