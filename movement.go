package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

func (g *Game) handleControlToggle() {
	if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
		g.controlEmitter = !g.controlEmitter
	}
}

func (g *Game) tickSeconds() float64 {
	if g != nil && g.lastUpdateDT > 0 {
		return g.lastUpdateDT
	}
	if defaultTPS <= 0 {
		return 0
	}
	return 1.0 / defaultTPS
}

func (g *Game) movementStepCellsPerTick(speedMPS float64) float64 {
	dxMeters := cellSizeMeters()
	if speedMPS < 0 {
		speedMPS = 0
	}
	dt := g.tickSeconds()
	if dt <= 0 {
		return 0
	}
	return speedMPS * dt / dxMeters
}

// enableAutoWalk schedules scripted movement for a limited duration.
func (g *Game) enableAutoWalk(duration time.Duration) {
	g.autoWalk = true
	g.autoWalkDeadline = time.Now().Add(duration)
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 3))
	}
	g.autoWalkFrameCount = 0
}

// movementVector selects either manual or automatic movement direction.
func (g *Game) movementVector() (float64, float64) {
	if g.autoWalk {
		if time.Now().After(g.autoWalkDeadline) {
			g.autoWalk = false
			return 0, 0
		}
		return g.autoWalkVector()
	}
	return g.manualMovementVector()
}

// manualMovementVector returns WASD-based input movement scaled by moveSpeed.
func (g *Game) manualMovementVector() (float64, float64) {
	dx, dy := 0.0, 0.0
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		dy -= 1
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) {
		dy += 1
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) {
		dx -= 1
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) {
		dx += 1
	}
	if dx != 0 && dy != 0 {
		dx *= 0.7071
		dy *= 0.7071
	}
	speedMPS := defaultRunSpeedMPS
	if runSpeedMPSFlag != nil {
		speedMPS = *runSpeedMPSFlag
	}
	if ebiten.IsKeyPressed(ebiten.KeyShiftLeft) || ebiten.IsKeyPressed(ebiten.KeyShiftRight) {
		speedMPS = defaultWalkSpeedMPS
		if walkSpeedMPSFlag != nil {
			speedMPS = *walkSpeedMPSFlag
		}
	}
	stepCells := g.movementStepCellsPerTick(speedMPS)
	return dx * stepCells, dy * stepCells
}

// autoWalkVector returns a pseudo-random, collision-aware movement vector.
func (g *Game) autoWalkVector() (float64, float64) {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 4))
	}
	speedMPS := defaultRunSpeedMPS
	if runSpeedMPSFlag != nil {
		speedMPS = *runSpeedMPSFlag
	}
	stepCells := g.movementStepCellsPerTick(speedMPS)
	for attempts := 0; attempts < 5; attempts++ {
		if g.autoWalkFrameCount <= 0 {
			g.randomizeAutoWalkDirection()
		}
		nextX := g.listenerX + g.autoWalkDirX*stepCells
		nextY := g.listenerY + g.autoWalkDirY*stepCells
		if nextX > float64(emitterRad) && nextX < float64(w-emitterRad-1) &&
			nextY > float64(emitterRad) && nextY < float64(h-emitterRad-1) &&
			!g.isWall(int(nextX), int(nextY)) {
			g.autoWalkFrameCount--
			return g.autoWalkDirX * stepCells, g.autoWalkDirY * stepCells
		}
		g.autoWalkFrameCount = 0
	}
	return 0, 0
}

// randomizeAutoWalkDirection chooses a new heading for automatic walking.
func (g *Game) randomizeAutoWalkDirection() {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 5))
	}
	angle := g.autoWalkRand.Float64() * 2 * math.Pi
	g.autoWalkDirX = math.Cos(angle)
	g.autoWalkDirY = math.Sin(angle)
	g.autoWalkFrameCount = 20 + g.autoWalkRand.Intn(50)
}

// handleDebugControls processes debug overlay hotkeys.
func (g *Game) handleDebugControls() {
	if !*debugFlag {
		return
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyMinus) || inpututil.IsKeyJustPressed(ebiten.KeyKPSubtract) {
		g.adjustSimMultiplier(-simMultiplierStep)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEqual) || inpututil.IsKeyJustPressed(ebiten.KeyKPAdd) {
		g.adjustSimMultiplier(simMultiplierStep)
	}
}

// adjustSimMultiplier clamps the simulation batch size delta within bounds.
func (g *Game) adjustSimMultiplier(delta int) {
	g.simStepMultiplier += delta
	if g.simStepMultiplier < minSimMultiplier {
		g.simStepMultiplier = minSimMultiplier
	} else if g.simStepMultiplier > maxSimMultiplier {
		g.simStepMultiplier = maxSimMultiplier
	}
}

// simStepsPerSecond returns the nominal simulation steps executed each second.
func (g *Game) simStepsPerSecond() float64 {
	dt := g.tickSeconds()
	if dt <= 0 {
		return 0
	}
	return (1.0 / dt) * float64(g.simStepMultiplier)
}
