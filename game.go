package main

import (
	"log"
	"math"
	"math/rand"
	"time"
)

// Game encapsulates the full simulation state, rendering buffers, and audio pipeline.
type Game struct {
	field *waveField

	ex float64
	ey float64

	stepTimer         int
	lastSimDuration   time.Duration
	simStepMultiplier int

	walls     []bool
	levelRand *rand.Rand
	levelData *levelDef

	listenerForwardX float64
	listenerForwardY float64

	autoWalk           bool
	autoWalkDeadline   time.Time
	autoWalkRand       *rand.Rand
	autoWalkDirX       float64
	autoWalkDirY       float64
	autoWalkFrameCount int

	visibleStamp []uint32
	visibleGen   uint32
	lastVisCX    int
	lastVisCY    int

	gpuSolver      *openCLWaveSolver
	impulsesActive bool
	wallsDirty     bool

	viewX, viewY int

	blockMask            []byte
	blockMaskDirty       bool
	blockMaskNeedsUpload bool
	viewPixels           []byte

	blockCols        int
	blockRows        int
	blockActive      []bool
	wallShiftScratch []bool

	worldOriginX int
	worldOriginY int
	renderState
}

// newGame constructs a fully initialized Game instance.
func newGame(level *levelDef) *Game {
	startX := float64(w / 2)
	startY := float64(h / 2)
	if level != nil {
		startX, startY = level.startPosition(w, h)
	}
	g := &Game{
		field:                newWaveField(w, h),
		ex:                   startX,
		ey:                   startY,
		levelRand:            rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:                make([]bool, w*h),
		levelData:            level,
		listenerForwardX:     0,
		listenerForwardY:     -1,
		autoWalkRand:         rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
		simStepMultiplier:    defaultSimMultiplier,
		blockMaskDirty:       true,
		blockMaskNeedsUpload: true,
		worldOriginX:         0,
		worldOriginY:         0,
	}
	// Audio removed
	if solver, err := newOpenCLWaveSolver(w, h); err != nil {
		log.Fatalf("OpenCL initialization failed: %v", err)
	} else {
		log.Printf("OpenCL solver enabled (device: %s)", solver.DeviceName())
		g.gpuSolver = solver
	}
	g.resetField()
	g.generateWalls()
	g.updateCamera()
	g.lastVisCX, g.lastVisCY = -1, -1
	return g
}

// Update advances the simulation, produces optional audio, and refreshes visibility data.
func (g *Game) Update() error {
	dx, dy := g.movementVector()
	oldX, oldY := g.ex, g.ey
	g.ex += dx
	g.ey += dy
	if g.isWall(int(g.ex), int(g.ey)) {
		g.ex, g.ey = oldX, oldY
	}

	g.handleDebugControls()
	g.updateCamera()
	if g.blockMaskDirty {
		g.rebuildBlockMask()
	}

	moving := dx != 0 || dy != 0
	impulsesFired := false
	if moving {
		length := math.Hypot(dx, dy)
		if length > 0 {
			g.listenerForwardX = dx / length
			g.listenerForwardY = dy / length
		}
		g.stepTimer++
		if g.stepTimer >= stepDelay {
			g.stepTimer = 0
			baseX := int(g.ex)
			baseY := int(g.ey)
			for _, offset := range emitterFootprint {
				cx := baseX + offset.dx
				cy := baseY + offset.dy
				localX := cx - g.worldOriginX
				localY := cy - g.worldOriginY
				if localX <= 0 || localX >= w-1 || localY <= 0 || localY >= h-1 {
					continue
				}
				if g.isWall(cx, cy) {
					continue
				}
				if g.field.queueImpulse(localX, localY, stepImpulseStrength) {
					impulsesFired = true
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

	g.impulsesActive = impulsesFired

	if *occludeLineOfSightFlag {
		g.refreshVisibleMask()
	}

	steps := g.simStepMultiplier
	simStart := time.Now()
	var visibleStamp []uint32
	var visibleGen uint32
	if *occludeLineOfSightFlag {
		visibleStamp = g.visibleStamp
		visibleGen = g.visibleGen
	}
	maskDirty := g.blockMaskNeedsUpload
	if err := g.gpuSolver.Step(g.field, g.walls, steps, g.wallsDirty, false, *occludeLineOfSightFlag, visibleStamp, visibleGen, g.blockMask, maskDirty); err != nil {
		return err
	}
	if maskDirty {
		g.blockMaskNeedsUpload = false
	}
	g.wallsDirty = false
	g.lastSimDuration = time.Since(simStart)

	return nil
}

func (g *Game) updateCamera() {
	prevX, prevY := g.viewX, g.viewY
	localX, localY := g.localListener()
	maxViewX := w - viewportWidth
	if maxViewX < 0 {
		maxViewX = 0
	}
	maxViewY := h - viewportHeight
	if maxViewY < 0 {
		maxViewY = 0
	}
	xMin, xMax := viewportScrollBounds(viewportWidth, w)
	yMin, yMax := viewportScrollBounds(viewportHeight, h)
	dx := g.scrollOffset(localX, xMin, xMax)
	dy := g.scrollOffset(localY, yMin, yMax)
	if dx != 0 || dy != 0 {
		g.shiftWindow(dx, dy)
		localX, localY = g.localListener()
	}
	g.viewX = clampCoord(localX-viewportWidth/2, 0, maxViewX)
	g.viewY = clampCoord(localY-viewportHeight/2, 0, maxViewY)
	if g.viewX != prevX || g.viewY != prevY {
		g.blockMaskDirty = true
	}
}

func (g *Game) rebuildBlockMask() {
	if viewportWidth <= 0 || viewportHeight <= 0 || blockWidth <= 0 || blockHeight <= 0 {
		g.blockMask = nil
		g.blockMaskNeedsUpload = false
		g.blockMaskDirty = false
		g.blockCols = 0
		g.blockRows = 0
		g.blockActive = nil
		return
	}
	cols, rows := g.ensureBlockGeometry()
	size := w * h
	if len(g.blockMask) != size {
		g.blockMask = make([]byte, size)
	}
	for i := range g.blockMask {
		g.blockMask[i] = 0
	}
	startBlockX := g.viewX / blockWidth
	endBlockX := (g.viewX + viewportWidth - 1) / blockWidth
	startBlockY := g.viewY / blockHeight
	endBlockY := (g.viewY + viewportHeight - 1) / blockHeight
	maxBlockX := cols - 1
	maxBlockY := rows - 1
	startBlockX = intMax(0, startBlockX)
	startBlockY = intMax(0, startBlockY)
	endBlockX = intMin(maxBlockX, endBlockX)
	endBlockY = intMin(maxBlockY, endBlockY)
	for by := startBlockY; by <= endBlockY; by++ {
		y0 := by * blockHeight
		y1 := y0 + blockHeight
		if y1 > h {
			y1 = h
		}
		for bx := startBlockX; bx <= endBlockX; bx++ {
			if bx < 0 || bx > maxBlockX {
				continue
			}
			if by < 0 || by > maxBlockY {
				continue
			}
			x0 := bx * blockWidth
			x1 := x0 + blockWidth
			if x1 > w {
				x1 = w
			}
			blockIdx := by*cols + bx
			if blockIdx >= 0 && blockIdx < len(g.blockActive) {
				g.blockActive[blockIdx] = true
			}
			for y := y0; y < y1; y++ {
				base := y * w
				for x := x0; x < x1; x++ {
					g.blockMask[base+x] = 1
				}
			}
		}
	}
	g.blockMaskNeedsUpload = true
	g.blockMaskDirty = false
}

func intMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func intMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (g *Game) shiftWindow(dx, dy int) {
	if dx == 0 && dy == 0 {
		return
	}
	g.field.shift(dx, dy)
	g.shiftWalls(dx, dy)
	g.worldOriginX += dx
	g.worldOriginY += dy
	g.visibleStamp = nil
	g.visibleGen = 0
	g.lastVisCX = -1
	g.lastVisCY = -1
	g.wallsDirty = true
}

func (g *Game) shiftWalls(dx, dy int) {
	if dx == 0 && dy == 0 || len(g.walls) == 0 {
		return
	}
	total := len(g.walls)
	if len(g.wallShiftScratch) != total {
		g.wallShiftScratch = make([]bool, total)
	}
	copy(g.wallShiftScratch, g.walls)
	for y := 0; y < h; y++ {
		srcY := y + dy
		rowStart := y * w
		for x := 0; x < w; x++ {
			srcX := x + dx
			if srcX < 0 || srcX >= w || srcY < 0 || srcY >= h {
				g.walls[rowStart+x] = false
				continue
			}
			g.walls[rowStart+x] = g.wallShiftScratch[srcY*w+srcX]
		}
	}
}

func (g *Game) resetField() {
	g.field.reset()
	g.impulsesActive = false
	if len(g.walls) != w*h {
		g.walls = make([]bool, w*h)
	} else {
		for i := range g.walls {
			g.walls[i] = false
		}
	}
	g.visibleStamp = nil
	g.visibleGen = 0
	g.wallsDirty = true
	g.blockMaskDirty = true
	g.blockMaskNeedsUpload = true
}

func (g *Game) ensureBlockGeometry() (int, int) {
	if blockWidth <= 0 || blockHeight <= 0 {
		g.blockActive = nil
		g.blockCols = 0
		g.blockRows = 0
		return 0, 0
	}
	cols := (w + blockWidth - 1) / blockWidth
	rows := (h + blockHeight - 1) / blockHeight
	if cols <= 0 {
		cols = 1
	}
	if rows <= 0 {
		rows = 1
	}
	total := cols * rows
	if len(g.blockActive) != total {
		g.blockActive = make([]bool, total)
	} else {
		for i := range g.blockActive {
			g.blockActive[i] = false
		}
	}
	g.blockCols = cols
	g.blockRows = rows
	return cols, rows
}

func (g *Game) localListener() (int, int) {
	localX := int(math.Round(g.ex)) - g.worldOriginX
	localY := int(math.Round(g.ey)) - g.worldOriginY
	return localX, localY
}

func (g *Game) scrollOffset(coord, minIncl, maxIncl int) int {
	if minIncl > maxIncl {
		return 0
	}
	if coord < minIncl {
		return coord - minIncl
	}
	if coord > maxIncl {
		return coord - maxIncl
	}
	return 0
}

func viewportScrollBounds(viewport, world int) (int, int) {
	if viewport <= 0 || world <= 0 {
		return 0, 0
	}
	half := viewport / 2
	min := emitterRad
	if half > min {
		min = half
	}
	max := world - emitterRad - 1
	rightBound := world - viewport + half
	if rightBound < max {
		max = rightBound
	}
	worldMax := world - 1
	if worldMax < 0 {
		worldMax = 0
	}
	max = clampCoord(max, 0, worldMax)
	min = clampCoord(min, 0, worldMax)
	if max < min {
		min = max
	}
	return min, max
}
