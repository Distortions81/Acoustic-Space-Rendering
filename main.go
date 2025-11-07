package main

import (
	"flag"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	w, h                  = 1024, 1024
	windowScale           = 1
	damp                  = 0.996
	speed                 = 0.5
	waveDamp32            = float32(damp)
	waveSpeed32           = float32(speed)
	emitterRad            = 3
	moveSpeed             = 2
	stepDelay             = 15
	defaultTPS            = 60.0
	simStepsPerSecond     = defaultTPS * 6
	earOffsetCells        = 5
	boundaryReflect       = 0.60
	stepImpulseStrength   = 10
	wallSegments          = 50
	wallMinLen            = 12
	wallMaxLen            = 100
	wallExclusionRadius   = 1
	wallThicknessVariance = 2
	pgoRecordDuration     = 15 * time.Second
)

var showWallsFlag = flag.Bool("show-walls", false, "render wall geometry overlays")
var recordDefaultPGO = flag.Bool("record-default-pgo", false, "walk randomly for 15s while capturing default.pgo")
var occludeLineOfSightFlag = flag.Bool("occlude-line-of-sight", true, "hide regions that are not in the listener's line of sight when rendering")

type intPoint struct {
	x int
	y int
}

var losPerimeterTargets = buildLOSPerimeterTargets()

func buildLOSPerimeterTargets() []intPoint {
	points := make([]intPoint, 0, 2*(w+h))
	for x := 0; x < w; x++ {
		points = append(points, intPoint{x: x, y: 0})
		points = append(points, intPoint{x: x, y: h - 1})
	}
	for y := 1; y < h-1; y++ {
		points = append(points, intPoint{x: 0, y: y})
		points = append(points, intPoint{x: w - 1, y: y})
	}
	return points
}

type wavePlane [][]float32

type waveField struct {
	width, height int
	curr          wavePlane
	prev          wavePlane
	next          wavePlane
}

func newWaveField(width, height int) *waveField {
	return &waveField{
		width: width, height: height,
		curr: makePlane(width, height),
		prev: makePlane(width, height),
		next: makePlane(width, height),
	}
}

func makePlane(width, height int) wavePlane {
	p := make(wavePlane, height)
	for y := range p {
		p[y] = make([]float32, width)
	}
	return p
}

func (f *waveField) setCurr(x, y int, value float32) {
	f.curr[y][x] = value
}

func (f *waveField) zeroCell(x, y int) {
	f.curr[y][x] = 0
	f.prev[y][x] = 0
	f.next[y][x] = 0
}

func (f *waveField) readCurr(x, y int) float32 {
	return f.curr[y][x]
}

func (f *waveField) swap() {
	f.prev, f.curr, f.next = f.curr, f.next, f.prev
}

func (f *waveField) zeroBoundaries() {
	lastRow := f.height - 1
	lastCol := f.width - 1
	reflect := float32(boundaryReflect)
	for x := 0; x < f.width; x++ {
		top := f.next[1][x]
		bottom := f.next[lastRow-1][x]
		f.next[0][x] = -top * reflect
		f.next[lastRow][x] = -bottom * reflect
	}
	for y := 1; y < lastRow; y++ {
		left := f.next[y][1]
		right := f.next[y][lastCol-1]
		f.next[y][0] = -left * reflect
		f.next[y][lastCol] = -right * reflect
	}
}

type rowMask struct {
	y  int
	xs []int
}

type workerMask struct {
	rows []rowMask
}

type rowCache struct {
	center []float32
	prev   []float32
	top    []float32
	bottom []float32
}

func newRowCache(width int) *rowCache {
	return &rowCache{
		center: make([]float32, width),
		prev:   make([]float32, width),
		top:    make([]float32, width),
		bottom: make([]float32, width),
	}
}

func (g *Game) waveWorkerLoop(index int) {
	cache := newRowCache(g.field.width)
	lastStep := 0
	g.workerMu.Lock()
	for {
		for g.workerStep == lastStep {
			g.workerCond.Wait()
		}
		lastStep = g.workerStep
		var mask workerMask
		if index < len(g.workerMasks) {
			mask = g.workerMasks[index]
		}
		g.workerMu.Unlock()

		if len(mask.rows) > 0 {
			processMask(g.field, &mask, cache)
		}

		g.workerMu.Lock()
		g.workerPending--
		if g.workerPending == 0 {
			g.workerCond.Broadcast()
		}
	}
}

func processMask(field *waveField, mask *workerMask, cache *rowCache) {
	for _, row := range mask.rows {
		y := row.y
		convertRow(field.curr[y], cache.center)
		convertRow(field.prev[y], cache.prev)
		convertRow(field.curr[y-1], cache.top)
		convertRow(field.curr[y+1], cache.bottom)

		nextRow := field.next[y]
		nextRow[0] = 0
		nextRow[len(nextRow)-1] = 0
		for _, x := range row.xs {
			lap := cache.center[x-1] + cache.center[x+1] + cache.top[x] + cache.bottom[x] - 4*cache.center[x]
			val := (2*cache.center[x] - cache.prev[x]) + waveSpeed32*lap
			val *= waveDamp32
			nextRow[x] = val
		}
	}
}

func convertRow(src []float32, dst []float32) {
	copy(dst, src)
}

func assignRowMasks(workerCount int, rows []rowMask) []workerMask {
	if workerCount < 1 {
		workerCount = 1
	}
	masks := make([]workerMask, workerCount)
	for idx, row := range rows {
		workerIdx := idx % workerCount
		masks[workerIdx].rows = append(masks[workerIdx].rows, row)
	}
	return masks
}

type Game struct {
	field              *waveField
	ex, ey             float64
	stepTimer          int
	physicsAccumulator float64
	walls              []bool
	levelRand          *rand.Rand
	workerCount        int
	workerMasks        []workerMask
	maskDirty          bool
	workerMu           sync.Mutex
	workerCond         *sync.Cond
	workerStep         int
	workerPending      int
	listenerForwardX   float64
	listenerForwardY   float64
	pixelBuf           []byte
	autoWalk           bool
	autoWalkDeadline   time.Time
	autoWalkRand       *rand.Rand
	autoWalkDirX       float64
	autoWalkDirY       float64
	autoWalkFrameCount int
	visibleMask        []bool
}

func newGame() *Game {
	workerCount := runtime.NumCPU()
	if workerCount < 1 {
		workerCount = 1
	}
	g := &Game{
		field:            newWaveField(w, h),
		ex:               float64(w / 2),
		ey:               float64(h / 2),
		levelRand:        rand.New(rand.NewSource(time.Now().UnixNano() + 1)),
		walls:            make([]bool, w*h),
		workerCount:      workerCount,
		maskDirty:        true,
		listenerForwardX: 0,
		listenerForwardY: -1,
		pixelBuf:         make([]byte, w*h*4),
		autoWalkRand:     rand.New(rand.NewSource(time.Now().UnixNano() + 2)),
	}
	g.workerCond = sync.NewCond(&g.workerMu)
	for i := 0; i < workerCount; i++ {
		go g.waveWorkerLoop(i)
	}
	g.generateWalls()
	g.rebuildInteriorMask()
	return g
}

func (g *Game) rebuildInteriorMask() {
	if g.workerCount < 1 {
		g.workerCount = 1
	}
	rows := make([]rowMask, 0, h-2)
	for y := 1; y < h-1; y++ {
		base := y * w
		xs := make([]int, 0, w-2)
		for x := 1; x < w-1; x++ {
			if g.walls[base+x] {
				continue
			}
			xs = append(xs, x)
		}
		if len(xs) == 0 {
			continue
		}
		rows = append(rows, rowMask{y: y, xs: xs})
	}
	g.workerMasks = assignRowMasks(g.workerCount, rows)
	g.maskDirty = false
}

func (g *Game) ensureInteriorMask() {
	if g.maskDirty || len(g.workerMasks) != g.workerCount {
		g.rebuildInteriorMask()
	}
}

func (g *Game) enableAutoWalk(duration time.Duration) {
	g.autoWalk = true
	g.autoWalkDeadline = time.Now().Add(duration)
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 3))
	}
	g.autoWalkFrameCount = 0
}

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

func (g *Game) manualMovementVector() (float64, float64) {
	dx, dy := 0.0, 0.0
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		dy -= moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) {
		dy += moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) {
		dx -= moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) {
		dx += moveSpeed
	}
	if dx != 0 && dy != 0 {
		dx *= 0.7071
		dy *= 0.7071
	}
	return dx, dy
}

func (g *Game) autoWalkVector() (float64, float64) {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 4))
	}
	for attempts := 0; attempts < 5; attempts++ {
		if g.autoWalkFrameCount <= 0 {
			g.randomizeAutoWalkDirection()
		}
		nextX := g.ex + g.autoWalkDirX*moveSpeed
		nextY := g.ey + g.autoWalkDirY*moveSpeed
		if nextX > float64(emitterRad) && nextX < float64(w-emitterRad-1) &&
			nextY > float64(emitterRad) && nextY < float64(h-emitterRad-1) &&
			!g.isWall(int(nextX), int(nextY)) {
			g.autoWalkFrameCount--
			return g.autoWalkDirX * moveSpeed, g.autoWalkDirY * moveSpeed
		}
		g.autoWalkFrameCount = 0
	}
	return 0, 0
}

func (g *Game) randomizeAutoWalkDirection() {
	if g.autoWalkRand == nil {
		g.autoWalkRand = rand.New(rand.NewSource(time.Now().UnixNano() + 5))
	}
	angle := g.autoWalkRand.Float64() * 2 * math.Pi
	g.autoWalkDirX = math.Cos(angle)
	g.autoWalkDirY = math.Sin(angle)
	g.autoWalkFrameCount = 20 + g.autoWalkRand.Intn(50)
}

func (g *Game) Update() error {
	dx, dy := g.movementVector()
	oldX, oldY := g.ex, g.ey
	g.ex = math.Max(emitterRad, math.Min(float64(w-emitterRad-1), g.ex+dx))
	g.ey = math.Max(emitterRad, math.Min(float64(h-emitterRad-1), g.ey+dy))
	if g.isWall(int(g.ex), int(g.ey)) {
		g.ex, g.ey = oldX, oldY
	}

	moving := dx != 0 || dy != 0
	if moving {
		length := math.Hypot(dx, dy)
		if length > 0 {
			g.listenerForwardX = dx / length
			g.listenerForwardY = dy / length
		}
		g.stepTimer++
		if g.stepTimer >= stepDelay {
			g.stepTimer = 0
			for y := -emitterRad; y <= emitterRad; y++ {
				for x := -emitterRad; x <= emitterRad; x++ {
					if x*x+y*y <= emitterRad*emitterRad {
						cx := int(g.ex) + x
						cy := int(g.ey) + y
						if cx <= 0 || cx >= w-1 || cy <= 0 || cy >= h-1 {
							continue
						}
						if g.isWall(cx, cy) {
							continue
						}
						g.field.setCurr(cx, cy, stepImpulseStrength)
					}
				}
			}
		}
	} else {
		g.stepTimer = stepDelay
	}

	actualTPS := ebiten.ActualTPS()
	if actualTPS < 1 {
		actualTPS = defaultTPS
	}
	g.physicsAccumulator += simStepsPerSecond / actualTPS
	steps := int(g.physicsAccumulator)
	if steps < 1 {
		steps = 1
	}
	for i := 0; i < steps; i++ {
		g.stepWave()
	}
	g.physicsAccumulator -= float64(steps)

	return nil
}

func (g *Game) generateWalls() {
	if len(g.walls) != w*h {
		g.walls = make([]bool, w*h)
	} else {
		for i := range g.walls {
			g.walls[i] = false
		}
	}
	if g.levelRand == nil {
		g.levelRand = rand.New(rand.NewSource(time.Now().UnixNano() + 1))
	}
	for s := 0; s < wallSegments; s++ {
		lengthRange := wallMaxLen - wallMinLen + 1
		if lengthRange <= 0 {
			lengthRange = 1
		}
		length := wallMinLen + g.levelRand.Intn(lengthRange)
		thickness := 1
		if wallThicknessVariance > 0 {
			thickness += g.levelRand.Intn(wallThicknessVariance + 1)
		}
		horizontal := g.levelRand.Intn(2) == 0
		x := g.levelRand.Intn(w-4) + 2
		y := g.levelRand.Intn(h-4) + 2
		dx, dy := 0, 1
		if horizontal {
			dx, dy = 1, 0
		}
		perpX, perpY := dy, dx
		cx, cy := x, y
		for l := 0; l < length; l++ {
			if cx <= 1 || cx >= w-1 || cy <= 1 || cy >= h-1 {
				break
			}
			for t := -thickness; t <= thickness; t++ {
				tx := cx + perpX*t
				ty := cy + perpY*t
				g.trySetWall(tx, ty)
			}
			cx += dx
			cy += dy
		}
	}
	g.maskDirty = true
}

func (g *Game) trySetWall(x, y int) {
	if x <= 1 || x >= w-1 || y <= 1 || y >= h-1 {
		return
	}
	dx := float64(x) - g.ex
	dy := float64(y) - g.ey
	if dx*dx+dy*dy < float64(wallExclusionRadius*wallExclusionRadius) {
		return
	}
	idx := y*w + x
	g.walls[idx] = true
	g.field.zeroCell(x, y)
	g.maskDirty = true
}

func (g *Game) isWall(x, y int) bool {
	if x < 0 || x >= w || y < 0 || y >= h {
		return true
	}
	if len(g.walls) == 0 {
		return false
	}
	return g.walls[y*w+x]
}

func (g *Game) earOffsets() (int, int) {
	fx, fy := g.listenerForwardX, g.listenerForwardY
	if fx == 0 && fy == 0 {
		fy = -1
	}
	earVecX := -fy
	earVecY := fx
	length := math.Hypot(earVecX, earVecY)
	if length == 0 {
		return earOffsetCells, 0
	}
	scale := float64(earOffsetCells) / length
	ox := int(math.Round(earVecX * scale))
	oy := int(math.Round(earVecY * scale))
	if ox == 0 && oy == 0 {
		if math.Abs(earVecX) >= math.Abs(earVecY) {
			if earVecX >= 0 {
				ox = earOffsetCells
			} else {
				ox = -earOffsetCells
			}
		} else {
			if earVecY >= 0 {
				oy = earOffsetCells
			} else {
				oy = -earOffsetCells
			}
		}
	}
	return ox, oy
}

func clampCoord(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func (g *Game) refreshVisibleMask() {
	if len(g.visibleMask) != w*h {
		g.visibleMask = make([]bool, w*h)
	}
	for i := range g.visibleMask {
		g.visibleMask[i] = false
	}
	cx := clampCoord(int(math.Round(g.ex)), 0, w-1)
	cy := clampCoord(int(math.Round(g.ey)), 0, h-1)
	g.visibleMask[cy*w+cx] = true
	for _, target := range losPerimeterTargets {
		g.castVisibilityRay(cx, cy, target.x, target.y)
	}
}

func (g *Game) castVisibilityRay(x0, y0, x1, y1 int) {
	dx := int(math.Abs(float64(x1 - x0)))
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -int(math.Abs(float64(y1 - y0)))
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		if x0 < 0 || x0 >= w || y0 < 0 || y0 >= h {
			break
		}
		idx := y0*w + x0
		g.visibleMask[idx] = true
		if g.isWall(x0, y0) && !(x0 == x1 && y0 == y1) {
			break
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func (g *Game) stepWave() {
	g.ensureInteriorMask()
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

func (g *Game) Draw(screen *ebiten.Image) {
	if len(g.pixelBuf) != w*h*4 {
		g.pixelBuf = make([]byte, w*h*4)
	}
	img := g.pixelBuf
	showWalls := *showWallsFlag
	occludeLOS := *occludeLineOfSightFlag
	if occludeLOS {
		g.refreshVisibleMask()
	}
	for i := 0; i < w*h; i++ {
		base := i * 4
		if occludeLOS && (len(g.visibleMask) == w*h) && !g.visibleMask[i] {
			img[base] = 0
			img[base+1] = 0
			img[base+2] = 0
			img[base+3] = 255
			continue
		}
		if showWalls && len(g.walls) > 0 && g.walls[i] {
			img[base] = 30
			img[base+1] = 40
			img[base+2] = 80
			img[base+3] = 255
			continue
		}
		x := i % w
		y := i / w
		v := g.field.readCurr(x, y)
		v = float32(math.Max(-1, math.Min(1, float64(v))))
		intensity := byte(math.Abs(float64(v)) * 255)
		img[base] = intensity
		img[base+1] = intensity
		img[base+2] = intensity
		img[base+3] = 255
	}
	screen.WritePixels(img)

	for y := -emitterRad; y <= emitterRad; y++ {
		for x := -emitterRad; x <= emitterRad; x++ {
			cx := int(g.ex) + x
			cy := int(g.ey) + y
			if cx >= 0 && cx < w && cy >= 0 && cy < h {
				screen.Set(cx, cy, color.RGBA{255, 0, 0, 255})
			}
		}
	}
	g.drawEarIndicators(screen, int(g.ex), int(g.ey))
}

func (g *Game) Layout(_, _ int) (int, int) { return w, h }

func (g *Game) drawEarIndicators(screen *ebiten.Image, cx, cy int) {
	ox, oy := g.earOffsets()
	leftX := clampCoord(cx-ox, 0, w-1)
	leftY := clampCoord(cy-oy, 0, h-1)
	rightX := clampCoord(cx+ox, 0, w-1)
	rightY := clampCoord(cy+oy, 0, h-1)
	drawLine(screen, cx, cy, leftX, leftY, color.RGBA{0, 255, 200, 200})
	drawLine(screen, cx, cy, rightX, rightY, color.RGBA{0, 200, 255, 200})
	if leftX >= 0 && leftX < w && leftY >= 0 && leftY < h {
		screen.Set(leftX, leftY, color.RGBA{0, 255, 200, 255})
	}
	if rightX >= 0 && rightX < w && rightY >= 0 && rightY < h {
		screen.Set(rightX, rightY, color.RGBA{0, 200, 255, 255})
	}
}

func drawLine(screen *ebiten.Image, x0, y0, x1, y1 int, clr color.Color) {
	dx := int(math.Abs(float64(x1 - x0)))
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -int(math.Abs(float64(y1 - y0)))
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
			screen.Set(x0, y0, clr)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func startDefaultPGORecording(path string) (func(), error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return nil, err
	}
	var once sync.Once
	stop := func() {
		once.Do(func() {
			pprof.StopCPUProfile()
			_ = f.Close()
		})
	}
	return stop, nil
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())

	var stopProfile func()
	if *recordDefaultPGO {
		var err error
		stopProfile, err = startDefaultPGORecording("default.pgo")
		if err != nil {
			log.Fatalf("unable to start PGO recording: %v", err)
		}
		defer stopProfile()
	}

	g := newGame()
	if *recordDefaultPGO {
		g.enableAutoWalk(pgoRecordDuration)
		go func(stop func()) {
			timer := time.NewTimer(pgoRecordDuration)
			<-timer.C
			stop()
			log.Printf("default.pgo captured after %s; exiting", pgoRecordDuration)
			os.Exit(0)
		}(stopProfile)
	}

	ebiten.SetWindowSize(w*windowScale, h*windowScale)
	ebiten.SetWindowTitle("Acoustic Steps")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
