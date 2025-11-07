package main

import (
	"flag"
	"image/color"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	w, h                  = 512, 512
	windowScale           = 2
	damp                  = 0.995
	speed                 = 0.5
	waveDamp32            = float32(damp)
	waveSpeed32           = float32(speed)
	emitterRad            = 3
	moveSpeed             = 2
	stepDelay             = 15
	defaultTPS            = 60.0
	simStepsPerSecond     = defaultTPS * 4
	earOffsetCells        = 5
	boundaryReflect       = 0.99
	stepImpulseStrength   = 10
	wallSegments          = 50
	wallMinLen            = 12
	wallMaxLen            = 42
	wallExclusionRadius   = 12
	wallThicknessVariance = 2
)

var showWallsFlag = flag.Bool("show-walls", false, "render wall geometry overlays")

type half uint16

func float32ToHalf(v float32) half {
	bits := math.Float32bits(v)
	sign := (bits >> 16) & 0x8000
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	switch {
	case exp <= 0:
		if exp < -10 {
			return half(sign)
		}
		mant |= 0x800000
		shift := uint32(1 - exp)
		mant16 := mant >> (shift + 13)
		if (mant>>(shift+12))&1 == 1 {
			mant16++
		}
		return half(sign | mant16)
	case exp >= 0x1F:
		return half(sign | 0x7C00)
	default:
		mant16 := mant >> 13
		if mant&0x1FFF > 0x1000 || (mant&0x1FFF == 0x1000 && mant16&1 == 1) {
			mant16++
			if mant16 == 0x0400 {
				mant16 = 0
				exp++
				if exp == 0x1F {
					return half(sign | 0x7C00)
				}
			}
		}
		return half(sign | uint32(exp)<<10 | mant16)
	}
}

func halfToFloat32(h half) float32 {
	bits := uint16(h)
	sign := uint32(bits&0x8000) << 16
	exp := (bits >> 10) & 0x1F
	mant := uint32(bits & 0x3FF)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3FF
		fbits := sign | ((uint32(exp - 15 + 127)) << 23) | (mant << 13)
		return math.Float32frombits(fbits)
	case 0x1F:
		fbits := sign | 0x7F800000 | (mant << 13)
		return math.Float32frombits(fbits)
	default:
		fbits := sign | ((uint32(exp - 15 + 127)) << 23) | (mant << 13)
		return math.Float32frombits(fbits)
	}
}

type wavePlane [][]half

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
		p[y] = make([]half, width)
	}
	return p
}

func (f *waveField) setCurr(x, y int, value float32) {
	f.curr[y][x] = float32ToHalf(value)
}

func (f *waveField) zeroCell(x, y int) {
	f.curr[y][x] = 0
	f.prev[y][x] = 0
	f.next[y][x] = 0
}

func (f *waveField) readCurr(x, y int) float32 {
	return halfToFloat32(f.curr[y][x])
}

func (f *waveField) swap() {
	f.prev, f.curr, f.next = f.curr, f.next, f.prev
}

func (f *waveField) zeroBoundaries() {
	lastRow := f.height - 1
	lastCol := f.width - 1
	reflect := float32(boundaryReflect)
	for x := 0; x < f.width; x++ {
		top := halfToFloat32(f.next[1][x])
		bottom := halfToFloat32(f.next[lastRow-1][x])
		f.next[0][x] = float32ToHalf(-top * reflect)
		f.next[lastRow][x] = float32ToHalf(-bottom * reflect)
	}
	for y := 1; y < lastRow; y++ {
		left := halfToFloat32(f.next[y][1])
		right := halfToFloat32(f.next[y][lastCol-1])
		f.next[y][0] = float32ToHalf(-left * reflect)
		f.next[y][lastCol] = float32ToHalf(-right * reflect)
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
			nextRow[x] = float32ToHalf(val)
		}
	}
}

func convertRow(src []half, dst []float32) {
	for i := range src {
		dst[i] = halfToFloat32(src[i])
	}
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

func (g *Game) Update() error {
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
	for i := 0; i < w*h; i++ {
		base := i * 4
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

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())
	g := newGame()

	ebiten.SetWindowSize(w*windowScale, h*windowScale)
	ebiten.SetWindowTitle("Acoustic Steps")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
