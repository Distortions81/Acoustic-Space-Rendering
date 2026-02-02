package main

import (
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
)

type openCLWaveSolver struct {
	context                  *cl.Context
	queue                    *cl.CommandQueue
	program                  *cl.Program
	kernel                   *cl.Kernel
	renderKernel             *cl.Kernel
	currBuf                  *cl.MemObject
	prevBuf                  *cl.MemObject
	nextBuf                  *cl.MemObject
	pixelBuf                 *cl.MemObject
	earSampleBuf             *cl.MemObject
	wallMaskBuf              *cl.MemObject
	width                    int
	height                   int
	useFP16                  bool
	elementBytes             int
	wallMaskSynced           bool
	deviceName               string
	device                   *cl.Device
	coldStart                bool
	waveGlobal               []int
	waveLocal                []int
	boundCurr                *cl.MemObject
	boundPrev                *cl.MemObject
	boundNext                *cl.MemObject
	hostPixels               []byte
	hostWallMask             []byte
	hostEarSamples           []float32
	hostEarSamplesHalf       []uint16
	pixelMu                  sync.Mutex
	pixelEvent               *cl.Event
	lastRenderShowWalls      int32
	hostCurrHalf             []uint16
	hostPrevHalf             []uint16
	hostNextHalf             []uint16
	earLeftSample            float32
	earRightSample           float32
	lastSampleCount          int
	lastDamp                 float32
	lastSpeed                float32
	lastWorldBoundaryReflect float32
	lastRoomWallReflect      float32
	warnedCoeffClamp         bool
}

type audioEmitterData struct {
	index   int32
	samples []float32
}

const waveKernelSource = `#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half real_t;
inline real_t to_real(float v) { return convert_half(v); }
inline float to_float(real_t v) { return convert_float(v); }
#else
typedef float real_t;
inline real_t to_real(float v) { return v; }
inline float to_float(real_t v) { return v; }
#endif

__kernel void wave_step(
    const int width,
    const int height,
    const float damp,
    const float speed,
    const float world_boundary_reflect,
    const float room_wall_reflect,
    __global const real_t* curr,
    __global const real_t* prev,
    __global real_t* next_buffer,
    __global const uchar* wall_mask,
    __global real_t* ear_samples,
    const int ear_sample_size,
    const int left_index,
    const int right_index,
    const int step_index,
    const int emitter_index,
    const real_t emitter_value)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;
    if (wall_mask[idx]) {
        next_buffer[idx] = (real_t)0.0f;
        return;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        next_buffer[idx] = (real_t)0.0f;
        return;
    }
    int left = idx - 1;
    int right = idx + 1;
    int top = idx - width;
    int bottom = idx + width;
    const real_t damp_r = to_real(damp);
    const real_t speed_r = to_real(speed);
    const real_t world_reflect_r = to_real(world_boundary_reflect);
    const real_t wall_reflect_r = to_real(room_wall_reflect);
    const real_t two = to_real(2.0f);
    const real_t four = to_real(4.0f);
    real_t center = curr[idx];
    real_t nL = curr[left];
    real_t nR = curr[right];
    real_t nT = curr[top];
    real_t nB = curr[bottom];
    if (x == 1) {
        nL = center * world_reflect_r;
    } else if (wall_mask[left]) {
        nL = center * wall_reflect_r;
    }
    if (x == width - 2) {
        nR = center * world_reflect_r;
    } else if (wall_mask[right]) {
        nR = center * wall_reflect_r;
    }
    if (y == 1) {
        nT = center * world_reflect_r;
    } else if (wall_mask[top]) {
        nT = center * wall_reflect_r;
    }
    if (y == height - 2) {
        nB = center * world_reflect_r;
    } else if (wall_mask[bottom]) {
        nB = center * wall_reflect_r;
    }
    real_t laplacian = nL + nR + nT + nB - four * center;
    real_t next_val = ((two * center - prev[idx]) + speed_r * laplacian) * damp_r;
    if (emitter_index >= 0) {
        int ex = emitter_index % width;
        int ey = emitter_index / width;
        int dx = x - ex;
        int dy = y - ey;
        int adx = abs(dx);
        int ady = abs(dy);
        if (adx <= 1 && ady <= 1) {
            int wx = (adx == 0) ? 2 : 1;
            int wy = (ady == 0) ? 2 : 1;
            real_t weight = to_real((float)(wx * wy) * 0.0625f);
            next_val += emitter_value * weight;
        }
    }
    next_buffer[idx] = next_val;
    if (step_index >= 0 && ear_samples && ear_sample_size > 0) {
        int base = step_index * 2;
        if (left_index >= 0 && left_index < ear_sample_size && idx == left_index) {
            ear_samples[base] = next_val;
        }
        if (right_index >= 0 && right_index < ear_sample_size && idx == right_index) {
            ear_samples[base + 1] = next_val;
        }
    }
}

__kernel void render_intensity(
    const int width,
    const int height,
    __global const real_t* curr,
    const float gamma,
    const int show_walls,
    __global const uchar* wall_mask,
    __global uchar4* pixels)
{
    int idx = get_global_id(0);
    int size = width * height;
    if (idx >= size) {
        return;
    }
    float value = to_float(curr[idx]);
    value = fmin(fmax(value, -1.0f), 1.0f);
    float brightness = fabs(value);
    float gammaRecip = gamma > 0.0f ? 1.0f / gamma : 1.0f;
    float corrected = pow(brightness, gammaRecip);
    uchar intensity = (uchar)(corrected * 255.0f);
    uchar4 color = (uchar4)(intensity, intensity, intensity, (uchar)255);
    if (show_walls) {
        if (wall_mask[idx]) {
            color.x = 30;
            color.y = 40;
            color.z = 80;
        }
    }
    pixels[idx] = color;
}

`

func newOpenCLWaveSolver(width, height int) (*openCLWaveSolver, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		msg := "querying OpenCL platforms"
		if strings.Contains(err.Error(), "-1001") {
			msg += ": no ICD loader reported any platforms; install OpenCL drivers and verify with `clinfo`"
		}
		return nil, fmt.Errorf("%s: %w", msg, err)
	}
	if len(platforms) == 0 {
		return nil, errors.New("no OpenCL platforms available; ensure a vendor driver is installed and detected by `clinfo`")
	}
	var device *cl.Device
	for _, p := range platforms {
		devices, derr := p.GetDevices(cl.DeviceTypeGPU)
		if derr != nil && derr != cl.ErrDeviceNotFound {
			continue
		}
		if len(devices) > 0 {
			device = devices[0]
			break
		}
	}
	if device == nil {
		for _, p := range platforms {
			devices, derr := p.GetDevices(cl.DeviceTypeCPU)
			if derr != nil && derr != cl.ErrDeviceNotFound {
				continue
			}
			if len(devices) > 0 {
				device = devices[0]
				break
			}
		}
	}
	if device == nil {
		return nil, errors.New("no suitable OpenCL devices found")
	}

	useFP16 := false
	if preferFP16Flag != nil && *preferFP16Flag {
		extensions := device.Extensions()
		if strings.Contains(extensions, "cl_khr_fp16") || strings.Contains(extensions, "cl_khr_half_float") {
			useFP16 = true
		}
	}
	elementBytes := int(unsafe.Sizeof(float32(0)))
	if useFP16 {
		elementBytes = 2
	}

	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, fmt.Errorf("creating OpenCL context: %w", err)
	}
	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		context.Release()
		return nil, fmt.Errorf("creating OpenCL command queue: %w", err)
	}
	program, err := context.CreateProgramWithSource([]string{waveKernelSource})
	if err != nil {
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating OpenCL program: %w", err)
	}
	buildOptions := ""
	if useFP16 {
		buildOptions = "-DUSE_FP16=1"
	}
	if err := program.BuildProgram([]*cl.Device{device}, buildOptions); err != nil {
		program.Release()
		queue.Release()
		context.Release()
		if buildErr, ok := err.(cl.BuildError); ok {
			return nil, fmt.Errorf("building OpenCL program: %s", string(buildErr))
		}
		return nil, fmt.Errorf("building OpenCL program: %w", err)
	}
	kernel, err := program.CreateKernel("wave_step")
	if err != nil {
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating OpenCL kernel: %w", err)
	}
	renderKernel, err := program.CreateKernel("render_intensity")
	if err != nil {
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating render kernel: %w", err)
	}
	size := width * height
	byteSize := size * elementBytes
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating current buffer: %w", err)
	}
	prevBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		currBuf.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating previous buffer: %w", err)
	}
	nextBuf, err := context.CreateEmptyBuffer(cl.MemWriteOnly, byteSize)
	if err != nil {
		prevBuf.Release()
		currBuf.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating next buffer: %w", err)
	}
	pixelBuf, err := context.CreateEmptyBuffer(cl.MemWriteOnly, size*4)
	if err != nil {
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating pixel buffer: %w", err)
	}
	wallMaskBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall mask buffer: %w", err)
	}

	earSampleBuf, err := context.CreateEmptyBuffer(cl.MemReadWrite, maxSimMultiplier*2*elementBytes)
	if err != nil {
		wallMaskBuf.Release()
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating ear sample buffer: %w", err)
	}

	waveGlobal, waveLocal := computeWaveKernelWorkSizes(width, height, kernel, device)
	solver := &openCLWaveSolver{
		context:             context,
		queue:               queue,
		program:             program,
		kernel:              kernel,
		renderKernel:        renderKernel,
		currBuf:             currBuf,
		prevBuf:             prevBuf,
		nextBuf:             nextBuf,
		pixelBuf:            pixelBuf,
		earSampleBuf:        earSampleBuf,
		wallMaskBuf:         wallMaskBuf,
		width:               width,
		height:              height,
		useFP16:             useFP16,
		elementBytes:        elementBytes,
		deviceName:          device.Name(),
		device:              device,
		waveGlobal:          waveGlobal,
		waveLocal:           waveLocal,
		coldStart:           true,
		hostPixels:          make([]byte, size*4),
		hostWallMask:        make([]byte, size),
		lastRenderShowWalls: -1,
	}

	precision := "fp32"
	if useFP16 {
		precision = "fp16"
	}
	fmt.Printf("OpenCL device: %s (precision %s)\n", solver.deviceName, precision)

	coeffs, err := computeWaveCoefficients(defaultTPS * float64(defaultSimMultiplier))
	if err != nil {
		solver.Close()
		return nil, fmt.Errorf("computing wave coefficients: %w", err)
	}
	// Kernel args: width, height, damp, speed, world_boundary_reflect, room_wall_reflect,
	//              curr, prev, next, wall_mask, ear_samples, ear_sample_size,
	//              left_index, right_index, step_index, emitter_index, emitter_value
	if err := solver.kernel.SetArgs(
		int32(width),
		int32(height),
		coeffs.DampPerStep,
		coeffs.SpeedCoeff,
		float32(worldBoundaryReflect),
		float32(roomWallReflect),
		solver.currBuf,
		solver.prevBuf,
		solver.nextBuf,
		solver.wallMaskBuf,
		solver.earSampleBuf,
		int32(size),
		int32(-1),
		int32(-1),
		int32(0),
		int32(-1),
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel arguments: %w", err)
	}
	if err := solver.setEmitterArgs(-1, 0); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel emitter defaults: %w", err)
	}
	solver.lastDamp = coeffs.DampPerStep
	solver.lastSpeed = coeffs.SpeedCoeff
	solver.lastWorldBoundaryReflect = float32(worldBoundaryReflect)
	solver.lastRoomWallReflect = float32(roomWallReflect)
	if err := solver.renderKernel.SetArgs(
		int32(width),
		int32(height),
		solver.currBuf,
		float32(visualGamma),
		int32(0),
		solver.wallMaskBuf,
		solver.pixelBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting render kernel arguments: %w", err)
	}

	return solver, nil
}

func ensureFloat32Slice(buf []float32, size int) []float32 {
	if cap(buf) < size {
		return make([]float32, size)
	}
	return buf[:size]
}

func ensureUint16Slice(buf []uint16, size int) []uint16 {
	if cap(buf) < size {
		return make([]uint16, size)
	}
	return buf[:size]
}

func computeWaveKernelWorkSizes(width, height int, kernel *cl.Kernel, device *cl.Device) ([]int, []int) {
	if width <= 0 || height <= 0 || kernel == nil || device == nil {
		return []int{width, height}, nil
	}
	maxWorkGroupSize, err := kernel.WorkGroupSize(device)
	if err != nil || maxWorkGroupSize <= 0 {
		return []int{width, height}, nil
	}
	localX := width
	if pref, err := kernel.PreferredWorkGroupSizeMultiple(device); err == nil && pref > 0 {
		localX = pref
	}
	if localX < 1 {
		localX = 1
	}
	if localX > width {
		localX = width
	}
	if localX > maxWorkGroupSize {
		localX = maxWorkGroupSize
	}
	if localX == 0 {
		localX = 1
	}
	maxY := maxWorkGroupSize / localX
	if maxY < 1 {
		maxY = 1
	}
	localY := height
	if localY > maxY {
		localY = maxY
	}
	if localY < 1 {
		localY = 1
	}
	globalX := roundUp(width, localX)
	globalY := roundUp(height, localY)
	return []int{globalX, globalY}, []int{localX, localY}
}

func roundUp(value, align int) int {
	if align <= 0 {
		return value
	}
	remainder := value % align
	if remainder == 0 {
		return value
	}
	return value + align - remainder
}

func (s *openCLWaveSolver) writeFieldBuffer(buf *cl.MemObject, data []float32, halfScratch *[]uint16) error {
	if len(data) == 0 {
		return nil
	}
	if s.useFP16 {
		*halfScratch = ensureUint16Slice(*halfScratch, len(data))
		float32ToFloat16(*halfScratch, data)
		byteLen := len(data) * s.elementBytes
		if _, err := s.queue.EnqueueWriteBuffer(buf, false, 0, byteLen, unsafe.Pointer(&(*halfScratch)[0]), nil); err != nil {
			return err
		}
		return nil
	}
	if _, err := s.queue.EnqueueWriteBufferFloat32(buf, false, 0, data, nil); err != nil {
		return err
	}
	return nil
}

func (s *openCLWaveSolver) setEmitterArgs(index int32, value float32) error {
	if err := s.kernel.SetArgInt32(15, index); err != nil {
		return err
	}
	return s.setEmitterValue(value)
}

func (s *openCLWaveSolver) setEmitterValue(val float32) error {
	if s.useFP16 {
		half := float32ToFloat16Bits(val)
		return s.kernel.SetArgUnsafe(16, int(unsafe.Sizeof(half)), unsafe.Pointer(&half))
	}
	return s.kernel.SetArgFloat32(16, val)
}

func (s *openCLWaveSolver) setEarSampleArgs(size int32, left int32, right int32) error {
	if s.kernel == nil {
		return nil
	}
	if err := s.kernel.SetArgInt32(11, size); err != nil {
		return err
	}
	if err := s.kernel.SetArgInt32(12, left); err != nil {
		return err
	}
	return s.kernel.SetArgInt32(13, right)
}

func (s *openCLWaveSolver) setStepIndex(step int32) error {
	if s.kernel == nil {
		return nil
	}
	return s.kernel.SetArgInt32(14, step)
}

func (s *openCLWaveSolver) bindDynamicBuffers() error {
	if s.boundCurr != s.currBuf {
		if err := s.kernel.SetArgBuffer(6, s.currBuf); err != nil {
			return err
		}
		s.boundCurr = s.currBuf
	}
	if s.boundPrev != s.prevBuf {
		if err := s.kernel.SetArgBuffer(7, s.prevBuf); err != nil {
			return err
		}
		s.boundPrev = s.prevBuf
	}
	if s.boundNext != s.nextBuf {
		if err := s.kernel.SetArgBuffer(8, s.nextBuf); err != nil {
			return err
		}
		s.boundNext = s.nextBuf
	}
	return nil
}

func (s *openCLWaveSolver) setWaveCoefficients(damp, speed float32) error {
	if s.kernel == nil {
		return nil
	}
	if damp != s.lastDamp {
		if err := s.kernel.SetArgFloat32(2, damp); err != nil {
			return err
		}
		s.lastDamp = damp
	}
	if speed != s.lastSpeed {
		if err := s.kernel.SetArgFloat32(3, speed); err != nil {
			return err
		}
		s.lastSpeed = speed
	}
	return nil
}

func (s *openCLWaveSolver) setWorldBoundaryReflect(reflect float32) error {
	if s.kernel == nil {
		return nil
	}
	if reflect != s.lastWorldBoundaryReflect {
		if err := s.kernel.SetArgFloat32(4, reflect); err != nil {
			return err
		}
		s.lastWorldBoundaryReflect = reflect
	}
	return nil
}

func (s *openCLWaveSolver) setRoomWallReflect(reflect float32) error {
	if s.kernel == nil {
		return nil
	}
	if reflect != s.lastRoomWallReflect {
		if err := s.kernel.SetArgFloat32(5, reflect); err != nil {
			return err
		}
		s.lastRoomWallReflect = reflect
	}
	return nil
}

func (s *openCLWaveSolver) refreshWallMask(walls []bool) error {
	size := s.width * s.height
	if len(walls) != size {
		s.wallMaskSynced = false
		return nil
	}
	for i, wall := range walls {
		if wall {
			s.hostWallMask[i] = 1
		} else {
			s.hostWallMask[i] = 0
		}
	}
	if size == 0 {
		s.wallMaskSynced = true
		return nil
	}
	if _, err := s.queue.EnqueueWriteBuffer(s.wallMaskBuf, false, 0, size, unsafe.Pointer(&s.hostWallMask[0]), nil); err != nil {
		return fmt.Errorf("writing wall mask buffer: %w", err)
	}
	s.wallMaskSynced = true
	return nil
}

func (s *openCLWaveSolver) setRenderFlags(showWalls bool) error {
	show := int32(0)
	if showWalls {
		show = 1
	}
	if s.lastRenderShowWalls != show {
		if err := s.renderKernel.SetArgInt32(4, show); err != nil {
			return err
		}
		s.lastRenderShowWalls = show
	}
	return nil
}

func (s *openCLWaveSolver) Step(field *waveField, walls []bool, steps int, dtSeconds float64, wallsDirty bool, showWalls bool, leftIndex int32, rightIndex int32, emitter *audioEmitterData) error {
	if steps <= 0 {
		return nil
	}
	if dtSeconds <= 0 {
		dtSeconds = 1.0 / defaultTPS
	}
	coeffs, err := computeWaveCoefficients(float64(steps) / dtSeconds)
	if err != nil {
		return err
	}
	if coeffs.Clamped && !s.warnedCoeffClamp {
		log.Printf("Wave coefficient clamped for stability: (c*dt/dx)^2=%.4f -> %.4f (c=%.1fm/s dx=%.4fm dt=%.6fs steps/s=%.1f). Increase sim steps or increase cell size.",
			float64(coeffs.Courant*coeffs.Courant), float64(coeffs.SpeedCoeff), coeffs.SpeedSoundMS, coeffs.DxMeters, coeffs.DtSeconds, coeffs.StepsPerSec)
		s.warnedCoeffClamp = true
	}
	if err := s.setWaveCoefficients(coeffs.DampPerStep, coeffs.SpeedCoeff); err != nil {
		return fmt.Errorf("setting wave coefficients: %w", err)
	}
	if err := s.setWorldBoundaryReflect(float32(worldBoundaryReflect)); err != nil {
		return fmt.Errorf("setting world boundary reflect: %w", err)
	}
	if err := s.setRoomWallReflect(float32(roomWallReflect)); err != nil {
		return fmt.Errorf("setting room wall reflect: %w", err)
	}
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return fmt.Errorf("unexpected field buffer size")
	}
	defaultIndex := int32((s.height/2)*s.width + (s.width / 2))
	if leftIndex < 0 || int(leftIndex) >= size {
		leftIndex = defaultIndex
	}
	if rightIndex < 0 || int(rightIndex) >= size {
		rightIndex = defaultIndex
	}
	var emitterIndex int32 = -1
	var emitterSamples []float32
	if emitter != nil && emitter.index >= 0 && len(emitter.samples) > 0 {
		if int(emitter.index) < size {
			emitterIndex = emitter.index
			emitterSamples = emitter.samples
		}
	}
	if s.coldStart && size > 0 {
		if err := s.writeFieldBuffer(s.currBuf, field.curr, &s.hostCurrHalf); err != nil {
			return fmt.Errorf("initializing current buffer: %w", err)
		}
		if err := s.writeFieldBuffer(s.prevBuf, field.prev, &s.hostPrevHalf); err != nil {
			return fmt.Errorf("initializing previous buffer: %w", err)
		}
		if err := s.writeFieldBuffer(s.nextBuf, field.next, &s.hostNextHalf); err != nil {
			return fmt.Errorf("initializing next buffer: %w", err)
		}
	}
	if !s.wallMaskSynced || wallsDirty {
		if err := s.refreshWallMask(walls); err != nil {
			return err
		}
	}
	if showWalls && len(walls) != size {
		showWalls = false
	}

	waveGlobal := s.waveGlobal
	if len(waveGlobal) != 2 {
		waveGlobal = []int{s.width, s.height}
	}
	waveLocal := s.waveLocal
	if len(waveLocal) != 0 && len(waveLocal) != len(waveGlobal) {
		waveLocal = nil
	}

	if steps > 0 {
		s.lastSampleCount = steps
		s.hostEarSamples = ensureFloat32Slice(s.hostEarSamples, steps*2)
		if s.useFP16 {
			s.hostEarSamplesHalf = ensureUint16Slice(s.hostEarSamplesHalf, steps*2)
		}
		if err := s.setEarSampleArgs(int32(size), leftIndex, rightIndex); err != nil {
			return fmt.Errorf("setting ear sample args: %w", err)
		}
	} else {
		s.lastSampleCount = 0
		s.earLeftSample = 0
		s.earRightSample = 0
		if err := s.setEarSampleArgs(int32(size), -1, -1); err != nil {
			return fmt.Errorf("clearing ear sample args: %w", err)
		}
	}
	if err := s.setEmitterArgs(emitterIndex, 0); err != nil {
		return fmt.Errorf("setting emitter args: %w", err)
	}
	for step := 0; step < steps; step++ {
		if err := s.setStepIndex(int32(step)); err != nil {
			return fmt.Errorf("setting step index: %w", err)
		}
		emitterValue := float32(0)
		if emitterIndex >= 0 && step < len(emitterSamples) {
			emitterValue = emitterSamples[step]
		}
		if err := s.setEmitterValue(emitterValue); err != nil {
			return fmt.Errorf("setting emitter value: %w", err)
		}
		if err := s.bindDynamicBuffers(); err != nil {
			return fmt.Errorf("binding buffers: %w", err)
		}
		if _, err := s.queue.EnqueueNDRangeKernel(s.kernel, nil, waveGlobal, waveLocal, nil); err != nil {
			return fmt.Errorf("enqueueing kernel: %w", err)
		}
		s.prevBuf, s.currBuf, s.nextBuf = s.currBuf, s.nextBuf, s.prevBuf
	}
	if steps > 0 {
		byteLen := steps * 2 * s.elementBytes
		if s.useFP16 {
			if _, err := s.queue.EnqueueReadBuffer(s.earSampleBuf, true, 0, byteLen, unsafe.Pointer(&s.hostEarSamplesHalf[0]), nil); err != nil {
				return fmt.Errorf("reading ear samples (fp16): %w", err)
			}
			float16ToFloat32(s.hostEarSamples, s.hostEarSamplesHalf)
		} else {
			if _, err := s.queue.EnqueueReadBufferFloat32(s.earSampleBuf, true, 0, s.hostEarSamples[:steps*2], nil); err != nil {
				return fmt.Errorf("reading ear samples: %w", err)
			}
		}
		base := (steps - 1) * 2
		s.earLeftSample = s.hostEarSamples[base]
		s.earRightSample = s.hostEarSamples[base+1]
	}

	// Update render kernel to use current buffer (always last frame only)
	if err := s.renderKernel.SetArgBuffer(2, s.currBuf); err != nil {
		return fmt.Errorf("setting render source: %w", err)
	}
	if err := s.setRenderFlags(showWalls); err != nil {
		return fmt.Errorf("configuring render overlays: %w", err)
	}
	renderGlobal := []int{size}
	if _, err := s.queue.EnqueueNDRangeKernel(s.renderKernel, nil, renderGlobal, nil, nil); err != nil {
		return fmt.Errorf("enqueueing render kernel: %w", err)
	}
	if size > 0 && len(s.hostPixels) > 0 {
		event, err := s.queue.EnqueueReadBuffer(s.pixelBuf, false, 0, len(s.hostPixels), unsafe.Pointer(&s.hostPixels[0]), nil)
		if err != nil {
			return fmt.Errorf("queueing pixel read: %w", err)
		}
		s.pixelMu.Lock()
		if s.pixelEvent != nil {
			s.pixelEvent.Release()
		}
		s.pixelEvent = event
		s.pixelMu.Unlock()
	}
	s.coldStart = false
	return nil
}

func (s *openCLWaveSolver) Close() {
	if err := s.waitForPixelEvent(); err != nil {
		fmt.Printf("waiting for pending pixel read during close: %v\n", err)
	}
	if s.pixelBuf != nil {
		s.pixelBuf.Release()
		s.pixelBuf = nil
	}
	if s.wallMaskBuf != nil {
		s.wallMaskBuf.Release()
		s.wallMaskBuf = nil
	}
	if s.earSampleBuf != nil {
		s.earSampleBuf.Release()
		s.earSampleBuf = nil
	}
	if s.nextBuf != nil {
		s.nextBuf.Release()
		s.nextBuf = nil
	}
	if s.prevBuf != nil {
		s.prevBuf.Release()
		s.prevBuf = nil
	}
	if s.currBuf != nil {
		s.currBuf.Release()
		s.currBuf = nil
	}
	if s.kernel != nil {
		s.kernel.Release()
		s.kernel = nil
	}
	if s.renderKernel != nil {
		s.renderKernel.Release()
		s.renderKernel = nil
	}
	if s.program != nil {
		s.program.Release()
		s.program = nil
	}
	if s.queue != nil {
		s.queue.Release()
		s.queue = nil
	}
	if s.context != nil {
		s.context.Release()
		s.context = nil
	}
}

func (s *openCLWaveSolver) DeviceName() string {
	return s.deviceName
}

func (s *openCLWaveSolver) PixelBytes() []byte {
	if err := s.waitForPixelEvent(); err != nil {
		fmt.Printf("waiting for pixel readback: %v\n", err)
	}
	return s.hostPixels
}

func (s *openCLWaveSolver) EarSample() (float32, float32) {
	return s.earLeftSample, s.earRightSample
}

func (s *openCLWaveSolver) EarSamplesInterleaved() []float32 {
	if s.lastSampleCount <= 0 || s.lastSampleCount*2 > len(s.hostEarSamples) {
		return nil
	}
	return s.hostEarSamples[:s.lastSampleCount*2]
}

func (s *openCLWaveSolver) waitForPixelEvent() error {
	s.pixelMu.Lock()
	event := s.pixelEvent
	s.pixelEvent = nil
	s.pixelMu.Unlock()
	if event == nil {
		return nil
	}
	defer event.Release()
	return cl.WaitForEvents([]*cl.Event{event})
}
