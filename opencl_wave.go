package main

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
)

type openCLWaveSolver struct {
	context                 *cl.Context
	queue                   *cl.CommandQueue
	program                 *cl.Program
	kernel                  *cl.Kernel
	renderKernel            *cl.Kernel
	sampleKernel            *cl.Kernel
	applyImpulsesKernel     *cl.Kernel
	boundaryAccumKernel     *cl.Kernel
	currBuf                 *cl.MemObject
	prevBuf                 *cl.MemObject
	nextBuf                 *cl.MemObject
	pixelBuf                *cl.MemObject
	accumBuf                *cl.MemObject
	centerSampleBuf         *cl.MemObject
	wallMaskBuf             *cl.MemObject
	visibilityBuf           *cl.MemObject
	impulseIndexBuf         *cl.MemObject
	impulseValueBuf         *cl.MemObject
	width                   int
	height                  int
	useFP16                 bool
	elementBytes            int
	wallMaskSynced          bool
	deviceName              string
	device                  *cl.Device
	coldStart               bool
	waveGlobal              []int
	waveLocal               []int
	boundCurr               *cl.MemObject
	boundPrev               *cl.MemObject
	boundNext               *cl.MemObject
	hostPixels              []byte
	hostWallMask            []byte
	hostVisibility          []byte
	hostCenterSamples       []float32
	hostCenterSamplesHalf   []uint16
	pixelMu                 sync.Mutex
	pixelEvent              *cl.Event
	uploadedVisibleGen      uint32
	visibleMaskSynced       bool
	lastRenderShowWalls     int32
	lastRenderUseVisibility int32
	debugVerify             bool
	debugScratch            []float32
	debugScratch16          []uint16
	impulseCurrIndices      []int32
	impulseCurrValues       []float32
	impulsePrevIndices      []int32
	impulsePrevValues       []float32
	hostCurrHalf            []uint16
	hostPrevHalf            []uint16
	hostNextHalf            []uint16
	impulseCurrHalf         []uint16
	impulsePrevHalf         []uint16
	centerSample            float32
	lastSampleCount         int
}

type audioEmitterData struct {
	index   int32
	samples []float32
}

const verifyTolerance = 1e-4

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
    __global const real_t* curr,
    __global const real_t* prev,
    __global real_t* next_buffer,
    __global const uchar* wall_mask,
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
        return;
    }
    int left = idx - 1;
    int right = idx + 1;
    int top = idx - width;
    int bottom = idx + width;
    const real_t damp_r = to_real(damp);
    const real_t speed_r = to_real(speed);
    const real_t two = to_real(2.0f);
    const real_t four = to_real(4.0f);
    real_t center = curr[idx];
    real_t laplacian = curr[left] + curr[right] + curr[top] + curr[bottom] - four * center;
    real_t next_val = ((two * center - prev[idx]) + speed_r * laplacian) * damp_r;
    if (idx == emitter_index && emitter_index >= 0) {
        next_buffer[idx] = emitter_value;
    } else {
        next_buffer[idx] = next_val;
    }
}

__kernel void apply_impulses(
    const int count,
    __global const int* indices,
    __global const real_t* values,
    __global real_t* buffer)
{
    int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    int idx = indices[gid];
    buffer[idx] = values[gid];
}

__kernel void render_intensity(
    const int width,
    const int height,
    __global const real_t* curr,
    const int show_walls,
    __global const uchar* wall_mask,
    const int use_visibility,
    __global const uchar* visibility_mask,
    __global uchar4* pixels)
{
    int idx = get_global_id(0);
    int size = width * height;
    if (idx >= size) {
        return;
    }
    float value = to_float(curr[idx]);
    value = fmin(fmax(value, -1.0f), 1.0f);
    uchar intensity = (uchar)(fabs(value) * 255.0f);
    uchar4 color = (uchar4)(intensity, intensity, intensity, (uchar)255);
    if (use_visibility) {
        if (!visibility_mask[idx]) {
            color.x = 0;
            color.y = 0;
            color.z = 0;
        }
    }
    if (show_walls) {
        if (wall_mask[idx]) {
            color.x = 30;
            color.y = 40;
            color.z = 80;
        }
    }
    pixels[idx] = color;
}

__kernel void accumulate_frame(
    const int size,
    const float scale,
    __global const real_t* source,
    __global real_t* accum)
{
    int idx = get_global_id(0);
    if (idx >= size) {
        return;
    }
    float value = fabs(to_float(source[idx]));
    float scaled = value * scale;
    accum[idx] += to_real(scaled);
}

__kernel void boundary_accumulate(
    const int width,
    const int height,
    const float reflect,
    const float scale,
    __global real_t* buffer,
    __global real_t* accum)
{
    if (width <= 0 || height <= 0) {
        return;
    }
    int idx = get_global_id(0);
    int size = width * height;
    if (idx >= size) {
        return;
    }
    int x = idx % width;
    int y = idx / width;
    const real_t reflect_r = to_real(reflect);
    if (height > 1 && y == 0) {
        int src = width + x;
        buffer[idx] = -buffer[src] * reflect_r;
    } else if (height > 1 && y == height - 1) {
        int src = (height - 2) * width + x;
        buffer[idx] = -buffer[src] * reflect_r;
    } else if (width > 1 && x == 0) {
        int src = y*width + 1;
        buffer[idx] = -buffer[src] * reflect_r;
    } else if (width > 1 && x == width - 1) {
        int src = y*width + width - 2;
        buffer[idx] = -buffer[src] * reflect_r;
    }
    float value = fabs(to_float(buffer[idx]));
    real_t scaled = to_real(value * scale);
    accum[idx] += scaled;
}

__kernel void sample_center(
    const int step_index,
    const int width,
    const int height,
    __global const real_t* curr,
    __global real_t* samples)
{
    int cx = width / 2;
    int cy = height / 2;
    if (cx < 0 || cy < 0 || cx >= width || cy >= height) {
        return;
    }
    int idx = cy * width + cx;
    samples[step_index] = curr[idx];
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
	boundaryAccumKernel, err := program.CreateKernel("boundary_accumulate")
	if err != nil {
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating boundary accumulate kernel: %w", err)
	}
	applyImpulsesKernel, err := program.CreateKernel("apply_impulses")
	if err != nil {
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating impulse kernel: %w", err)
	}
	size := width * height
	byteSize := size * elementBytes
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
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
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
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
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
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
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating pixel buffer: %w", err)
	}
	accumBuf, err := context.CreateEmptyBuffer(cl.MemReadWrite, byteSize)
	if err != nil {
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating accumulation buffer: %w", err)
	}
	wallMaskBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall mask buffer: %w", err)
	}
	visibilityBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		wallMaskBuf.Release()
		pixelBuf.Release()
		accumBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating visibility buffer: %w", err)
	}
	impulseIndexBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*int(unsafe.Sizeof(int32(0))))
	if err != nil {
		visibilityBuf.Release()
		wallMaskBuf.Release()
		pixelBuf.Release()
		accumBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating impulse index buffer: %w", err)
	}
	impulseValueBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*elementBytes)
	if err != nil {
		impulseIndexBuf.Release()
		visibilityBuf.Release()
		wallMaskBuf.Release()
		pixelBuf.Release()
		accumBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		boundaryAccumKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating impulse value buffer: %w", err)
	}

	var centerSampleBuf *cl.MemObject
	var sampleKernel *cl.Kernel
	if captureStepSamplesFlag != nil && *captureStepSamplesFlag {
		centerSampleBuf, err = context.CreateEmptyBuffer(cl.MemReadWrite, maxSimMultiplier*elementBytes)
		if err != nil {
			impulseValueBuf.Release()
			impulseIndexBuf.Release()
			visibilityBuf.Release()
			wallMaskBuf.Release()
			pixelBuf.Release()
			accumBuf.Release()
			nextBuf.Release()
			prevBuf.Release()
			currBuf.Release()
			applyImpulsesKernel.Release()
			boundaryAccumKernel.Release()
			renderKernel.Release()
			kernel.Release()
			program.Release()
			queue.Release()
			context.Release()
			return nil, fmt.Errorf("allocating center sample buffer: %w", err)
		}

		sampleKernel, err = program.CreateKernel("sample_center")
		if err != nil {
			centerSampleBuf.Release()
			impulseValueBuf.Release()
			impulseIndexBuf.Release()
			visibilityBuf.Release()
			wallMaskBuf.Release()
			pixelBuf.Release()
			accumBuf.Release()
			nextBuf.Release()
			prevBuf.Release()
			currBuf.Release()
			applyImpulsesKernel.Release()
			boundaryAccumKernel.Release()
			renderKernel.Release()
			kernel.Release()
			program.Release()
			queue.Release()
			context.Release()
			return nil, fmt.Errorf("creating sample kernel: %w", err)
		}
	}

	waveGlobal, waveLocal := computeWaveKernelWorkSizes(width, height, kernel, device)
	solver := &openCLWaveSolver{
		context:                 context,
		queue:                   queue,
		program:                 program,
		kernel:                  kernel,
		renderKernel:            renderKernel,
		sampleKernel:            sampleKernel,
		applyImpulsesKernel:     applyImpulsesKernel,
		boundaryAccumKernel:     boundaryAccumKernel,
		currBuf:                 currBuf,
		prevBuf:                 prevBuf,
		nextBuf:                 nextBuf,
		pixelBuf:                pixelBuf,
		accumBuf:                accumBuf,
		centerSampleBuf:         centerSampleBuf,
		wallMaskBuf:             wallMaskBuf,
		visibilityBuf:           visibilityBuf,
		impulseIndexBuf:         impulseIndexBuf,
		impulseValueBuf:         impulseValueBuf,
		width:                   width,
		height:                  height,
		useFP16:                 useFP16,
		elementBytes:            elementBytes,
		deviceName:              device.Name(),
		device:                  device,
		waveGlobal:              waveGlobal,
		waveLocal:               waveLocal,
		coldStart:               true,
		hostPixels:              make([]byte, size*4),
		hostWallMask:            make([]byte, size),
		hostVisibility:          make([]byte, size),
		lastRenderShowWalls:     -1,
		lastRenderUseVisibility: -1,
		debugVerify:             verifyOpenCLSyncFlag != nil && *verifyOpenCLSyncFlag,
	}

	precision := "fp32"
	if useFP16 {
		precision = "fp16"
	}
	fmt.Printf("OpenCL device: %s (precision %s)\n", solver.deviceName, precision)

	if err := solver.kernel.SetArgs(
		int32(width),
		int32(height),
		waveDamp32,
		waveSpeed32,
		solver.currBuf,
		solver.prevBuf,
		solver.nextBuf,
		solver.wallMaskBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel arguments: %w", err)
	}
	if err := solver.setEmitterArgs(-1, 0); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel emitter defaults: %w", err)
	}
	if err := solver.renderKernel.SetArgs(
		int32(width),
		int32(height),
		solver.accumBuf,
		int32(0),
		solver.wallMaskBuf,
		int32(0),
		solver.visibilityBuf,
		solver.pixelBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting render kernel arguments: %w", err)
	}

	return solver, nil
}

// audio sampling helpers removed

func (s *openCLWaveSolver) ensureDebugScratch(size int) []float32 {
	if cap(s.debugScratch) < size {
		s.debugScratch = make([]float32, size)
	}
	s.debugScratch = s.debugScratch[:size]
	return s.debugScratch
}

func (s *openCLWaveSolver) ensureDebugScratch16(size int) []uint16 {
	if cap(s.debugScratch16) < size {
		s.debugScratch16 = make([]uint16, size)
	}
	s.debugScratch16 = s.debugScratch16[:size]
	return s.debugScratch16
}

func ensureInt32Slice(buf []int32, size int) []int32 {
	if cap(buf) < size {
		return make([]int32, size)
	}
	return buf[:size]
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

func (s *openCLWaveSolver) verifyBufferMatchesSlice(buf *cl.MemObject, host []float32, label string) error {
	if len(host) == 0 {
		return nil
	}
	scratch := s.ensureDebugScratch(len(host))
	if s.useFP16 {
		raw := s.ensureDebugScratch16(len(host))
		if _, err := s.queue.EnqueueReadBuffer(buf, true, 0, len(raw)*s.elementBytes, unsafe.Pointer(&raw[0]), nil); err != nil {
			return fmt.Errorf("reading %s for verification: %w", label, err)
		}
		float16ToFloat32(scratch, raw)
	} else {
		if _, err := s.queue.EnqueueReadBufferFloat32(buf, true, 0, scratch, nil); err != nil {
			return fmt.Errorf("reading %s for verification: %w", label, err)
		}
	}
	for i, hv := range host {
		if diff := math.Abs(float64(scratch[i] - hv)); diff > verifyTolerance {
			return fmt.Errorf("%s mismatch at index %d: device=%f host=%f diff=%f", label, i, scratch[i], hv, diff)
		}
	}
	return nil
}

func (s *openCLWaveSolver) dispatchImpulses(target *cl.MemObject, indices []int32, values []float32, halfScratch *[]uint16) error {
	if len(indices) == 0 {
		return nil
	}
	if len(values) != len(indices) {
		return fmt.Errorf("impulse data mismatch: %d indices vs %d values", len(indices), len(values))
	}
	byteIdx := len(indices) * int(unsafe.Sizeof(int32(0)))
	if _, err := s.queue.EnqueueWriteBuffer(s.impulseIndexBuf, false, 0, byteIdx, unsafe.Pointer(&indices[0]), nil); err != nil {
		return fmt.Errorf("uploading impulse indices: %w", err)
	}
	if len(values) > 0 {
		byteVal := len(values) * s.elementBytes
		if s.useFP16 {
			*halfScratch = ensureUint16Slice(*halfScratch, len(values))
			float32ToFloat16(*halfScratch, values)
			if _, err := s.queue.EnqueueWriteBuffer(s.impulseValueBuf, false, 0, byteVal, unsafe.Pointer(&(*halfScratch)[0]), nil); err != nil {
				return fmt.Errorf("uploading impulse values: %w", err)
			}
		} else {
			if _, err := s.queue.EnqueueWriteBuffer(s.impulseValueBuf, false, 0, byteVal, unsafe.Pointer(&values[0]), nil); err != nil {
				return fmt.Errorf("uploading impulse values: %w", err)
			}
		}
	}
	if err := s.applyImpulsesKernel.SetArgInt32(0, int32(len(indices))); err != nil {
		return fmt.Errorf("setting impulse count: %w", err)
	}
	if err := s.applyImpulsesKernel.SetArgBuffer(1, s.impulseIndexBuf); err != nil {
		return fmt.Errorf("binding impulse indices: %w", err)
	}
	if err := s.applyImpulsesKernel.SetArgBuffer(2, s.impulseValueBuf); err != nil {
		return fmt.Errorf("binding impulse values: %w", err)
	}
	if err := s.applyImpulsesKernel.SetArgBuffer(3, target); err != nil {
		return fmt.Errorf("binding impulse target: %w", err)
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.applyImpulsesKernel, nil, []int{len(indices)}, nil, nil); err != nil {
		return fmt.Errorf("dispatching impulse kernel: %w", err)
	}
	return nil
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

func (s *openCLWaveSolver) applyQueuedImpulses(field *waveField) error {
	impulses := field.takeImpulses()
	if len(impulses) == 0 {
		return nil
	}
	count := len(impulses)
	s.impulseCurrIndices = ensureInt32Slice(s.impulseCurrIndices, count)
	s.impulseCurrValues = ensureFloat32Slice(s.impulseCurrValues, count)
	s.impulsePrevIndices = ensureInt32Slice(s.impulsePrevIndices, count)
	s.impulsePrevValues = ensureFloat32Slice(s.impulsePrevValues, count)
	prevCount := 0
	for i, imp := range impulses {
		s.impulseCurrIndices[i] = imp.index
		s.impulseCurrValues[i] = imp.value
		if imp.applyPrev {
			s.impulsePrevIndices[prevCount] = imp.index
			s.impulsePrevValues[prevCount] = imp.value
			prevCount++
		}
	}
	s.impulsePrevIndices = s.impulsePrevIndices[:prevCount]
	s.impulsePrevValues = s.impulsePrevValues[:prevCount]
	if err := s.dispatchImpulses(s.currBuf, s.impulseCurrIndices, s.impulseCurrValues, &s.impulseCurrHalf); err != nil {
		return err
	}
	if err := s.dispatchImpulses(s.prevBuf, s.impulsePrevIndices, s.impulsePrevValues, &s.impulsePrevHalf); err != nil {
		return err
	}
	return nil
}

func (s *openCLWaveSolver) sampleCenter(step int) error {
	if s.sampleKernel == nil || s.centerSampleBuf == nil {
		return nil
	}
	if err := s.sampleKernel.SetArgInt32(0, int32(step)); err != nil {
		return fmt.Errorf("setting sample step index: %w", err)
	}
	if err := s.sampleKernel.SetArgInt32(1, int32(s.width)); err != nil {
		return fmt.Errorf("setting sample width: %w", err)
	}
	if err := s.sampleKernel.SetArgInt32(2, int32(s.height)); err != nil {
		return fmt.Errorf("setting sample height: %w", err)
	}
	if err := s.sampleKernel.SetArgBuffer(3, s.currBuf); err != nil {
		return fmt.Errorf("binding sample source buffer: %w", err)
	}
	if err := s.sampleKernel.SetArgBuffer(4, s.centerSampleBuf); err != nil {
		return fmt.Errorf("binding sample target buffer: %w", err)
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.sampleKernel, nil, []int{1}, nil, nil); err != nil {
		return fmt.Errorf("enqueueing sample kernel: %w", err)
	}
	return nil
}

func (s *openCLWaveSolver) setEmitterArgs(index int32, value float32) error {
	if err := s.kernel.SetArgInt32(8, index); err != nil {
		return err
	}
	return s.setEmitterValue(value)
}

func (s *openCLWaveSolver) setEmitterValue(val float32) error {
	if s.useFP16 {
		half := float32ToFloat16Bits(val)
		return s.kernel.SetArgUnsafe(9, int(unsafe.Sizeof(half)), unsafe.Pointer(&half))
	}
	return s.kernel.SetArgFloat32(9, val)
}

func (s *openCLWaveSolver) bindDynamicBuffers() error {
	if s.boundCurr != s.currBuf {
		if err := s.kernel.SetArgBuffer(4, s.currBuf); err != nil {
			return err
		}
		s.boundCurr = s.currBuf
	}
	if s.boundPrev != s.prevBuf {
		if err := s.kernel.SetArgBuffer(5, s.prevBuf); err != nil {
			return err
		}
		s.boundPrev = s.prevBuf
	}
	if s.boundNext != s.nextBuf {
		if err := s.kernel.SetArgBuffer(6, s.nextBuf); err != nil {
			return err
		}
		s.boundNext = s.nextBuf
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

func (s *openCLWaveSolver) refreshVisibilityMask(stamp []uint32, gen uint32) error {
	size := s.width * s.height
	if len(stamp) != size {
		s.visibleMaskSynced = false
		return nil
	}
	if s.visibleMaskSynced && s.uploadedVisibleGen == gen {
		return nil
	}
	for i, value := range stamp {
		if value == gen {
			s.hostVisibility[i] = 1
		} else {
			s.hostVisibility[i] = 0
		}
	}
	if size == 0 {
		s.visibleMaskSynced = true
		s.uploadedVisibleGen = gen
		return nil
	}
	if _, err := s.queue.EnqueueWriteBuffer(s.visibilityBuf, false, 0, size, unsafe.Pointer(&s.hostVisibility[0]), nil); err != nil {
		return fmt.Errorf("writing visibility buffer: %w", err)
	}
	s.visibleMaskSynced = true
	s.uploadedVisibleGen = gen
	return nil
}

func (s *openCLWaveSolver) runBoundaryAccumulate(global []int, scale float32, reflect float32) error {
	if s.boundaryAccumKernel == nil || s.currBuf == nil || s.accumBuf == nil {
		return nil
	}
	if len(global) == 0 {
		return nil
	}
	if err := s.boundaryAccumKernel.SetArgInt32(0, int32(s.width)); err != nil {
		return fmt.Errorf("setting boundary accumulate width: %w", err)
	}
	if err := s.boundaryAccumKernel.SetArgInt32(1, int32(s.height)); err != nil {
		return fmt.Errorf("setting boundary accumulate height: %w", err)
	}
	if err := s.boundaryAccumKernel.SetArgFloat32(2, reflect); err != nil {
		return fmt.Errorf("setting boundary accumulate reflect: %w", err)
	}
	if err := s.boundaryAccumKernel.SetArgFloat32(3, scale); err != nil {
		return fmt.Errorf("setting boundary accumulate scale: %w", err)
	}
	if err := s.boundaryAccumKernel.SetArgBuffer(4, s.currBuf); err != nil {
		return fmt.Errorf("binding boundary accumulate buffer: %w", err)
	}
	if err := s.boundaryAccumKernel.SetArgBuffer(5, s.accumBuf); err != nil {
		return fmt.Errorf("binding boundary accumulate accum: %w", err)
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.boundaryAccumKernel, nil, global, nil, nil); err != nil {
		return fmt.Errorf("enqueueing boundary accumulate kernel: %w", err)
	}
	return nil
}

func (s *openCLWaveSolver) setRenderFlags(showWalls bool, useVisibility bool) error {
	show := int32(0)
	if showWalls {
		show = 1
	}
	if s.lastRenderShowWalls != show {
		if err := s.renderKernel.SetArgInt32(3, show); err != nil {
			return err
		}
		s.lastRenderShowWalls = show
	}
	useVis := int32(0)
	if useVisibility {
		useVis = 1
	}
	if s.lastRenderUseVisibility != useVis {
		if err := s.renderKernel.SetArgInt32(5, useVis); err != nil {
			return err
		}
		s.lastRenderUseVisibility = useVis
	}
	return nil
}

func (s *openCLWaveSolver) Step(field *waveField, walls []bool, steps int, wallsDirty bool, showWalls bool, occludeLOS bool, visibleStamp []uint32, visibleGen uint32, emitter *audioEmitterData) error {
	if steps <= 0 {
		return nil
	}
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return fmt.Errorf("unexpected field buffer size")
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
	if err := s.applyQueuedImpulses(field); err != nil {
		return fmt.Errorf("applying impulses: %w", err)
	}
	if s.debugVerify {
		if err := s.verifyBufferMatchesSlice(s.currBuf, field.curr, "pre-step curr"); err != nil {
			return err
		}
		if err := s.verifyBufferMatchesSlice(s.prevBuf, field.prev, "pre-step prev"); err != nil {
			return err
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
	useVisibility := false
	if occludeLOS && len(visibleStamp) == size {
		if err := s.refreshVisibilityMask(visibleStamp, visibleGen); err != nil {
			return err
		}
		useVisibility = true
	}
	if size > 0 {
		var zero32 float32
		var zero16 uint16
		pattern := unsafe.Pointer(&zero32)
		if s.elementBytes == 2 {
			pattern = unsafe.Pointer(&zero16)
		}
		byteSize := size * s.elementBytes
		if _, err := s.queue.EnqueueFillBuffer(s.accumBuf, pattern, s.elementBytes, 0, byteSize, nil); err != nil {
			return fmt.Errorf("clearing accumulation buffer: %w", err)
		}
	}
	accumGlobal := []int{size}
	waveGlobal := s.waveGlobal
	if len(waveGlobal) != 2 {
		waveGlobal = []int{s.width, s.height}
	}
	waveLocal := s.waveLocal
	if len(waveLocal) != 0 && len(waveLocal) != len(waveGlobal) {
		waveLocal = nil
	}
	scale := float32(1)
	if steps > 0 {
		scale = 1 / float32(steps)
	}
	if steps > 0 && s.sampleKernel != nil && s.centerSampleBuf != nil {
		s.lastSampleCount = steps
		s.hostCenterSamples = ensureFloat32Slice(s.hostCenterSamples, steps)
		if s.useFP16 {
			s.hostCenterSamplesHalf = ensureUint16Slice(s.hostCenterSamplesHalf, steps)
		}
	} else {
		s.lastSampleCount = 0
		s.centerSample = 0
	}
	reflect32 := float32(boundaryReflect)
	if err := s.setEmitterArgs(emitterIndex, 0); err != nil {
		return fmt.Errorf("setting emitter args: %w", err)
	}
	for step := 0; step < steps; step++ {
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
		if size > 0 {
			if err := s.runBoundaryAccumulate(accumGlobal, scale, reflect32); err != nil {
				return err
			}
			if s.sampleKernel != nil && s.centerSampleBuf != nil {
				if err := s.sampleCenter(step); err != nil {
					return err
				}
			}
		}
	}
	if steps > 0 && s.sampleKernel != nil && s.centerSampleBuf != nil {
		byteLen := steps * s.elementBytes
		if s.useFP16 {
			if _, err := s.queue.EnqueueReadBuffer(s.centerSampleBuf, true, 0, byteLen, unsafe.Pointer(&s.hostCenterSamplesHalf[0]), nil); err != nil {
				return fmt.Errorf("reading center samples (fp16): %w", err)
			}
			float16ToFloat32(s.hostCenterSamples, s.hostCenterSamplesHalf)
		} else {
			if _, err := s.queue.EnqueueReadBufferFloat32(s.centerSampleBuf, true, 0, s.hostCenterSamples[:steps], nil); err != nil {
				return fmt.Errorf("reading center samples: %w", err)
			}
		}
		s.centerSample = s.hostCenterSamples[steps-1]
	} else if steps > 0 && size > 0 && s.width > 0 && s.height > 0 {
		// Fallback: sample a single center value from the current buffer when
		// per-step sampling is disabled.
		cx := s.width / 2
		cy := s.height / 2
		if cx >= 0 && cx < s.width && cy >= 0 && cy < s.height {
			idx := cy*s.width + cx
			offset := idx * s.elementBytes
			if s.useFP16 {
				var raw uint16
				if _, err := s.queue.EnqueueReadBuffer(s.currBuf, true, offset, s.elementBytes, unsafe.Pointer(&raw), nil); err == nil {
					v := float16BitsToFloat32(raw)
					if v > 1 {
						v = 1
					} else if v < -1 {
						v = -1
					}
					s.centerSample = v
				}
			} else {
				var v float32
				if _, err := s.queue.EnqueueReadBuffer(s.currBuf, true, offset, s.elementBytes, unsafe.Pointer(&v), nil); err == nil {
					if v > 1 {
						v = 1
					} else if v < -1 {
						v = -1
					}
					s.centerSample = v
				}
			}
		}
	}
	if err := s.setRenderFlags(showWalls, useVisibility); err != nil {
		return fmt.Errorf("configuring render overlays: %w", err)
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.renderKernel, nil, accumGlobal, nil, nil); err != nil {
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
	if s.debugVerify && size > 0 {
		if _, err := s.queue.EnqueueReadBufferFloat32(s.currBuf, true, 0, field.curr, nil); err != nil {
			return fmt.Errorf("reading current buffer for debug: %w", err)
		}
		if _, err := s.queue.EnqueueReadBufferFloat32(s.prevBuf, true, 0, field.prev, nil); err != nil {
			return fmt.Errorf("reading previous buffer for debug: %w", err)
		}
	}
	// No need to read back next buffer; it will be regenerated on subsequent steps.
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
	if s.accumBuf != nil {
		s.accumBuf.Release()
		s.accumBuf = nil
	}
	if s.visibilityBuf != nil {
		s.visibilityBuf.Release()
		s.visibilityBuf = nil
	}
	if s.wallMaskBuf != nil {
		s.wallMaskBuf.Release()
		s.wallMaskBuf = nil
	}
	if s.impulseValueBuf != nil {
		s.impulseValueBuf.Release()
		s.impulseValueBuf = nil
	}
	if s.impulseIndexBuf != nil {
		s.impulseIndexBuf.Release()
		s.impulseIndexBuf = nil
	}
	if s.centerSampleBuf != nil {
		s.centerSampleBuf.Release()
		s.centerSampleBuf = nil
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
	if s.applyImpulsesKernel != nil {
		s.applyImpulsesKernel.Release()
		s.applyImpulsesKernel = nil
	}
	if s.boundaryAccumKernel != nil {
		s.boundaryAccumKernel.Release()
		s.boundaryAccumKernel = nil
	}
	if s.sampleKernel != nil {
		s.sampleKernel.Release()
		s.sampleKernel = nil
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

func (s *openCLWaveSolver) CenterSample() float32 {
	return s.centerSample
}

func (s *openCLWaveSolver) CenterSamples() []float32 {
	if s.lastSampleCount <= 0 || s.lastSampleCount > len(s.hostCenterSamples) {
		return nil
	}
	return s.hostCenterSamples[:s.lastSampleCount]
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
