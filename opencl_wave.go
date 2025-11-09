package main

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
)

type openCLWaveSolver struct {
	context                 *cl.Context
	queue                   *cl.CommandQueue
	program                 *cl.Program
	kernel                  *cl.Kernel
	renderKernel            *cl.Kernel
	zeroWallsKernel         *cl.Kernel
	applyImpulsesKernel     *cl.Kernel
	boundaryRowKernel       *cl.Kernel
	boundaryColKernel       *cl.Kernel
	currBuf                 *cl.MemObject
	prevBuf                 *cl.MemObject
	nextBuf                 *cl.MemObject
	pixelBuf                *cl.MemObject
	wallMaskBuf             *cl.MemObject
	visibilityBuf           *cl.MemObject
	wallIndexBuf            *cl.MemObject
	impulseIndexBuf         *cl.MemObject
	impulseValueBuf         *cl.MemObject
	width                   int
	height                  int
	wallIndices             []int32
	wallCount               int
	wallsSynced             bool
	wallMaskSynced          bool
	deviceName              string
	coldStart               bool
	boundCurr               *cl.MemObject
	boundPrev               *cl.MemObject
	boundNext               *cl.MemObject
	hostPixels              []byte
	hostWallMask            []byte
	hostVisibility          []byte
	uploadedVisibleGen      uint32
	visibleMaskSynced       bool
	lastRenderShowWalls     int32
	lastRenderUseVisibility int32
	debugVerify             bool
	debugScratch            []float32
	impulseCurrIndices      []int32
	impulseCurrValues       []float32
	impulsePrevIndices      []int32
	impulsePrevValues       []float32
}

const verifyTolerance = 1e-4

const waveKernelSource = `__kernel void wave_step(
    const int width,
    const int height,
    const float damp,
    const float speed,
    __global const float* curr,
    __global const float* prev,
    __global float* next_buffer)
{
    int idx = get_global_id(0);
    int size = width * height;
    if (idx >= size) {
        return;
    }
    int x = idx % width;
    int y = idx / width;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return;
    }
    int left = idx - 1;
    int right = idx + 1;
    int top = idx - width;
    int bottom = idx + width;
    float center = curr[idx];
    float laplacian = curr[left] + curr[right] + curr[top] + curr[bottom] - 4.0f * center;
    next_buffer[idx] = ((2.0f * center - prev[idx]) + speed * laplacian) * damp;
}

__kernel void clear_walls(
    __global float* buffer,
    __global const int* wall_indices,
    const int wall_count)
{
    int gid = get_global_id(0);
    if (gid >= wall_count) {
        return;
    }
    int idx = wall_indices[gid];
    buffer[idx] = 0.0f;
}

__kernel void apply_impulses(
    const int count,
    __global const int* indices,
    __global const float* values,
    __global float* buffer)
{
    int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    int idx = indices[gid];
    buffer[idx] = values[gid];
}

__kernel void apply_boundary_rows(
    const int width,
    const int height,
    const float reflect,
    __global float* buffer)
{
    int x = get_global_id(0);
    if (x >= width) {
        return;
    }
    int last_row = height - 1;
    int top_idx = x;
    int bottom_idx = last_row * width + x;
    int top_src = width + x;
    int bottom_src = (last_row - 1) * width + x;
    buffer[top_idx] = -buffer[top_src] * reflect;
    buffer[bottom_idx] = -buffer[bottom_src] * reflect;
}

__kernel void apply_boundary_cols(
    const int width,
    const int height,
    const float reflect,
    __global float* buffer)
{
    int y = get_global_id(0) + 1;
    int last_row = height - 1;
    if (y >= last_row) {
        return;
    }
    int base = y * width;
    int left_idx = base;
    int right_idx = base + width - 1;
    buffer[left_idx] = -buffer[left_idx + 1] * reflect;
    buffer[right_idx] = -buffer[right_idx - 1] * reflect;
}

__kernel void render_intensity(
    const int width,
    const int height,
    __global const float* curr,
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
    float value = curr[idx];
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
}`

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
	if err := program.BuildProgram([]*cl.Device{device}, ""); err != nil {
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
	zeroWallsKernel, err := program.CreateKernel("clear_walls")
	if err != nil {
		kernel.Release()
		renderKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating wall kernel: %w", err)
	}
	applyImpulsesKernel, err := program.CreateKernel("apply_impulses")
	if err != nil {
		zeroWallsKernel.Release()
		renderKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating impulse kernel: %w", err)
	}
	boundaryRowKernel, err := program.CreateKernel("apply_boundary_rows")
	if err != nil {
		applyImpulsesKernel.Release()
		zeroWallsKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating boundary row kernel: %w", err)
	}
	boundaryColKernel, err := program.CreateKernel("apply_boundary_cols")
	if err != nil {
		applyImpulsesKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		kernel.Release()
		renderKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating boundary column kernel: %w", err)
	}
	size := width * height
	byteSize := size * int(unsafe.Sizeof(float32(0)))
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		applyImpulsesKernel.Release()
		kernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating current buffer: %w", err)
	}
	prevBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		currBuf.Release()
		applyImpulsesKernel.Release()
		kernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
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
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
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
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
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
		applyImpulsesKernel.Release()
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall mask buffer: %w", err)
	}
	visibilityBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		wallMaskBuf.Release()
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating visibility buffer: %w", err)
	}
	wallIndexBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*int(unsafe.Sizeof(int32(0))))
	if err != nil {
		visibilityBuf.Release()
		wallMaskBuf.Release()
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall index buffer: %w", err)
	}
	impulseIndexBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*int(unsafe.Sizeof(int32(0))))
	if err != nil {
		wallIndexBuf.Release()
		visibilityBuf.Release()
		wallMaskBuf.Release()
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating impulse index buffer: %w", err)
	}
	impulseValueBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*int(unsafe.Sizeof(float32(0))))
	if err != nil {
		impulseIndexBuf.Release()
		wallIndexBuf.Release()
		visibilityBuf.Release()
		wallMaskBuf.Release()
		pixelBuf.Release()
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		applyImpulsesKernel.Release()
		kernel.Release()
		renderKernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating impulse value buffer: %w", err)
	}

	solver := &openCLWaveSolver{
		context:                 context,
		queue:                   queue,
		program:                 program,
		kernel:                  kernel,
		renderKernel:            renderKernel,
		zeroWallsKernel:         zeroWallsKernel,
		applyImpulsesKernel:     applyImpulsesKernel,
		boundaryRowKernel:       boundaryRowKernel,
		boundaryColKernel:       boundaryColKernel,
		currBuf:                 currBuf,
		prevBuf:                 prevBuf,
		nextBuf:                 nextBuf,
		pixelBuf:                pixelBuf,
		wallMaskBuf:             wallMaskBuf,
		visibilityBuf:           visibilityBuf,
		wallIndexBuf:            wallIndexBuf,
		impulseIndexBuf:         impulseIndexBuf,
		impulseValueBuf:         impulseValueBuf,
		width:                   width,
		height:                  height,
		deviceName:              device.Name(),
		coldStart:               true,
		hostPixels:              make([]byte, size*4),
		hostWallMask:            make([]byte, size),
		hostVisibility:          make([]byte, size),
		lastRenderShowWalls:     -1,
		lastRenderUseVisibility: -1,
		debugVerify:             verifyOpenCLSyncFlag != nil && *verifyOpenCLSyncFlag,
	}

	if err := solver.kernel.SetArgs(
		int32(width),
		int32(height),
		waveDamp32,
		waveSpeed32,
		solver.currBuf,
		solver.prevBuf,
		solver.nextBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel arguments: %w", err)
	}
	if err := solver.renderKernel.SetArgs(
		int32(width),
		int32(height),
		solver.currBuf,
		int32(0),
		solver.wallMaskBuf,
		int32(0),
		solver.visibilityBuf,
		solver.pixelBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting render kernel arguments: %w", err)
	}
	if err := solver.zeroWallsKernel.SetArgs(
		solver.nextBuf,
		solver.wallIndexBuf,
		int32(0),
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting wall kernel arguments: %w", err)
	}
	reflect32 := float32(boundaryReflect)
	if err := solver.boundaryRowKernel.SetArgs(
		int32(width),
		int32(height),
		reflect32,
		solver.nextBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting boundary row kernel arguments: %w", err)
	}
	if err := solver.boundaryColKernel.SetArgs(
		int32(width),
		int32(height),
		reflect32,
		solver.nextBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting boundary column kernel arguments: %w", err)
	}

	return solver, nil
}

func (s *openCLWaveSolver) ensureWallIndices(walls []bool) []int32 {
	size := s.width * s.height
	if len(walls) != size {
		s.wallIndices = s.wallIndices[:0]
		return s.wallIndices
	}
	if cap(s.wallIndices) < size {
		s.wallIndices = make([]int32, 0, size)
	} else {
		s.wallIndices = s.wallIndices[:0]
	}
	for i, w := range walls {
		if w {
			s.wallIndices = append(s.wallIndices, int32(i))
		}
	}
	return s.wallIndices
}

// audio sampling helpers removed

func (s *openCLWaveSolver) ensureDebugScratch(size int) []float32 {
	if cap(s.debugScratch) < size {
		s.debugScratch = make([]float32, size)
	}
	s.debugScratch = s.debugScratch[:size]
	return s.debugScratch
}

func (s *openCLWaveSolver) verifyBufferMatchesSlice(buf *cl.MemObject, host []float32, label string) error {
	if len(host) == 0 {
		return nil
	}
	scratch := s.ensureDebugScratch(len(host))
	if _, err := s.queue.EnqueueReadBufferFloat32(buf, true, 0, scratch, nil); err != nil {
		return fmt.Errorf("reading %s for verification: %w", label, err)
	}
	for i, hv := range host {
		if diff := math.Abs(float64(scratch[i] - hv)); diff > verifyTolerance {
			return fmt.Errorf("%s mismatch at index %d: device=%f host=%f diff=%f", label, i, scratch[i], hv, diff)
		}
	}
	return nil
}

func (s *openCLWaveSolver) dispatchImpulses(target *cl.MemObject, indices []int32, values []float32) error {
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
	byteVal := len(values) * int(unsafe.Sizeof(float32(0)))
	if _, err := s.queue.EnqueueWriteBuffer(s.impulseValueBuf, false, 0, byteVal, unsafe.Pointer(&values[0]), nil); err != nil {
		return fmt.Errorf("uploading impulse values: %w", err)
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

func (s *openCLWaveSolver) applyQueuedImpulses(field *waveField) error {
	impulses := field.takeImpulses()
	if len(impulses) == 0 {
		return nil
	}
	s.impulseCurrIndices = s.impulseCurrIndices[:0]
	s.impulseCurrValues = s.impulseCurrValues[:0]
	s.impulsePrevIndices = s.impulsePrevIndices[:0]
	s.impulsePrevValues = s.impulsePrevValues[:0]
	if cap(s.impulseCurrIndices) < len(impulses) {
		s.impulseCurrIndices = make([]int32, 0, len(impulses))
	}
	if cap(s.impulseCurrValues) < len(impulses) {
		s.impulseCurrValues = make([]float32, 0, len(impulses))
	}
	if cap(s.impulsePrevIndices) < len(impulses) {
		s.impulsePrevIndices = make([]int32, 0, len(impulses))
	}
	if cap(s.impulsePrevValues) < len(impulses) {
		s.impulsePrevValues = make([]float32, 0, len(impulses))
	}
	for _, imp := range impulses {
		s.impulseCurrIndices = append(s.impulseCurrIndices, imp.index)
		s.impulseCurrValues = append(s.impulseCurrValues, imp.value)
		if imp.applyPrev {
			s.impulsePrevIndices = append(s.impulsePrevIndices, imp.index)
			s.impulsePrevValues = append(s.impulsePrevValues, imp.value)
		}
	}
	if err := s.dispatchImpulses(s.currBuf, s.impulseCurrIndices, s.impulseCurrValues); err != nil {
		return err
	}
	if err := s.dispatchImpulses(s.prevBuf, s.impulsePrevIndices, s.impulsePrevValues); err != nil {
		return err
	}
	return nil
}

func (s *openCLWaveSolver) bindDynamicBuffers() error {
	if s.boundCurr != s.currBuf {
		if err := s.kernel.SetArgBuffer(4, s.currBuf); err != nil {
			return err
		}
		if err := s.renderKernel.SetArgBuffer(2, s.currBuf); err != nil {
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
		if err := s.zeroWallsKernel.SetArgBuffer(0, s.nextBuf); err != nil {
			return err
		}
		if err := s.boundaryRowKernel.SetArgBuffer(3, s.nextBuf); err != nil {
			return err
		}
		if err := s.boundaryColKernel.SetArgBuffer(3, s.nextBuf); err != nil {
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

func (s *openCLWaveSolver) Step(field *waveField, walls []bool, steps int, wallsDirty bool, showWalls bool, occludeLOS bool, visibleStamp []uint32, visibleGen uint32) error {
	if steps <= 0 {
		return nil
	}
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return fmt.Errorf("unexpected field buffer size")
	}
	if s.coldStart && size > 0 {
		if _, err := s.queue.EnqueueWriteBufferFloat32(s.currBuf, false, 0, field.curr, nil); err != nil {
			return fmt.Errorf("initializing current buffer: %w", err)
		}
		if _, err := s.queue.EnqueueWriteBufferFloat32(s.prevBuf, false, 0, field.prev, nil); err != nil {
			return fmt.Errorf("initializing previous buffer: %w", err)
		}
		if _, err := s.queue.EnqueueWriteBufferFloat32(s.nextBuf, false, 0, field.next, nil); err != nil {
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
	if !s.wallsSynced || wallsDirty {
		indices := s.ensureWallIndices(walls)
		s.wallCount = len(indices)
		if s.wallCount > 0 {
			ptr := unsafe.Pointer(&indices[0])
			byteLen := len(indices) * int(unsafe.Sizeof(int32(0)))
			if _, err := s.queue.EnqueueWriteBuffer(s.wallIndexBuf, false, 0, byteLen, ptr, nil); err != nil {
				return fmt.Errorf("writing wall index buffer: %w", err)
			}
		} else {
			s.wallCount = 0
		}
		if err := s.zeroWallsKernel.SetArgInt32(2, int32(s.wallCount)); err != nil {
			return fmt.Errorf("setting wall count: %w", err)
		}
		s.wallsSynced = true
	} else {
		if err := s.zeroWallsKernel.SetArgInt32(2, int32(s.wallCount)); err != nil {
			return fmt.Errorf("refreshing wall count: %w", err)
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
	global := []int{size}
	for step := 0; step < steps; step++ {
		if err := s.bindDynamicBuffers(); err != nil {
			return fmt.Errorf("binding buffers: %w", err)
		}
		if _, err := s.queue.EnqueueNDRangeKernel(s.kernel, nil, global, nil, nil); err != nil {
			return fmt.Errorf("enqueueing kernel: %w", err)
		}
		if s.wallCount > 0 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.zeroWallsKernel, nil, []int{s.wallCount}, nil, nil); err != nil {
				return fmt.Errorf("clearing walls: %w", err)
			}
		}
		if s.height > 1 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.boundaryRowKernel, nil, []int{s.width}, nil, nil); err != nil {
				return fmt.Errorf("applying boundary rows: %w", err)
			}
		}
		if s.height > 2 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.boundaryColKernel, nil, []int{s.height - 2}, nil, nil); err != nil {
				return fmt.Errorf("applying boundary columns: %w", err)
			}
		}
		s.prevBuf, s.currBuf, s.nextBuf = s.currBuf, s.nextBuf, s.prevBuf
	}
	if err := s.setRenderFlags(showWalls, useVisibility); err != nil {
		return fmt.Errorf("configuring render overlays: %w", err)
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.renderKernel, nil, global, nil, nil); err != nil {
		return fmt.Errorf("enqueueing render kernel: %w", err)
	}
	if size > 0 {
		if _, err := s.queue.EnqueueReadBuffer(s.pixelBuf, true, 0, len(s.hostPixels), unsafe.Pointer(&s.hostPixels[0]), nil); err != nil {
			return fmt.Errorf("reading pixel buffer: %w", err)
		}
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
	if s.pixelBuf != nil {
		s.pixelBuf.Release()
		s.pixelBuf = nil
	}
	if s.visibilityBuf != nil {
		s.visibilityBuf.Release()
		s.visibilityBuf = nil
	}
	if s.wallMaskBuf != nil {
		s.wallMaskBuf.Release()
		s.wallMaskBuf = nil
	}
	if s.wallIndexBuf != nil {
		s.wallIndexBuf.Release()
		s.wallIndexBuf = nil
	}
	if s.impulseValueBuf != nil {
		s.impulseValueBuf.Release()
		s.impulseValueBuf = nil
	}
	if s.impulseIndexBuf != nil {
		s.impulseIndexBuf.Release()
		s.impulseIndexBuf = nil
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
	if s.zeroWallsKernel != nil {
		s.zeroWallsKernel.Release()
		s.zeroWallsKernel = nil
	}
	if s.applyImpulsesKernel != nil {
		s.applyImpulsesKernel.Release()
		s.applyImpulsesKernel = nil
	}
	if s.boundaryRowKernel != nil {
		s.boundaryRowKernel.Release()
		s.boundaryRowKernel = nil
	}
	if s.boundaryColKernel != nil {
		s.boundaryColKernel.Release()
		s.boundaryColKernel = nil
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
	return s.hostPixels
}
