//go:build opencl

package main

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
)

type openCLWaveSolver struct {
	context           *cl.Context
	queue             *cl.CommandQueue
	program           *cl.Program
	kernel            *cl.Kernel
	zeroWallsKernel   *cl.Kernel
	boundaryRowKernel *cl.Kernel
	boundaryColKernel *cl.Kernel
	currBuf           *cl.MemObject
	prevBuf           *cl.MemObject
	nextBuf           *cl.MemObject
	wallIndexBuf      *cl.MemObject
	width             int
	height            int
	wallIndices       []int32
	wallCount         int
	wallsSynced       bool
	deviceName        string
	sampleIndex       int
	sampleOffset      int
	sampleBuffer      []int16
	sampleScratch     []float32
	sampleDeviceBuf   *cl.MemObject
	sampleDeviceCap   int
}

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
}`

func newOpenCLWaveSolver(width, height int, sampleIndex int) (*openCLWaveSolver, error) {
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
	zeroWallsKernel, err := program.CreateKernel("clear_walls")
	if err != nil {
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating wall kernel: %w", err)
	}
	boundaryRowKernel, err := program.CreateKernel("apply_boundary_rows")
	if err != nil {
		zeroWallsKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating boundary row kernel: %w", err)
	}
	boundaryColKernel, err := program.CreateKernel("apply_boundary_cols")
	if err != nil {
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating boundary column kernel: %w", err)
	}
	size := width * height
	if sampleIndex < 0 || sampleIndex >= size {
		sampleIndex = defaultPressureSampleIndex(width, height)
	}
	sampleOffset := sampleIndex * int(unsafe.Sizeof(float32(0)))
	byteSize := size * int(unsafe.Sizeof(float32(0)))
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
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
		kernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating next buffer: %w", err)
	}
	wallIndexBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size*int(unsafe.Sizeof(int32(0))))
	if err != nil {
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		kernel.Release()
		boundaryColKernel.Release()
		boundaryRowKernel.Release()
		zeroWallsKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall index buffer: %w", err)
	}

	solver := &openCLWaveSolver{
		context:           context,
		queue:             queue,
		program:           program,
		kernel:            kernel,
		zeroWallsKernel:   zeroWallsKernel,
		boundaryRowKernel: boundaryRowKernel,
		boundaryColKernel: boundaryColKernel,
		currBuf:           currBuf,
		prevBuf:           prevBuf,
		nextBuf:           nextBuf,
		wallIndexBuf:      wallIndexBuf,
		width:             width,
		height:            height,
		deviceName:        device.Name(),
		sampleIndex:       sampleIndex,
		sampleOffset:      sampleOffset,
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

func (s *openCLWaveSolver) ensureSampleBuffer(steps int) []int16 {
	if steps <= 0 {
		if s.sampleBuffer != nil {
			s.sampleBuffer = s.sampleBuffer[:0]
			return s.sampleBuffer
		}
		return nil
	}
	if cap(s.sampleBuffer) < steps {
		s.sampleBuffer = make([]int16, steps)
	} else {
		s.sampleBuffer = s.sampleBuffer[:steps]
	}
	return s.sampleBuffer
}

func (s *openCLWaveSolver) ensureSampleScratch(steps int) []float32 {
	if steps <= 0 {
		if s.sampleScratch != nil {
			s.sampleScratch = s.sampleScratch[:0]
			return s.sampleScratch
		}
		return nil
	}
	if cap(s.sampleScratch) < steps {
		s.sampleScratch = make([]float32, steps)
	} else {
		s.sampleScratch = s.sampleScratch[:steps]
	}
	return s.sampleScratch
}

func (s *openCLWaveSolver) ensureSampleDeviceBuffer(steps int) error {
	if steps <= 0 {
		return nil
	}
	if steps <= s.sampleDeviceCap {
		return nil
	}
	if s.sampleDeviceBuf != nil {
		s.sampleDeviceBuf.Release()
		s.sampleDeviceBuf = nil
		s.sampleDeviceCap = 0
	}
	byteSize := steps * int(unsafe.Sizeof(float32(0)))
	buf, err := s.context.CreateEmptyBuffer(cl.MemReadWrite, byteSize)
	if err != nil {
		return fmt.Errorf("allocating sample buffer: %w", err)
	}
	s.sampleDeviceBuf = buf
	s.sampleDeviceCap = steps
	return nil
}

func (s *openCLWaveSolver) bindDynamicBuffers() error {
	if err := s.kernel.SetArgBuffer(4, s.currBuf); err != nil {
		return err
	}
	if err := s.kernel.SetArgBuffer(5, s.prevBuf); err != nil {
		return err
	}
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
	return nil
}

func (s *openCLWaveSolver) Step(field *waveField, walls []bool, steps int, wallsDirty bool) ([]int16, error) {
	if steps <= 0 {
		return s.ensureSampleBuffer(0), nil
	}
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return nil, fmt.Errorf("unexpected field buffer size")
	}
	if _, err := s.queue.EnqueueWriteBufferFloat32(s.currBuf, false, 0, field.curr, nil); err != nil {
		return nil, fmt.Errorf("writing current buffer: %w", err)
	}
	if _, err := s.queue.EnqueueWriteBufferFloat32(s.prevBuf, false, 0, field.prev, nil); err != nil {
		return nil, fmt.Errorf("writing previous buffer: %w", err)
	}
	if !s.wallsSynced || wallsDirty {
		indices := s.ensureWallIndices(walls)
		s.wallCount = len(indices)
		if s.wallCount > 0 {
			ptr := unsafe.Pointer(&indices[0])
			byteLen := len(indices) * int(unsafe.Sizeof(int32(0)))
			if _, err := s.queue.EnqueueWriteBuffer(s.wallIndexBuf, false, 0, byteLen, ptr, nil); err != nil {
				return nil, fmt.Errorf("writing wall index buffer: %w", err)
			}
		} else {
			s.wallCount = 0
		}
		if err := s.zeroWallsKernel.SetArgInt32(2, int32(s.wallCount)); err != nil {
			return nil, fmt.Errorf("setting wall count: %w", err)
		}
		s.wallsSynced = true
	} else {
		if err := s.zeroWallsKernel.SetArgInt32(2, int32(s.wallCount)); err != nil {
			return nil, fmt.Errorf("refreshing wall count: %w", err)
		}
	}
	if err := s.bindDynamicBuffers(); err != nil {
		return nil, fmt.Errorf("binding buffers: %w", err)
	}
	global := []int{size}
	samples := s.ensureSampleBuffer(steps)
	if err := s.ensureSampleDeviceBuffer(steps); err != nil {
		return nil, err
	}
	floatScratch := s.ensureSampleScratch(steps)
	floatSize := int(unsafe.Sizeof(float32(0)))
	for step := 0; step < steps; step++ {
		if _, err := s.queue.EnqueueNDRangeKernel(s.kernel, nil, global, nil, nil); err != nil {
			return nil, fmt.Errorf("enqueueing kernel: %w", err)
		}
		if s.wallCount > 0 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.zeroWallsKernel, nil, []int{s.wallCount}, nil, nil); err != nil {
				return nil, fmt.Errorf("clearing walls: %w", err)
			}
		}
		if s.height > 1 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.boundaryRowKernel, nil, []int{s.width}, nil, nil); err != nil {
				return nil, fmt.Errorf("applying boundary rows: %w", err)
			}
		}
		if s.height > 2 {
			if _, err := s.queue.EnqueueNDRangeKernel(s.boundaryColKernel, nil, []int{s.height - 2}, nil, nil); err != nil {
				return nil, fmt.Errorf("applying boundary columns: %w", err)
			}
		}
		s.prevBuf, s.currBuf, s.nextBuf = s.currBuf, s.nextBuf, s.prevBuf
		if err := s.bindDynamicBuffers(); err != nil {
			return nil, fmt.Errorf("updating buffers: %w", err)
		}
		if _, err := s.queue.EnqueueCopyBuffer(s.currBuf, s.sampleDeviceBuf, s.sampleOffset, step*floatSize, floatSize, nil); err != nil {
			return nil, fmt.Errorf("copying sample value: %w", err)
		}
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.sampleDeviceBuf, true, 0, floatScratch, nil); err != nil {
		return nil, fmt.Errorf("reading sample buffer: %w", err)
	}
	for i := range samples {
		samples[i] = floatToPCM16(floatScratch[i])
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.currBuf, true, 0, field.curr, nil); err != nil {
		return nil, fmt.Errorf("reading current buffer: %w", err)
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.prevBuf, true, 0, field.prev, nil); err != nil {
		return nil, fmt.Errorf("reading previous buffer: %w", err)
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.nextBuf, true, 0, field.next, nil); err != nil {
		return nil, fmt.Errorf("reading next buffer: %w", err)
	}
	return samples, nil
}

func (s *openCLWaveSolver) Close() {
	if s.wallIndexBuf != nil {
		s.wallIndexBuf.Release()
		s.wallIndexBuf = nil
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
	if s.zeroWallsKernel != nil {
		s.zeroWallsKernel.Release()
		s.zeroWallsKernel = nil
	}
	if s.boundaryRowKernel != nil {
		s.boundaryRowKernel.Release()
		s.boundaryRowKernel = nil
	}
	if s.boundaryColKernel != nil {
		s.boundaryColKernel.Release()
		s.boundaryColKernel = nil
	}
	if s.sampleDeviceBuf != nil {
		s.sampleDeviceBuf.Release()
		s.sampleDeviceBuf = nil
		s.sampleDeviceCap = 0
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
