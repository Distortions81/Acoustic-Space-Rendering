//go:build opencl

package main

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
)

type openCLWaveSolver struct {
	context    *cl.Context
	queue      *cl.CommandQueue
	program    *cl.Program
	kernel     *cl.Kernel
	currBuf    *cl.MemObject
	prevBuf    *cl.MemObject
	nextBuf    *cl.MemObject
	wallBuf    *cl.MemObject
	width      int
	height     int
	wallBytes  []byte
	deviceName string
}

const waveKernelSource = `__kernel void wave_step(
    const int width,
    const int height,
    const float damp,
    const float speed,
    __global const float* curr,
    __global const float* prev,
    __global const uchar* walls,
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
        next_buffer[idx] = 0.0f;
        return;
    }
    if (walls[idx]) {
        next_buffer[idx] = 0.0f;
        return;
    }
    int left = idx - 1;
    int right = idx + 1;
    int top = idx - width;
    int bottom = idx + width;
    float center = curr[idx];
    float laplacian = curr[left] + curr[right] + curr[top] + curr[bottom] - 4.0f * center;
    next_buffer[idx] = ((2.0f * center - prev[idx]) + speed * laplacian) * damp;
}`

func newOpenCLWaveSolver(width, height int) (*openCLWaveSolver, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, fmt.Errorf("querying OpenCL platforms: %w", err)
	}
	if len(platforms) == 0 {
		return nil, errors.New("no OpenCL platforms available")
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
		buildLog, _ := program.GetProgramBuildLog(device)
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("building OpenCL program: %w\n%s", err, buildLog)
	}
	kernel, err := program.CreateKernel("wave_step")
	if err != nil {
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("creating OpenCL kernel: %w", err)
	}
	size := width * height
	byteSize := size * int(unsafe.Sizeof(float32(0)))
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating current buffer: %w", err)
	}
	prevBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		currBuf.Release()
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
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating next buffer: %w", err)
	}
	wallBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		kernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall buffer: %w", err)
	}

	solver := &openCLWaveSolver{
		context:    context,
		queue:      queue,
		program:    program,
		kernel:     kernel,
		currBuf:    currBuf,
		prevBuf:    prevBuf,
		nextBuf:    nextBuf,
		wallBuf:    wallBuf,
		width:      width,
		height:     height,
		wallBytes:  make([]byte, size),
		deviceName: device.Name(),
	}

	if err := solver.kernel.SetArgs(
		int32(width),
		int32(height),
		waveDamp32,
		waveSpeed32,
		solver.currBuf,
		solver.prevBuf,
		solver.wallBuf,
		solver.nextBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel arguments: %w", err)
	}

	return solver, nil
}

func (s *openCLWaveSolver) ensureWallBytes(walls []bool) []byte {
	size := s.width * s.height
	if len(walls) != size {
		for i := range s.wallBytes {
			s.wallBytes[i] = 0
		}
		return s.wallBytes
	}
	for i, w := range walls {
		if w {
			s.wallBytes[i] = 1
		} else {
			s.wallBytes[i] = 0
		}
	}
	return s.wallBytes
}

func (s *openCLWaveSolver) Step(field *waveField, walls []bool) error {
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return fmt.Errorf("unexpected field buffer size")
	}
	if _, err := s.queue.EnqueueWriteBufferFloat32(s.currBuf, true, 0, field.curr, nil); err != nil {
		return fmt.Errorf("writing current buffer: %w", err)
	}
	if _, err := s.queue.EnqueueWriteBufferFloat32(s.prevBuf, true, 0, field.prev, nil); err != nil {
		return fmt.Errorf("writing previous buffer: %w", err)
	}
	wallsBytes := s.ensureWallBytes(walls)
	if len(wallsBytes) > 0 {
		ptr := unsafe.Pointer(&wallsBytes[0])
		if _, err := s.queue.EnqueueWriteBuffer(s.wallBuf, true, 0, len(wallsBytes), ptr, nil); err != nil {
			return fmt.Errorf("writing wall buffer: %w", err)
		}
	}
	global := []int{size}
	if _, err := s.queue.EnqueueNDRangeKernel(s.kernel, nil, global, nil, nil); err != nil {
		return fmt.Errorf("enqueueing kernel: %w", err)
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.nextBuf, true, 0, field.next, nil); err != nil {
		return fmt.Errorf("reading next buffer: %w", err)
	}
	return nil
}

func (s *openCLWaveSolver) Close() {
	if s.wallBuf != nil {
		s.wallBuf.Release()
		s.wallBuf = nil
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
