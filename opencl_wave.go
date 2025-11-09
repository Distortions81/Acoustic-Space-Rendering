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
	context      *cl.Context
	queue        *cl.CommandQueue
	program      *cl.Program
	kernel       *cl.Kernel
	renderKernel *cl.Kernel
	currBuf      *cl.MemObject
	prevBuf      *cl.MemObject
	nextBuf      *cl.MemObject
	pixelBuf     *cl.MemObject
	wallMaskBuf  *cl.MemObject
	width        int
	height       int
	wallMask     []uint8
	wallsSynced  bool
	deviceName   string
	coldStart    bool
	boundCurr    *cl.MemObject
	boundPrev    *cl.MemObject
	boundNext    *cl.MemObject
	hostPixels   []byte
	debugVerify  bool
	debugScratch []float32
}

const verifyTolerance = 1e-4

const waveKernelSource = `__kernel void wave_step(
    const int width,
    const int height,
    const float damp,
    const float speed,
    const float reflect,
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
    if (walls[idx]) {
        next_buffer[idx] = 0.0f;
        return;
    }
    int x = idx % width;
    int y = idx / width;
    float center = curr[idx];
    int last_col = width - 1;
    int last_row = height - 1;
    float left;
    if (x > 0) {
        left = curr[idx - 1];
    } else if (width > 1) {
        left = -curr[idx + 1] * reflect;
    } else {
        left = -center * reflect;
    }
    float right;
    if (x < last_col) {
        right = curr[idx + 1];
    } else if (width > 1) {
        right = -curr[idx - 1] * reflect;
    } else {
        right = -center * reflect;
    }
    float top;
    if (y > 0) {
        top = curr[idx - width];
    } else if (height > 1) {
        top = -curr[idx + width] * reflect;
    } else {
        top = -center * reflect;
    }
    float bottom;
    if (y < last_row) {
        bottom = curr[idx + width];
    } else if (height > 1) {
        bottom = -curr[idx - width] * reflect;
    } else {
        bottom = -center * reflect;
    }
    float laplacian = left + right + top + bottom - 4.0f * center;
    float value = ((2.0f * center - prev[idx]) + speed * laplacian) * damp;
    if (x == 0) {
        int mirror = (width > 1) ? idx + 1 : idx;
        value = -curr[mirror] * reflect;
    } else if (x == last_col) {
        int mirror = (width > 1) ? idx - 1 : idx;
        value = -curr[mirror] * reflect;
    }
    if (y == 0) {
        int mirror = (height > 1) ? idx + width : idx;
        value = -curr[mirror] * reflect;
    } else if (y == last_row) {
        int mirror = (height > 1) ? idx - width : idx;
        value = -curr[mirror] * reflect;
    }
    next_buffer[idx] = value;
}

__kernel void render_intensity(
    const int width,
    const int height,
    __global const float* curr,
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
    pixels[idx] = (uchar4)(intensity, intensity, intensity, (uchar)255);
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
	size := width * height
	byteSize := size * int(unsafe.Sizeof(float32(0)))
	currBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		kernel.Release()
		renderKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating current buffer: %w", err)
	}
	prevBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, byteSize)
	if err != nil {
		currBuf.Release()
		kernel.Release()
		renderKernel.Release()
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
		renderKernel.Release()
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
		kernel.Release()
		renderKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating pixel buffer: %w", err)
	}
	wallMaskBuf, err := context.CreateEmptyBuffer(cl.MemReadOnly, size)
	if err != nil {
		nextBuf.Release()
		prevBuf.Release()
		currBuf.Release()
		pixelBuf.Release()
		kernel.Release()
		renderKernel.Release()
		program.Release()
		queue.Release()
		context.Release()
		return nil, fmt.Errorf("allocating wall mask buffer: %w", err)
	}

	solver := &openCLWaveSolver{
		context:      context,
		queue:        queue,
		program:      program,
		kernel:       kernel,
		renderKernel: renderKernel,
		currBuf:      currBuf,
		prevBuf:      prevBuf,
		nextBuf:      nextBuf,
		pixelBuf:     pixelBuf,
		wallMaskBuf:  wallMaskBuf,
		width:        width,
		height:       height,
		deviceName:   device.Name(),
		coldStart:    true,
		hostPixels:   make([]byte, size*4),
		debugVerify:  verifyOpenCLSyncFlag != nil && *verifyOpenCLSyncFlag,
	}

	if err := solver.kernel.SetArgs(
		int32(width),
		int32(height),
		waveDamp32,
		waveSpeed32,
		float32(boundaryReflect),
		solver.currBuf,
		solver.prevBuf,
		solver.wallMaskBuf,
		solver.nextBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting kernel arguments: %w", err)
	}
	if err := solver.renderKernel.SetArgs(
		int32(width),
		int32(height),
		solver.currBuf,
		solver.pixelBuf,
	); err != nil {
		solver.Close()
		return nil, fmt.Errorf("setting render kernel arguments: %w", err)
	}

	return solver, nil
}

func (s *openCLWaveSolver) ensureWallMask(walls []bool) []uint8 {
	size := s.width * s.height
	if cap(s.wallMask) < size {
		s.wallMask = make([]uint8, size)
	}
	mask := s.wallMask[:size]
	if len(walls) != size {
		for i := range mask {
			mask[i] = 0
		}
		s.wallMask = mask
		return mask
	}
	for i, w := range walls {
		if w {
			mask[i] = 1
		} else {
			mask[i] = 0
		}
	}
	s.wallMask = mask
	return mask
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

func (s *openCLWaveSolver) bindDynamicBuffers() error {
	if s.boundCurr != s.currBuf {
		if err := s.kernel.SetArgBuffer(5, s.currBuf); err != nil {
			return err
		}
		if err := s.renderKernel.SetArgBuffer(2, s.currBuf); err != nil {
			return err
		}
		s.boundCurr = s.currBuf
	}
	if s.boundPrev != s.prevBuf {
		if err := s.kernel.SetArgBuffer(6, s.prevBuf); err != nil {
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

func (s *openCLWaveSolver) Step(field *waveField, walls []bool, steps int, wallsDirty bool) error {
	if steps <= 0 {
		return nil
	}
	size := s.width * s.height
	if len(field.curr) != size || len(field.prev) != size || len(field.next) != size {
		return fmt.Errorf("unexpected field buffer size")
	}
	currDirty := field.currWasModified()
	skipCurrUpload := !currDirty
	if currDirty {
		if _, err := s.queue.EnqueueWriteBufferFloat32(s.currBuf, false, 0, field.curr, nil); err != nil {
			return fmt.Errorf("writing current buffer: %w", err)
		}
		field.clearCurrDirty()
	} else if s.debugVerify {
		if err := s.verifyBufferMatchesSlice(s.currBuf, field.curr, "pre-step curr"); err != nil {
			return err
		}
	}
	// Avoid re-uploading previous buffer after the initial frame; device keeps it updated.
	if s.coldStart {
		if _, err := s.queue.EnqueueWriteBufferFloat32(s.prevBuf, false, 0, field.prev, nil); err != nil {
			return fmt.Errorf("writing previous buffer: %w", err)
		}
	}
	if !s.wallsSynced || wallsDirty {
		mask := s.ensureWallMask(walls)
		if len(mask) > 0 {
			ptr := unsafe.Pointer(&mask[0])
			if _, err := s.queue.EnqueueWriteBuffer(s.wallMaskBuf, false, 0, len(mask), ptr, nil); err != nil {
				return fmt.Errorf("writing wall mask buffer: %w", err)
			}
		}
		s.wallsSynced = true
	}
	global := []int{size}
	didSwap := false
	for step := 0; step < steps; step++ {
		if err := s.bindDynamicBuffers(); err != nil {
			return fmt.Errorf("binding buffers: %w", err)
		}
		if _, err := s.queue.EnqueueNDRangeKernel(s.kernel, nil, global, nil, nil); err != nil {
			return fmt.Errorf("enqueueing kernel: %w", err)
		}
		s.prevBuf, s.currBuf, s.nextBuf = s.currBuf, s.nextBuf, s.prevBuf
		didSwap = true
	}
	if _, err := s.queue.EnqueueNDRangeKernel(s.renderKernel, nil, global, nil, nil); err != nil {
		return fmt.Errorf("enqueueing render kernel: %w", err)
	}
	if size > 0 {
		if _, err := s.queue.EnqueueReadBuffer(s.pixelBuf, true, 0, len(s.hostPixels), unsafe.Pointer(&s.hostPixels[0]), nil); err != nil {
			return fmt.Errorf("reading pixel buffer: %w", err)
		}
	}
	if _, err := s.queue.EnqueueReadBufferFloat32(s.currBuf, true, 0, field.curr, nil); err != nil {
		return fmt.Errorf("reading current buffer: %w", err)
	}
	if didSwap {
		if _, err := s.queue.EnqueueReadBufferFloat32(s.prevBuf, true, 0, field.prev, nil); err != nil {
			return fmt.Errorf("reading previous buffer: %w", err)
		}
	}
	if skipCurrUpload && s.debugVerify {
		if err := s.verifyBufferMatchesSlice(s.currBuf, field.curr, "post-step curr"); err != nil {
			return err
		}
		if didSwap {
			if err := s.verifyBufferMatchesSlice(s.prevBuf, field.prev, "post-step prev"); err != nil {
				return err
			}
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
	if s.wallMaskBuf != nil {
		s.wallMaskBuf.Release()
		s.wallMaskBuf = nil
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
	return s.hostPixels
}
