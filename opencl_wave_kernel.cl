#ifdef USE_FP16
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
    const real_t damp_r,
    const real_t speed_r,
    const real_t two,
    const real_t four,
    __global const real_t* curr,
    __global const real_t* prev,
    __global real_t* next_buffer,
    __global const uchar* wall_mask,
    __global const uchar* block_mask)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;
    if (!block_mask[idx]) {
        next_buffer[idx] = curr[idx];
        return;
    }
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
    real_t center = curr[idx];
    real_t laplacian = curr[left] + curr[right] + curr[top] + curr[bottom] - four * center;
    next_buffer[idx] = ((two * center - prev[idx]) + speed_r * laplacian) * damp_r;
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
