#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#   include <OpenCL/cl_ext.h>
#else
#   include <CL/cl.h>
#   include <CL/cl_ext.h>
#endif

/*
 * Some OpenCL headers distributed with GPU drivers omit symbols that were
 * introduced in later 1.2 revisions (e.g. mesa's OpenCL 1.1 headers).  The Go
 * bindings expect these enums to exist, so provide fallback definitions when
 * they are missing in the system headers to keep the build working with a
 * wider range of OpenCL SDKs.
 */
#ifndef CL_UNORM_INT24
#define CL_UNORM_INT24 0x10DF
#endif

#ifndef CL_DEPTH
#define CL_DEPTH 0x10BD
#endif

#ifndef CL_DEPTH_STENCIL
#define CL_DEPTH_STENCIL 0x10BE
#endif
