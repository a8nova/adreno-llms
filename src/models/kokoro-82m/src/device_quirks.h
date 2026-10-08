// Device checks for non-Adreno GPUs (PowerVR Rogue GE8320).
#pragma once

#include <CL/cl.h>
#include <cstring>

// True when the kernel's static __local arrays fit the device. The LDS-cached
// conv kernels declare 8-32 KB caches sized for Adreno; PowerVR GE8320 has 4 KB,
// where the launch fails with CL_OUT_OF_RESOURCES (-5) and the F0/N/noise convs
// silently produced nothing. Callers fall back to the uncached kernel.
inline bool nnopt_kernel_local_fits(cl_kernel k, cl_device_id dev) {
    if (!k || !dev) return false;
    cl_ulong dev_lm = 0, k_lm = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(dev_lm), &dev_lm, nullptr) != CL_SUCCESS)
        return true;
    if (clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(k_lm), &k_lm, nullptr) != CL_SUCCESS)
        return true;
    return k_lm <= dev_lm;
}

// PowerVR Rogue (Vivo Y21's GE8320). CL_DEVICE_NAME there is "PowerVR GE8320";
// the newer PowerVR DXT (Tensor G5) is a different architecture and is NOT matched.
inline bool nnopt_is_powervr_rogue(cl_device_id dev) {
    char name[128] = {0};
    if (!dev || clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr) != CL_SUCCESS)
        return false;
    for (char* p = name; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p = (char)(*p - 'A' + 'a');
    return std::strstr(name, "powervr ge") || std::strstr(name, "rogue");
}
