// Texture (image2d) weight-path policy, shared by every layer that wraps a
// weight buffer as an image.
//
// Policy is a DENYLIST, mirroring lfm2-5-350m: the image kernels work on
// Adreno and Mali and are known to break only on PowerVR Rogue. On the Vivo
// Y21 (PowerVR GE8320) the image2d-from-buffer wraps are created without error
// but read back wrong (the device even reports CL_DEVICE_IMAGE_PITCH_ALIGNMENT
// as size 0), so the model decoded garbage. Denying textures there routes every
// layer to its existing buffer path. A match can only DISABLE an optimisation:
// worst case slower, never wrong.
//
//   NNOPT_NO_IMAGES=1     force textures OFF on any device
//   NNOPT_FORCE_IMAGES=1  force textures ON even on a denied device
#pragma once

#include <CL/cl.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

inline bool nnopt_textures_allowed(cl_device_id dev) {
    static int cached = -1;
    if (cached >= 0) return cached == 1;

    const char* no  = std::getenv("NNOPT_NO_IMAGES");
    const char* yes = std::getenv("NNOPT_FORCE_IMAGES");
    if (no && no[0] == '1') {
        std::fprintf(stderr, "NNOPT_IMAGES: texture path FORCED OFF (NNOPT_NO_IMAGES)\n");
        cached = 0;
        return false;
    }

    char name[256] = {0}, plat[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
    cl_platform_id pid = nullptr;
    if (clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(pid), &pid, nullptr) == CL_SUCCESS && pid)
        clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(plat) - 1, plat, nullptr);
    for (char* p = name; *p; ++p) *p = (char)std::tolower((unsigned char)*p);
    for (char* p = plat; *p; ++p) *p = (char)std::tolower((unsigned char)*p);

    // "powervr ge": CL_DEVICE_NAME on the Vivo Y21 is exactly "PowerVR GE8320".
    // Not "powervr" alone: Tensor G5's PowerVR DXT is a different architecture.
    static const char* kDeny[] = {"powervr ge", "rogue"};
    const char* hit = nullptr;
    for (const char* pat : kDeny)
        if (std::strstr(name, pat) || std::strstr(plat, pat)) { hit = pat; break; }

    const bool forced_on = yes && yes[0] == '1';
    const bool allowed = !hit || forced_on;
    std::fprintf(stderr, "NNOPT_IMAGES: texture path %s (device \"%s\"%s%s)\n",
                 allowed ? "ON" : "OFF -> buffer kernels", name,
                 hit ? ", denylist match " : "", hit ? hit : "");
    cached = allowed ? 1 : 0;
    return allowed;
}
