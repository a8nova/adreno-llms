// One-shot device probe: validates cl_qcom_recordable_queues is present and
// the extension function pointers actually load. Before refactoring the
// vocoder dispatch path we want a hard go/no-go.
//
// Build (host cross to Android): linked alongside the main inference binary
// via add_executable in CMakeLists.txt and pushed via deploy_android.sh.
// Run:   ./probe_recordable
// Exits 0 on success, 1 on failure (extension absent or function ptr null).

#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define NNOPT_CL_QUEUE_RECORDABLE_QCOM (1 << 5)

typedef struct _cl_recording_qcom* cl_recording_qcom;

// Spec (Qualcomm 80-NB295-11): clNewRecordingQCOM takes ONLY the queue and
// returns the recording handle. No errcode_ret slot.
typedef cl_recording_qcom (CL_API_CALL *PFN_clNewRecordingQCOM)(cl_command_queue);
typedef cl_int (CL_API_CALL *PFN_clEndRecordingQCOM)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *PFN_clReleaseRecordingQCOM)(cl_recording_qcom);

int main() {
  cl_uint n_plat = 0;
  if (clGetPlatformIDs(0, nullptr, &n_plat) != CL_SUCCESS || n_plat == 0) {
    fprintf(stderr, "no platforms\n"); return 1;
  }
  std::vector<cl_platform_id> plats(n_plat);
  clGetPlatformIDs(n_plat, plats.data(), nullptr);
  cl_platform_id plat = plats[0];

  cl_uint n_dev = 0;
  if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_dev) != CL_SUCCESS || n_dev == 0) {
    fprintf(stderr, "no GPU devices\n"); return 1;
  }
  std::vector<cl_device_id> devs(n_dev);
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, n_dev, devs.data(), nullptr);
  cl_device_id dev = devs[0];

  size_t name_sz = 0;
  clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &name_sz);
  std::vector<char> name(name_sz);
  clGetDeviceInfo(dev, CL_DEVICE_NAME, name_sz, name.data(), nullptr);
  printf("Device: %s\n", name.data());

  size_t ext_sz = 0;
  clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_sz);
  std::vector<char> ext(ext_sz);
  clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_sz, ext.data(), nullptr);
  const bool has_rec = std::strstr(ext.data(), "cl_qcom_recordable_queues") != nullptr;
  printf("cl_qcom_recordable_queues in device extensions: %s\n", has_rec ? "YES" : "NO");
  if (!has_rec) return 1;

  // Try ICD-loader name first (clGetExtensionFunctionAddressForPlatform).
  auto fnNew = (PFN_clNewRecordingQCOM)clGetExtensionFunctionAddressForPlatform(
      plat, "clNewRecordingQCOM");
  auto fnEnd = (PFN_clEndRecordingQCOM)clGetExtensionFunctionAddressForPlatform(
      plat, "clEndRecordingQCOM");
  auto fnRel = (PFN_clReleaseRecordingQCOM)clGetExtensionFunctionAddressForPlatform(
      plat, "clReleaseRecordingQCOM");
  printf("via clGetExtensionFunctionAddressForPlatform:\n");
  printf("  clNewRecordingQCOM     -> %p\n", (void*)fnNew);
  printf("  clEndRecordingQCOM     -> %p\n", (void*)fnEnd);
  printf("  clReleaseRecordingQCOM -> %p\n", (void*)fnRel);

  // Fallback: dlsym against the vendor library. Adreno's libOpenCL exports
  // both standard names and `qCLDrvAPI_*` prefixed copies.
  void* h = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
  printf("dlopen(libOpenCL) -> %p (err=%s)\n", h, h ? "" : dlerror());
  if (h) {
    if (!fnNew) fnNew = (PFN_clNewRecordingQCOM)dlsym(h, "clNewRecordingQCOM");
    if (!fnEnd) fnEnd = (PFN_clEndRecordingQCOM)dlsym(h, "clEndRecordingQCOM");
    if (!fnRel) fnRel = (PFN_clReleaseRecordingQCOM)dlsym(h, "clReleaseRecordingQCOM");
    printf("after dlsym(clNewRecordingQCOM)     -> %p\n", (void*)fnNew);
    printf("after dlsym(clEndRecordingQCOM)     -> %p\n", (void*)fnEnd);
    printf("after dlsym(clReleaseRecordingQCOM) -> %p\n", (void*)fnRel);
    if (!fnNew) fnNew = (PFN_clNewRecordingQCOM)dlsym(h, "qCLDrvAPI_clNewRecordingQCOM");
    if (!fnEnd) fnEnd = (PFN_clEndRecordingQCOM)dlsym(h, "qCLDrvAPI_clEndRecordingQCOM");
    if (!fnRel) fnRel = (PFN_clReleaseRecordingQCOM)dlsym(h, "qCLDrvAPI_clReleaseRecordingQCOM");
    printf("after dlsym(qCLDrvAPI_clNewRecordingQCOM)     -> %p\n", (void*)fnNew);
    printf("after dlsym(qCLDrvAPI_clEndRecordingQCOM)     -> %p\n", (void*)fnEnd);
    printf("after dlsym(qCLDrvAPI_clReleaseRecordingQCOM) -> %p\n", (void*)fnRel);
  }
  if (!fnNew || !fnEnd || !fnRel) {
    fprintf(stderr, "function pointer lookup failed\n");
    return 1;
  }

  // Try to actually create a recordable queue.
  cl_int err = CL_SUCCESS;
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "createContext err=%d\n", err); return 1; }

  // Probe for CL_QUEUE_RECORDABLE_QCOM property token. Try OpenCL 2.0 API
  // (clCreateCommandQueueWithProperties) with a range of likely token values.
  // The Qualcomm extension token range is around 0x40A0–0x40C0.
  cl_command_queue q = nullptr;
  cl_command_queue_properties queue_props_bitfield = 0;
  for (cl_uint tok = 0x4090; tok <= 0x40D0; ++tok) {
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, queue_props_bitfield,
        (cl_queue_properties)tok, 1,
        0
    };
    err = CL_SUCCESS;
    q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
    if (err == CL_SUCCESS && q) {
      // Queue was accepted with this token. Now check if it's RECORDABLE
      // by trying clNewRecordingQCOM — it returns NULL on a non-recordable
      // queue.
      cl_recording_qcom rec_try = fnNew(q);
      printf("token 0x%04x: queue=%p, clNewRecordingQCOM -> %p\n",
             tok, (void*)q, (void*)rec_try);
      if (rec_try) {
        fnRel(rec_try);
        printf("  --> token 0x%04x looks like CL_QUEUE_RECORDABLE_QCOM\n", tok);
        break;
      }
      clReleaseCommandQueue(q);
      q = nullptr;
    }
  }
  if (!q) {
    printf("No token in 0x4090..0x40D0 produces a recordable queue.\n");
    // Last resort: try the old OpenCL 1.2 bitfield API with common candidates.
    for (int bit = 0; bit < 64; ++bit) {
      const cl_command_queue_properties p = (cl_command_queue_properties)1ULL << bit;
      err = CL_SUCCESS;
      cl_command_queue qq = clCreateCommandQueue(ctx, dev, p, &err);
      if (err == CL_SUCCESS && qq) {
        cl_recording_qcom rec_try = fnNew(qq);
        if (rec_try) {
          fnRel(rec_try);
          printf("bitfield bit %d (=0x%llx) IS recordable!\n", bit, (unsigned long long)p);
          q = qq; break;
        }
        clReleaseCommandQueue(qq);
      }
    }
  }
  if (!q) { puts("FAILED to create a recordable queue."); return 1; }

  cl_recording_qcom rec = fnNew(q);
  printf("FINAL clNewRecordingQCOM(queue) recording=%p\n", (void*)rec);
  if (!rec) return 1;

  cl_int eerr = fnEnd(rec);
  printf("clEndRecordingQCOM(rec) err=%d\n", eerr);
  cl_int rerr = fnRel(rec);
  printf("clReleaseRecordingQCOM(rec) err=%d\n", rerr);

  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  puts("PROBE OK");
  return 0;
}
