
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================
//
// NNOPT PATCH (depth-anything campaign, 2026-07-07): persistent on-disk program binary cache.
// Stock CLBlast lazily compiles each routine's kernels on first call, per process — ~3.2s of
// one-shot inference wall on Adreno 620. This patch adds a disk layer around BinaryCache:
// when NNOPT_CLBLAST_CACHE_DIR is set, compiled binaries (Program::GetIR()) are written to
// <dir>/<fnv1a-key>.clbin and reloaded on the next process launch via the existing
// binary-constructor path. Keyed by routine_info + device + driver version + precision, so a
// driver update changes the key; a stale/corrupt binary falls back to source compile.
// Unset env = byte-identical stock behavior (A/B kill-switch).
// CMake copies this file over _deps/clblast-src/src/routine.cpp at configure time.
//
// =================================================================================================

#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#ifndef _WIN32
#include <sys/stat.h>
#endif

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// For each kernel this map contains a list of routines it is used in
const std::vector<std::string> Routine::routines_axpy = {"AXPY", "COPY", "SCAL", "SWAP"};
const std::vector<std::string> Routine::routines_dot = {"AMAX", "ASUM", "DOT", "DOTC", "DOTU", "MAX", "MIN", "NRM2", "SUM"};
const std::vector<std::string> Routine::routines_ger = {"GER", "GERC", "GERU", "HER", "HER2", "HPR", "HPR2", "SPR", "SPR2", "SYR", "SYR2"};
const std::vector<std::string> Routine::routines_gemv = {"GBMV", "GEMV", "HBMV", "HEMV", "HPMV", "SBMV", "SPMV", "SYMV", "TMBV", "TPMV", "TRMV", "TRSV"};
const std::vector<std::string> Routine::routines_gemm = {"GEMM", "HEMM", "SYMM", "TRMM"};
const std::vector<std::string> Routine::routines_gemm_syrk = {"GEMM", "HEMM", "HER2K", "HERK", "SYMM", "SYR2K", "SYRK", "TRMM", "TRSM"};
const std::vector<std::string> Routine::routines_trsm = {"TRSM"};
const std::unordered_map<std::string, const std::vector<std::string>> Routine::routines_by_kernel = {
  {"Xaxpy", routines_axpy},
  {"Xdot", routines_dot},
  {"Xgemv", routines_gemv},
  {"XgemvFast", routines_gemv},
  {"XgemvFastRot", routines_gemv},
  {"Xtrsv", routines_gemv},
  {"Xger", routines_ger},
  {"Copy", routines_gemm_syrk},
  {"Pad", routines_gemm_syrk},
  {"Transpose", routines_gemm_syrk},
  {"Padtranspose", routines_gemm_syrk},
  {"Xgemm", routines_gemm_syrk},
  {"XgemmDirect", routines_gemm},
  {"GemmRoutine", routines_gemm},
  {"Invert", routines_trsm},
};
// =================================================================================================

// NNOPT disk-cache helpers. All no-ops unless NNOPT_CLBLAST_CACHE_DIR is set.
namespace {

uint64_t NnoptFnv1a(const std::string &s) {
  uint64_t h = 1469598103934665603ULL;
  for (const char c : s) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

std::string NnoptDriverVersion(const cl_device_id device) {
  size_t n = 0;
  if (clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &n) != CL_SUCCESS || n == 0) {
    return "unknown-driver";
  }
  std::string v(n, '\0');
  clGetDeviceInfo(device, CL_DRIVER_VERSION, n, &v[0], nullptr);
  while (!v.empty() && (v.back() == '\0' || v.back() == ' ')) { v.pop_back(); }
  return v;
}

// Returns "" when the disk cache is disabled, else the full path for this key.
std::string NnoptDiskCachePath(const std::string &routine_info, const std::string &device_name,
                               const cl_device_id device, const Precision precision) {
  const char *dir = std::getenv("NNOPT_CLBLAST_CACHE_DIR");
  if (dir == nullptr || dir[0] == '\0') { return ""; }
#ifndef _WIN32
  mkdir(dir, 0755);  // best-effort; existing dir is fine
#endif
  const auto key = routine_info + "|" + device_name + "|" + NnoptDriverVersion(device) + "|" +
                   std::to_string(static_cast<int>(precision));
  char name[32];
  std::snprintf(name, sizeof(name), "%016llx",
                static_cast<unsigned long long>(NnoptFnv1a(key)));
  return std::string(dir) + "/" + name + ".clbin";
}

std::string NnoptDiskCacheLoad(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { return ""; }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void NnoptDiskCacheStore(const std::string &path, const std::string &binary) {
  if (path.empty() || binary.empty()) { return; }
  const auto tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) { return; }
    f.write(binary.data(), static_cast<std::streamsize>(binary.size()));
    if (!f.good()) { return; }
  }
  std::rename(tmp.c_str(), path.c_str());
}

}  // namespace
// =================================================================================================

// The constructor does all heavy work, errors are returned as exceptions
Routine::Routine(Queue &queue, EventPointer event, const std::string &name,
                 const std::vector<std::string> &kernel_names, const Precision precision,
                 const std::vector<database::DatabaseEntry> &userDatabase,
                 std::initializer_list<const char *> source):
    precision_(precision),
    routine_name_(name),
    kernel_names_(kernel_names),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    db_(kernel_names) {

  InitDatabase(device_, kernel_names, precision, userDatabase, db_);
  InitProgram(source);
}

void Routine::InitProgram(std::initializer_list<const char *> source) {

  // Determines the identifier for this particular routine call
  auto routine_info = routine_name_;
  for (const auto &kernel_name : kernel_names_) {
    routine_info += "_" + kernel_name + db_(kernel_name).GetValuesString();
  }
  log_debug(routine_info);

  // Queries the cache to see whether or not the program (context-specific) is already there
  bool has_program;
  program_ = ProgramCache::Instance().Get(ProgramKeyRef{ context_(), device_(), precision_, routine_info },
                                          &has_program);
  if (has_program) { return; }

  // Sets the build options from an environmental variable (if set)
  auto options = std::vector<std::string>();
  const auto environment_variable = std::getenv("CLBLAST_BUILD_OPTIONS");
  if (environment_variable != nullptr) {
    options.push_back(std::string(environment_variable));
  }

  // Queries the cache to see whether or not the binary (device-specific) is already there. If it
  // is, a program is created and stored in the cache
  const auto device_name = GetDeviceName(device_);
  const auto platform_id = device_.PlatformID();
  bool has_binary;
  auto binary = BinaryCache::Instance().Get(BinaryKeyRef{platform_id,  precision_, routine_info, device_name },
                                            &has_binary);

  // NNOPT: on in-memory miss, try the on-disk cache from a previous process launch.
  const auto disk_path = NnoptDiskCachePath(routine_info, device_name, device_(), precision_);
  auto from_disk = false;
  if (!has_binary && !disk_path.empty()) {
    binary = NnoptDiskCacheLoad(disk_path);
    if (!binary.empty()) { has_binary = true; from_disk = true; }
  }

  if (has_binary) {
    auto binary_ok = true;
    try {
      program_ = std::make_shared<Program>(device_, context_, binary);
      SetOpenCLKernelStandard(device_, options);
      program_->Build(device_, options);
    } catch (...) {
      // NNOPT: a stale disk binary (driver/device change, truncated file) must fall back to
      // source compile, not kill the process. In-memory cache hits keep stock throw behavior.
      if (!from_disk) { throw; }
      binary_ok = false;
      std::remove(disk_path.c_str());
    }
    if (binary_ok) {
      if (from_disk) {
        BinaryCache::Instance().Store(BinaryKey{platform_id, precision_, routine_info, device_name},
                                      std::string{binary});
      }
      ProgramCache::Instance().Store(ProgramKey{ context_(), device_(), precision_, routine_info },
                                      std::shared_ptr<Program>{program_});
      return;
    }
  }

  // Otherwise, the kernel will be compiled and program will be built. Both the binary and the
  // program will be added to the cache.

  // Inspects whether or not FP64 is supported in case of double precision
  if ((precision_ == Precision::kDouble && !PrecisionSupported<double>(device_)) ||
      (precision_ == Precision::kComplexDouble && !PrecisionSupported<double2>(device_))) {
    throw RuntimeErrorCode(StatusCode::kNoDoublePrecision);
  }

  // As above, but for FP16 (half precision)
  if (precision_ == Precision::kHalf && !PrecisionSupported<half>(device_)) {
    throw RuntimeErrorCode(StatusCode::kNoHalfPrecision);
  }

  // Collects the parameters for this device in the form of defines
  auto source_string = std::string{""};
  for (const auto &kernel_name : kernel_names_) {
    source_string += db_(kernel_name).GetDefines();
  }

  // Adds routine-specific code to the constructed source string
  for (const char *s: source) {
    source_string += s;
  }

  // Completes the source and compiles the kernel
  program_ = CompileFromSource(source_string, precision_, routine_name_,
                               device_, context_, options, 0);

  // Store the compiled binary and program in the cache
  BinaryCache::Instance().Store(BinaryKey{platform_id, precision_, routine_info, device_name},
                                program_->GetIR());

  // NNOPT: persist for the next process launch.
  NnoptDiskCacheStore(disk_path, program_->GetIR());

  ProgramCache::Instance().Store(ProgramKey{context_(), device_(), precision_, routine_info},
                                 std::shared_ptr<Program>{program_});
}

// =================================================================================================
} // namespace clblast
