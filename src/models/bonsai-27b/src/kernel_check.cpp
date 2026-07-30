// Standalone kernel compiler check — builds every .cl in a directory against
// the REAL device compiler and prints the build log. No weights, no .nnb.
//
// Why this exists: the Adreno compiler is the one with the quirks (register
// cliffs, emulated short-vector ops, local-mem limits), and waiting for a
// 3.8 GB model to convert before finding a syntax error in a new kernel is a
// terrible loop. This catches build errors in seconds.
//
//   ./scripts/check_kernels.sh
#include <dirent.h>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "opencl_context.h"

static std::string read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

int main(int argc, char** argv) {
    const std::string kdir = argc > 1 ? argv[1] : "kernels";
    const char* opts = getenv("BONSAI_KERNEL_OPTS");

    OpenCLContext ocl;
    if (!ocl.initialize()) { fprintf(stderr, "FATAL: no OpenCL device\n"); return 1; }
    fprintf(stderr, "kernel_check: building every .cl in %s\n", kdir.c_str());

    std::vector<std::string> names;
    if (DIR* d = opendir(kdir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 3 && n.compare(n.size() - 3, 3, ".cl") == 0)
                names.push_back(n);
        }
        closedir(d);
    }
    if (names.empty()) { fprintf(stderr, "FATAL: no .cl files in %s\n", kdir.c_str()); return 1; }
    std::sort(names.begin(), names.end());

    int failed = 0;
    for (const auto& n : names) {
        const std::string src = read_file(kdir + "/" + n);
        if (src.empty()) { fprintf(stderr, "  %-24s SKIP (empty)\n", n.c_str()); continue; }
        const char* p = src.c_str();
        size_t len = src.size();
        cl_int err = CL_SUCCESS;
        cl_program prog = clCreateProgramWithSource(ocl.context(), 1, &p, &len, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "  %-24s FAIL (create %d)\n", n.c_str(), err);
            failed++;
            continue;
        }
        cl_device_id dev = ocl.device();
        err = clBuildProgram(prog, 1, &dev, opts ? opts : "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t ls = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
            std::vector<char> log(ls + 1, 0);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, log.data(), nullptr);
            fprintf(stderr, "  %-24s FAIL (build %d)\n%s\n", n.c_str(), err, log.data());
            failed++;
        } else {
            // list the kernels it exposes — a cheap check that names match what
            // model.cpp asks clCreateKernel for
            size_t ns = 0;
            clGetProgramInfo(prog, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &ns);
            std::vector<char> kn(ns + 1, 0);
            clGetProgramInfo(prog, CL_PROGRAM_KERNEL_NAMES, ns, kn.data(), nullptr);
            fprintf(stderr, "  %-24s OK   [%s]\n", n.c_str(), kn.data());
        }
        clReleaseProgram(prog);
    }
    fprintf(stderr, failed ? "\n%d KERNEL(S) FAILED TO BUILD\n" : "\nALL KERNELS BUILD\n",
            failed);
    return failed ? 1 : 0;
}
