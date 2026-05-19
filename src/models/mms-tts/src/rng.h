// rng.h — deterministic Gaussian RNG for on-device VITS noise generation.
//
// VITS samples two noise tensors per utterance: duration_noise and
// prior_noise. The Python reference uses torch.manual_seed(); the on-device
// equivalent is std::mt19937 + Box-Muller. Seeding with the same int yields
// bit-reproducible audio across runs, which makes regression diffs feasible.
//
// Header-only because it has zero deps and is used once in main.cpp.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

class GaussianRng {
public:
    explicit GaussianRng(uint32_t seed) : eng_(seed) {}

    // Fill `n` floats with i.i.d. standard normal samples (mean 0, var 1).
    // Uses the polar form of Box-Muller — slightly faster than std::normal_distribution
    // on small N and avoids the implementation-defined cache it carries internally.
    void fill(float* out, size_t n) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        size_t i = 0;
        while (i < n) {
            double u1 = U(eng_);
            // Guard against log(0) — extremely rare, but the assert would
            // pop in debug builds.
            if (u1 < 1e-30) u1 = 1e-30;
            const double u2 = U(eng_);
            const double r  = std::sqrt(-2.0 * std::log(u1));
            const double th = 6.283185307179586 * u2;  // 2π
            out[i++] = static_cast<float>(r * std::cos(th));
            if (i < n) out[i++] = static_cast<float>(r * std::sin(th));
        }
    }

private:
    std::mt19937 eng_;
};
