//===---------- curand_shim.h - Host stand-in for cuRAND device API ------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Minimal host implementation of the few cuRAND device-API calls used by
/// SuperConga, for the CPU backend. Each kernel "thread" owns its own state,
/// seeded from (seed, subsequence), so results are reproducible for a given
/// seed regardless of thread count.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_CPU_CURAND_SHIM_H_
#define CONGA_CPU_CURAND_SHIM_H_

#include <cstdint>
#include <random>

struct curandState {
  std::mt19937_64 engine;
  std::normal_distribution<double> normal{0.0, 1.0};
};

inline void curand_init(const unsigned long long seed,
                        const unsigned long long subsequence,
                        const unsigned long long offset, curandState *state) {
  // Mix seed and subsequence (splitmix64 step) so neighbouring subsequences
  // give uncorrelated streams.
  std::uint64_t z = seed ^ (subsequence * 0x9E3779B97F4A7C15ULL) ^ offset;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  state->engine.seed(z);
  state->normal.reset();
}

inline double curand_normal_double(curandState *state) {
  return state->normal(state->engine);
}

inline float curand_normal(curandState *state) {
  return static_cast<float>(state->normal(state->engine));
}

#endif // CONGA_CPU_CURAND_SHIM_H_
