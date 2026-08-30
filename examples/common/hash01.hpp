/// \file
/// \brief Deterministic per-index input data of the batch examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#ifndef TDLS_EXAMPLES_HASH01_HPP
#define TDLS_EXAMPLES_HASH01_HPP

#include <tdls/core/macros.hpp>

/// \brief Deterministic map from an index to a value in [0, 1),
/// standing in for per-cell or per-point input data.
/// \param[in] i index
/// \return value in [0, 1)
TDLS_HOST_DEVICE inline double hash01(const unsigned long long i) {
    unsigned long long z = i + 0x9e3779b97f4a7c15ull;
    z                    = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z                    = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z                    = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

#endif // TDLS_EXAMPLES_HASH01_HPP
