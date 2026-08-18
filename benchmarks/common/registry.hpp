#ifndef TDLS_BENCHMARKS_COMMON_REGISTRY_HPP
#define TDLS_BENCHMARKS_COMMON_REGISTRY_HPP



/// \file
/// \brief Variant registry of the benchmark executables.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A variant is one named point of the sweep space, bound to a runner
/// closure. Translation units (one per system dimension, generated at
/// configure time) register their variants at static-initialization
/// time, the same pattern as the test harness Registrar; the main
/// program lists, filters and runs them. External comparison solvers
/// will register into the same registry, so downstream handling has no
/// special cases.



#include <functional>
#include <string>
#include <vector>

#include "options.hpp"
#include "record.hpp"



namespace tdls_bench {



/// \brief One registered benchmark variant.
struct Variant {
    std::string tag; ///< unique variant tag, the CSV primary key
    std::function<Record(const Options&, Distribution)> run; ///< measurement closure
};

/// \return the variant registry of the current executable
inline std::vector<Variant>& registry() {
    static std::vector<Variant> variants;
    return variants;
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_REGISTRY_HPP
