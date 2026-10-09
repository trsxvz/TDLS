#ifndef TDLS_TESTS_SOLVERS_COOPERATIVE_LUPP_GROUP_RUNNER_HPP
#define TDLS_TESTS_SOLVERS_COOPERATIVE_LUPP_GROUP_RUNNER_HPP



/// \file
/// \brief Shared plumbing of the CooperativeLUpp suites: groups of CPU
/// threads and operand storage per residency.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The group of threads that solves one system is a set of std::thread
/// instances synchronized by GroupBarrier, a condition-variable barrier
/// (std::barrier is avoided on purpose: the standard library of the
/// GCC 10 floor lacks it). A group of one thread runs on the calling
/// thread with the default tdls::NoSync barrier, the sequential path.
/// The threads of a group do not run in lockstep, so a data race in the
/// solver turns into wrong values, which the bitwise bridges catch.
///
/// GroupRunner materializes one system for the compile-time solver, in
/// the storage dictated by the residency template booleans: per-thread slices in local arrays of each
/// thread for internal operands, slices of a small strided arena shared
/// by the group for external ones. It runs one entry path and gathers
/// every output back to canonical storage (factored rows row-major in
/// physical order, row positions indexed by physical row, solution), so
/// that different combinations can be compared bitwise.
///
/// DynamicGroupRunner does the same for the runtime solver, whose operands
/// are all external. The external arena holds three system slots and the
/// system under test lives in the middle one, so external accesses
/// exercise a stride greater than one.



#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include <tdls/tdls.hpp>



namespace tdls_tests {



/// \brief Reusable barrier of a fixed number of threads, built on a mutex
/// and a condition variable.
class GroupBarrier {
  public:
    /// \param[in] count number of threads taking part in each barrier
    explicit GroupBarrier(const int count) : count(count) {
    }

    /// \brief Blocks until every thread of the group has arrived; the
    /// mutex makes the memory writes of each thread visible to all.
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const unsigned long long arrival_generation = generation;
        if (++waiting == count) {
            waiting = 0;
            ++generation;
            condition.notify_all();
        } else {
            condition.wait(lock, [&] { return generation != arrival_generation; });
        }
    }

  private:
    std::mutex mutex;
    std::condition_variable condition;
    int count;
    int waiting                   = 0;
    unsigned long long generation = 0;
};

/// \brief Runs fn(tx, sync) on `threads` new threads sharing a
/// GroupBarrier.
/// \tparam Fn callable type, invoked as fn(int tx, auto&& sync)
/// \param[in] threads number of threads of the group
/// \param[in] fn      the work of one thread
template<typename Fn>
void spawn_group(const int threads, Fn& fn) {
    GroupBarrier barrier(threads);
    std::vector<std::thread> group;
    group.reserve(threads);
    for (int tx = 0; tx < threads; ++tx)
        group.emplace_back([&fn, &barrier, tx] {
            auto sync = [&barrier] { barrier.arrive_and_wait(); };
            fn(tx, sync);
        });
    for (auto& thread : group)
        thread.join();
}

/// \brief Runs fn(tx, sync) on every thread of a group of compile-time
/// size: on the calling thread with tdls::NoSync for a group of one, on
/// `threads` new threads sharing a GroupBarrier otherwise. The dispatch is
/// compile-time, as the barrier contract of the compile-time solver.
/// \tparam threads number of threads of the group
/// \tparam Fn      callable type, invoked as fn(int tx, auto&& sync)
/// \param[in] fn the work of one thread
template<int threads, typename Fn>
void run_group(Fn&& fn) {
    if constexpr (threads == 1)
        fn(0, tdls::NoSync{});
    else
        spawn_group(threads, fn);
}

/// \brief Runs fn(tx, sync) on every thread of a group of runtime size,
/// for the runtime solver, whose barrier contract is a runtime
/// precondition: tdls::NoSync on the calling thread for a group of one, a
/// GroupBarrier otherwise.
/// \tparam Fn callable type, invoked as fn(int tx, auto&& sync)
/// \param[in] threads number of threads of the group
/// \param[in] fn      the work of one thread
template<typename Fn>
void run_group(const int threads, Fn&& fn) {
    if (threads == 1)
        fn(0, tdls::NoSync{});
    else
        spawn_group(threads, fn);
}

/// \brief Unroll policy of the test configurations: forced for groups
/// of several threads on small systems (N <= 8), the GPU configuration,
/// and no pragma otherwise. The unrolled branch is the same code at every
/// dimension, while its instantiated size grows as rows_per_thread x N^2
/// (N^3 for one thread per system): under the sanitizers at -O2, the
/// large instantiations dominated the build time of the suites, for no
/// extra coverage. Every suite keeps at least one unrolled group of a
/// small system, with full and mixed slots; the config_knobs suite ties
/// both branches bitwise, and the constexpr suite certifies the unrolled
/// branch on one thread per system.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
template<int N, int rows_per_thread>
constexpr bool test_unroll = rows_per_thread < N && N <= 8;

/// \brief Entry paths exercised through the runner.
enum class Path {
    combined,      ///< solve()
    split,         ///< factorize() then substitute()
    fused,         ///< solve_inplace()
    split_inplace, ///< factorize() then substitute_inplace()
    canonical      ///< factorize() then substitute_canonical()
};

/// \brief Calls fn once per path, in order, with the path as a
/// std::integral_constant: each runner instantiation then compiles only
/// the entry points of its path.
/// \tparam paths entry paths to exercise
/// \tparam Fn    callable type, invoked as fn(std::integral_constant<Path, path>{})
/// \param[in] fn the work of one path
template<Path... paths, typename Fn>
void for_paths(Fn&& fn) {
    (fn(std::integral_constant<Path, paths>{}), ...);
}

/// \brief Runs one entry path of the CooperativeLUpp solver under one
/// residency combination, on a group of CPU threads.
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam Config          solver configuration
/// \tparam internal_rhs    residency of the right-hand side and solution
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of the matrix
template<typename T, int N, tdls::CooperativeLUppConfig<T> Config, bool internal_rhs,
         bool internal_piv, bool internal_matrix>
struct GroupRunner {
    /// \brief Solver under test.
    using Solver = tdls::CooperativeLUppSolverStatic<T, N, Config>;
    /// \brief Rows held by each thread.
    static constexpr int rows = Solver::rows_per_thread;
    /// \brief Threads of the group.
    static constexpr int threads = Solver::threads_per_system;
    /// \brief Number of system slots of the external arena.
    static constexpr int arena_count = 3;
    /// \brief Arena slot holding the system under test.
    static constexpr int slot = 1;

    /// \brief Flat index of matrix element (r, c) under the configured layout.
    /// \param[in] r row
    /// \param[in] c column
    /// \return the flat index
    static constexpr std::size_t matrix_index(const int r, const int c) {
        return Config.layout == tdls::MatrixLayout::RowMajor ? std::size_t(r) * N + c
                                                             : std::size_t(c) * N + r;
    }

    /// \brief Index of element (K, c) in the slice of a thread under the
    /// configured layout.
    /// \param[in] K row slot
    /// \param[in] c column
    /// \return the slice index
    static constexpr std::size_t slice_index(const int K, const int c) {
        return Config.layout == tdls::MatrixLayout::RowMajor ? std::size_t(K) * N + c
                                                             : std::size_t(c) * rows + K;
    }

    /// \brief Solves one system and returns every output in canonical
    /// storage.
    /// \tparam path entry path to exercise
    /// \param[in]  A0      original matrix, contiguous row-major
    /// \param[in]  b0      right-hand side, contiguous (ignored by the
    ///             canonical path)
    /// \param[out] A_out   factored rows, contiguous row-major, physical
    ///             order
    /// \param[out] piv_out position of each physical row in the pivoted
    ///             order
    /// \param[out] x_out   solution
    /// \param[out] uniform whether every thread returned the same verdict
    /// \param[in]  col     canonical column of the canonical path
    /// \return the verdict of thread 0 (true for the substitution-only
    ///         paths when the factorization succeeded).
    template<Path path>
    [[nodiscard]] static bool run(const T* A0, const T* b0, T* A_out, int* piv_out, T* x_out,
                                  bool& uniform, [[maybe_unused]] const int col = 0) {
        std::vector<T> A_arena(static_cast<std::size_t>(N) * N * arena_count, T(0));
        std::vector<int> piv_arena(static_cast<std::size_t>(N) * arena_count, 0);
        std::vector<T> b_arena(static_cast<std::size_t>(N) * arena_count, T(0));
        std::vector<T> x_arena(static_cast<std::size_t>(N) * arena_count, T(0));
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                A_arena[matrix_index(r, c) * arena_count + slot] = A0[r * N + c];
        for (int r = 0; r < N; ++r) {
            b_arena[static_cast<std::size_t>(r) * arena_count + slot] = b0[r];
            x_arena[static_cast<std::size_t>(r) * arena_count + slot] = b0[r];
        }
        std::vector<T> work(Solver::workspace_size, T(0));
        std::vector<int> verdicts(threads, 0);

        run_group<threads>([&](const int tx, auto&& sync) {
            // Per-thread slices: the rows tx + K * threads of the system.
            T A_slice[rows * N];
            int piv_slice[rows];
            T b_slice[rows], x_slice[rows];
            for (int K = 0; K < rows; ++K) {
                const int r = tx + K * threads;
                for (int c = 0; c < N; ++c)
                    A_slice[slice_index(K, c)] = r < N ? A0[r * N + c] : T(0);
                piv_slice[K] = 0;
                b_slice[K]   = r < N ? b0[r] : T(0);
                x_slice[K]   = b_slice[K];
            }

            T* A                        = internal_matrix ? A_slice : A_arena.data() + slot;
            const int A_str             = internal_matrix ? 1 : arena_count;
            int* piv                    = internal_piv ? piv_slice : piv_arena.data() + slot;
            const int p_str             = internal_piv ? 1 : arena_count;
            [[maybe_unused]] const T* b = internal_rhs ? b_slice : b_arena.data() + slot;
            T* x                        = internal_rhs ? x_slice : x_arena.data() + slot;
            const int r_str             = internal_rhs ? 1 : arena_count;

            bool ok = true;
            if constexpr (path == Path::combined) {
                ok = Solver::template solve<internal_rhs, internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, b, x, r_str, work.data(), sync);
            }
            if constexpr (path == Path::split) {
                ok = Solver::template factorize<internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, work.data(), sync);
                Solver::template substitute<internal_rhs, internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, b, x, r_str, work.data(), sync);
            }
            if constexpr (path == Path::fused) {
                ok = Solver::template solve_inplace<internal_rhs, internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, x, r_str, work.data(), sync);
            }
            if constexpr (path == Path::split_inplace) {
                ok = Solver::template factorize<internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, work.data(), sync);
                Solver::template substitute_inplace<internal_rhs, internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, x, r_str, work.data(), sync);
            }
            if constexpr (path == Path::canonical) {
                ok = Solver::template factorize<internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, work.data(), sync);
                Solver::template substitute_canonical<internal_rhs, internal_piv, internal_matrix>(
                    tx, A, A_str, piv, p_str, col, x, r_str, work.data(), sync);
            }
            verdicts[tx] = ok ? 1 : 0;

            // Each thread gathers the outputs of its own rows: distinct
            // elements of the canonical arrays.
            for (int K = 0; K < rows; ++K) {
                const int r = tx + K * threads;
                if (r >= N) continue;
                for (int c = 0; c < N; ++c)
                    A_out[r * N + c] = internal_matrix
                                           ? A_slice[slice_index(K, c)]
                                           : A_arena[matrix_index(r, c) * arena_count + slot];
                piv_out[r] = internal_piv
                                 ? piv_slice[K]
                                 : piv_arena[static_cast<std::size_t>(r) * arena_count + slot];
                x_out[r] = internal_rhs ? x_slice[K]
                                        : x_arena[static_cast<std::size_t>(r) * arena_count + slot];
            }
        });

        uniform = true;
        for (const int v : verdicts)
            if (v != verdicts[0]) uniform = false;
        return verdicts[0] == 1;
    }
};


/// \brief Runs one entry path of the runtime CooperativeLUpp solver on a
/// group of CPU threads. Every operand is external, in the middle slot of
/// a three-slot arena, as in the external mode of GroupRunner.
/// \tparam T      scalar type
/// \tparam Config solver configuration
template<typename T, tdls::CooperativeLUppConfig<T> Config>
struct DynamicGroupRunner {
    /// \brief Solver under test.
    using Solver = tdls::CooperativeLUppSolverDynamic<T, Config>;
    /// \brief Number of system slots of the external arena.
    static constexpr int arena_count = 3;
    /// \brief Arena slot holding the system under test.
    static constexpr int slot = 1;

    /// \brief Solves one system and returns every output in canonical
    /// storage, as GroupRunner::run.
    /// \tparam path entry path to exercise
    /// \param[in]  n       system dimension
    /// \param[in]  A0      original matrix, contiguous row-major
    /// \param[in]  b0      right-hand side, contiguous (ignored by the
    ///             canonical path)
    /// \param[out] A_out   factored rows, contiguous row-major, physical
    ///             order
    /// \param[out] piv_out position of each physical row in the pivoted
    ///             order
    /// \param[out] x_out   solution
    /// \param[out] uniform whether every thread returned the same verdict
    /// \param[in]  col     canonical column of the canonical path
    /// \return the verdict of thread 0 (true for the substitution-only
    ///         paths when the factorization succeeded).
    template<Path path>
    [[nodiscard]] static bool run(const int n, const T* A0, const T* b0, T* A_out, int* piv_out,
                                  T* x_out, bool& uniform, [[maybe_unused]] const int col = 0) {
        const auto matrix_index = [n](const int r, const int c) {
            return Config.layout == tdls::MatrixLayout::RowMajor ? std::size_t(r) * n + c
                                                                 : std::size_t(c) * n + r;
        };
        std::vector<T> A(static_cast<std::size_t>(n) * n * arena_count, T(0));
        std::vector<int> piv(static_cast<std::size_t>(n) * arena_count, 0);
        std::vector<T> b(static_cast<std::size_t>(n) * arena_count, T(0));
        std::vector<T> x(static_cast<std::size_t>(n) * arena_count, T(0));
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c)
                A[matrix_index(r, c) * arena_count + slot] = A0[r * n + c];
            b[static_cast<std::size_t>(r) * arena_count + slot] = b0[r];
            x[static_cast<std::size_t>(r) * arena_count + slot] = b0[r];
        }
        std::vector<T> work(Solver::workspace_size(n), T(0));
        const int threads = Solver::threads_per_system(n);
        std::vector<int> verdicts(threads, 0);
        T* As                  = A.data() + slot;
        int* ps                = piv.data() + slot;
        [[maybe_unused]] T* bs = b.data() + slot;
        T* xs                  = x.data() + slot;
        const int stride       = arena_count;

        run_group(threads, [&](const int tx, auto&& sync) {
            bool ok = true;
            if constexpr (path == Path::combined) {
                ok =
                    Solver::solve(n, tx, As, stride, ps, stride, bs, xs, stride, work.data(), sync);
            }
            if constexpr (path == Path::split) {
                ok = Solver::factorize(n, tx, As, stride, ps, stride, work.data(), sync);
                Solver::substitute(n, tx, As, stride, ps, stride, bs, xs, stride, work.data(),
                                   sync);
            }
            if constexpr (path == Path::fused) {
                ok = Solver::solve_inplace(n, tx, As, stride, ps, stride, xs, stride, work.data(),
                                           sync);
            }
            if constexpr (path == Path::split_inplace) {
                ok = Solver::factorize(n, tx, As, stride, ps, stride, work.data(), sync);
                Solver::substitute_inplace(n, tx, As, stride, ps, stride, xs, stride, work.data(),
                                           sync);
            }
            if constexpr (path == Path::canonical) {
                ok = Solver::factorize(n, tx, As, stride, ps, stride, work.data(), sync);
                Solver::substitute_canonical(n, tx, As, stride, ps, stride, col, xs, stride,
                                             work.data(), sync);
            }
            verdicts[tx] = ok ? 1 : 0;
        });

        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c)
                A_out[r * n + c] = A[matrix_index(r, c) * arena_count + slot];
            piv_out[r] = piv[static_cast<std::size_t>(r) * arena_count + slot];
            x_out[r]   = x[static_cast<std::size_t>(r) * arena_count + slot];
        }
        uniform = true;
        for (const int v : verdicts)
            if (v != verdicts[0]) uniform = false;
        return verdicts[0] == 1;
    }
};


} // namespace tdls_tests



#endif // TDLS_TESTS_SOLVERS_COOPERATIVE_LUPP_GROUP_RUNNER_HPP
