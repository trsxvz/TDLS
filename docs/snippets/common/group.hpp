#ifndef TDLS_SNIPPETS_GROUP_HPP
#define TDLS_SNIPPETS_GROUP_HPP



/// \file
/// \brief Groups of CPU threads of the CooperativeLUpp snippets.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// run_group runs its function on every thread of a group, passing the
/// rank of the thread, as a GPU kernel runs on the lanes of a warp. The
/// threads are std::thread instances; a group of one thread runs on the
/// calling thread. The solver deduces the barrier of the group, so the
/// function passes none. Barrier is the explicit barrier of the snippet
/// that passes one: a condition-variable barrier, which the standard
/// library of every supported compiler provides.



#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>



namespace snippets {



/// \brief Reusable barrier of a fixed number of threads.
class Barrier {
  public:
    /// \param[in] count number of threads taking part in each barrier
    explicit Barrier(const int count) : count(count) {
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

/// \brief Runs fn(tx) on every thread of a group whose size is a
/// runtime value, as the number of threads of a runtime solver: new
/// threads, or the calling thread for a group of one.
/// \tparam Fn callable type, invoked as fn(int tx)
/// \param[in] threads number of threads of the group
/// \param[in] fn      the work of one thread
template<typename Fn>
void run_group(const int threads, Fn&& fn) {
    if (threads == 1) {
        fn(0);
        return;
    }
    std::vector<std::thread> group;
    for (int tx = 0; tx < threads; ++tx)
        group.emplace_back([&fn, tx] { fn(tx); });
    for (auto& thread : group)
        thread.join();
}

/// \brief Runs fn(tx) on every thread of a group whose size is known at
/// compile time, as the number of threads of a compile-time solver.
/// \tparam threads number of threads of the group
/// \tparam Fn      callable type, invoked as fn(int tx)
/// \param[in] fn the work of one thread
template<int threads, typename Fn>
void run_group(Fn&& fn) {
    run_group(threads, fn);
}



} // namespace snippets



#endif // TDLS_SNIPPETS_GROUP_HPP
