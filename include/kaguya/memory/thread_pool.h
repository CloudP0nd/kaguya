#pragma once

#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <span>

namespace kaguya {

/// Work-stealing thread pool with CPU affinity support
class ThreadPool {
public:
    /// Create thread pool with specified number of threads
    /// @param num_threads Number of worker threads (0 = hardware concurrency)
    /// @param pin_threads Whether to pin threads to specific CPUs
    explicit ThreadPool(int num_threads = 0, bool pin_threads = true);

    /// Destructor — waits for all tasks to complete
    ~ThreadPool();

    /// Submit a task for execution
    void submit(std::function<void()> task);

    /// Submit multiple tasks and wait for all to complete
    void parallel_for(int count, const std::function<void(int)>& func);

    /// Number of worker threads
    int num_threads() const { return static_cast<int>(workers_.size()); }

    /// Wait for all submitted tasks to complete
    void wait_all();

    /// Get the thread pool instance (lazy singleton)
    static ThreadPool& instance();

private:
    void worker_loop(int thread_id);

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_tasks_{0};
    bool pin_threads_;
};

} // namespace kaguya
