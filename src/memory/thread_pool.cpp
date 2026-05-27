#include "kaguya/thread_pool.h"
#include "kaguya/thread_affinity.h"
#include "kaguya/cpu_features.h"

#include <algorithm>
#include <cassert>

namespace kaguya {

ThreadPool::ThreadPool(int num_threads, bool pin_threads)
    : pin_threads_(pin_threads) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    workers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        if (pin_threads_) {
            // Pin thread after creation — must be done from within the thread
            // We'll set affinity in worker_loop
        }
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::worker_loop(int thread_id) {
    // Pin this thread to a specific CPU
    if (pin_threads_) {
        auto cpus = ThreadAffinity::distribute_threads(
            static_cast<int>(workers_.size()));
        if (thread_id < static_cast<int>(cpus.size())) {
            ThreadAffinity::pin_to_cpu(cpus[thread_id]);
        }
    }

    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        active_tasks_++;
        task();
        active_tasks_--;

        // Check if all tasks are done
        if (active_tasks_ == 0 && tasks_.empty()) {
            cv_done_.notify_all();
        }
    }
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::parallel_for(int count, const std::function<void(int)>& func) {
    if (count <= 0) return;

    if (count == 1 || num_threads() <= 1) {
        for (int i = 0; i < count; ++i) func(i);
        return;
    }

    std::atomic<int> next_task{0};
    std::atomic<int> completed{0};

    int n_workers = std::min(num_threads(), count);

    for (int t = 0; t < n_workers; ++t) {
        submit([&next_task, &completed, count, &func, n_workers]() {
            while (true) {
                int i = next_task.fetch_add(1);
                if (i >= count) break;
                func(i);
                completed.fetch_add(1);
            }
        });
    }

    // Wait for completion
    wait_all();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool(std::thread::hardware_concurrency(), true);
    return pool;
}

} // namespace kaguya
