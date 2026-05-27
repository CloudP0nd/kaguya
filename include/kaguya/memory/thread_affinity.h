#pragma once

#include <vector>
#include <cstdint>

namespace kaguya {

/// CPU affinity manager — pin threads to specific CPU cores
class ThreadAffinity {
public:
    /// Pin the current thread to the specified CPU
    static bool pin_to_cpu(int cpu_id);

    /// Pin the current thread to a set of CPUs
    static bool pin_to_cpus(const std::vector<int>& cpu_ids);

    /// Get the current thread's CPU affinity
    static std::vector<int> get_affinity();

    /// Get the number of available CPUs
    static int num_cpus();

    /// Distribute n threads across available CPUs (respects NUMA if possible)
    static std::vector<int> distribute_threads(int num_threads, int numa_node = -1);

    /// Reset affinity (allow any CPU)
    static bool reset_affinity();
};

} // namespace kaguya
