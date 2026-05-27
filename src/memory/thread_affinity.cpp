#include "kaguya/thread_affinity.h"
#include "kaguya/cpu_features.h"

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace kaguya {

bool ThreadAffinity::pin_to_cpu(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
#else
    (void)cpu_id;
    return false;
#endif
}

bool ThreadAffinity::pin_to_cpus(const std::vector<int>& cpu_ids) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int id : cpu_ids) {
        CPU_SET(id, &cpuset);
    }
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
#else
    (void)cpu_ids;
    return false;
#endif
}

std::vector<int> ThreadAffinity::get_affinity() {
    std::vector<int> result;
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                result.push_back(i);
            }
        }
    }
#endif
    return result;
}

int ThreadAffinity::num_cpus() {
#ifdef __linux__
    return sysconf(_SC_NPROCESSORS_ONLN);
#else
    return std::thread::hardware_concurrency();
#endif
}

std::vector<int> ThreadAffinity::distribute_threads(int num_threads, int numa_node) {
    const auto& info = CpuFeatureDetector::get();

    std::vector<int> available_cpus;
    if (numa_node >= 0 && numa_node < static_cast<int>(info.numa_nodes.size())) {
        available_cpus = info.numa_nodes[numa_node].cpu_list;
    } else {
        // Use all CPUs
        for (const auto& node : info.numa_nodes) {
            for (int cpu : node.cpu_list) {
                available_cpus.push_back(cpu);
            }
        }
    }

    if (available_cpus.empty()) {
        // Fallback: sequential CPU IDs
        for (int i = 0; i < num_threads; ++i) {
            available_cpus.push_back(i % num_cpus());
        }
        return available_cpus;
    }

    // Distribute threads across available CPUs, round-robin
    std::vector<int> result;
    for (int i = 0; i < num_threads; ++i) {
        result.push_back(available_cpus[i % available_cpus.size()]);
    }
    return result;
}

bool ThreadAffinity::reset_affinity() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int n = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < n; ++i) {
        CPU_SET(i, &cpuset);
    }
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
#else
    return false;
#endif
}

} // namespace kaguya
