#include "strata/core/numa_policy.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sched.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace {
constexpr size_t bytes = 8u << 20;
bool placed_on(void* memory, int node) {
    std::array<void*, 8> pages;
    std::array<int, 8> status;
    for (size_t i = 0; i < pages.size(); ++i) pages[i] = (char*) memory + (i << 20);
    if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) != 0) return false;
    for (int actual : status) if (actual != node) return false;
    return true;
}
}

int main() {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return 77;
    std::vector<std::pair<int, int>> nodes;
    for (int n = 0; n < 64 && nodes.size() < 2; ++n)
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &allowed) && std::filesystem::exists(
                    "/sys/devices/system/node/node" + std::to_string(n) + "/cpu" + std::to_string(c))) {
                nodes.emplace_back(n, c);
                break;
            }
    if (nodes.size() < 2) { std::puts("numa_policy_test: needs two CPU NUMA nodes (skipped)"); return 77; }
    // MPOL_LOCAL may legitimately spill under pressure. Require free pages above
    // the node's watermarks, rather than testing locality while a model fills RAM.
    for (const auto& [node, cpu] : nodes) {
        (void) cpu;
        std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
        std::string line;
        uint64_t free_kib = 0;
        while (std::getline(f, line)) {
            std::istringstream fields(line);
            std::string a, id, key;
            if (fields >> a >> id >> key >> free_kib && key == "MemFree:") break;
            free_kib = 0;
        }
        if (free_kib < 512u * 1024u) {
            std::printf("numa_policy_test: node %d has only %llu KiB free (skipped under memory pressure)\n",
                        node, (unsigned long long) free_kib);
            return 77;
        }
    }
    std::string error;
    if (!strata::core::detail::prefer_numa_node(nodes[0].first, error)) {
        std::printf("numa_policy_test: allocation policy unavailable: %s (skipped)\n", error.c_str());
        return 77;
    }
    void* shared = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) return 1;
    std::memset(shared, 1, bytes);
    const bool preferred = placed_on(shared, nodes[0].first);
    bool local = false, policy = false, affinity = false;
    std::thread toucher([&] {
        cpu_set_t on;
        CPU_ZERO(&on);
        CPU_SET(nodes[1].second, &on);
        affinity = sched_setaffinity(0, sizeof on, &on) == 0;
        policy = strata::core::detail::use_local_numa_policy(error);
        void* expert = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (expert == MAP_FAILED) return;
        std::memset(expert, 2, bytes);
        local = placed_on(expert, nodes[1].first);
        munmap(expert, bytes);
    });
    toucher.join();
    munmap(shared, bytes);
    const bool ok = preferred && affinity && policy && local;
    std::printf("numa_policy_test: preferred node %d %s, helper override node %d %s; %s\n",
                nodes[0].first, preferred ? "ok" : "FAILED", nodes[1].first, local ? "ok" : "FAILED",
                ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
