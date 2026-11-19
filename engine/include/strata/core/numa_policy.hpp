#pragma once

#include <cerrno>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core::detail {

// Allocation policy belongs to the calling thread and is inherited by children;
// it changes neither CPU affinity nor the placement of pages already allocated.
inline bool prefer_numa_node(int node, std::string& error) {
#if defined(__linux__)
    if (node < 0 || node >= 64 || sizeof(unsigned long) < 8) {
        error = "node must be in 0..63 on a 64-bit Linux host";
        return false;
    }
    const unsigned long mask = 1ul << node;
    if (syscall(SYS_set_mempolicy, MPOL_PREFERRED, &mask, 64ul) == 0) return true;
    error = std::strerror(errno);
#else
    (void) node;
    error = "NUMA allocation policy requires Linux";
#endif
    return false;
}

inline bool use_local_numa_policy(std::string& error) {
#if defined(__linux__)
    if (syscall(SYS_set_mempolicy, MPOL_LOCAL, nullptr, 0ul) == 0) return true;
    error = std::strerror(errno);
#else
    error = "NUMA allocation policy requires Linux";
#endif
    return false;
}

} // namespace strata::core::detail
