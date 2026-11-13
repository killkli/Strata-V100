#include "strata/core/numa_stage.hpp"

#include <cstdio>
#include <cstdlib>

using strata::core::detail::make_stage_numa_plan;
using strata::core::detail::NumaLayerLocation;

namespace {
int checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
}

int main() {
    const std::vector<NumaLayerLocation> layers{{0, 0}, {1, 14}};
    const uint64_t absent = UINT64_MAX;
    const auto aligned = make_stage_numa_plan({0, 8192, 16384, 24576}, 2, {8192, 8192}, layers);
    check(aligned.ready && aligned.nodes[0] == 0 && aligned.nodes[1] == 1, "two GPU nodes selected");
    check(aligned.ranges[0].size() == 1 && aligned.ranges[0][0] == std::pair<uint64_t, uint64_t>{0, 16384},
          "adjacent experts on one node coalesce");
    check(aligned.bytes[0] == 16384 && aligned.bytes[1] == 16384, "physical page bytes accounted");
    const auto edges = make_stage_numa_plan({1, 6001}, 1, {6000, 6000}, layers);
    check(edges.ready && edges.bytes[0] == 0 && edges.bytes[1] == 0, "shared boundary pages stay with the default toucher");
    const auto holes = make_stage_numa_plan({absent, absent, 0, 8192}, 2, {8192, 8192}, layers);
    check(holes.ready && holes.bytes[0] == 0 && holes.bytes[1] == 16384 && holes.cores[0] == 0,
          "fully resident stages need no complement pages");
    const auto overlap = make_stage_numa_plan({0, 4096}, 1, {8192, 8192}, layers);
    check(!overlap.ready, "overlapping cross-node ranges are refused before first touch");
    check(!make_stage_numa_plan({absent - 5, 0}, 1, {8192, 8192}, layers).ready, "overflow is refused");
    check(!make_stage_numa_plan({0}, 1, {8192, 8192}, layers).ready, "truncated layer directory refused");
    check(!make_stage_numa_plan({0, 8192}, 0, {8192, 8192}, layers).ready, "zero experts refused");
    check(!make_stage_numa_plan({0, 8192}, 1, {8192, 8192}, {{0, 0}, {-1, 14}}).ready, "unknown node refused");
    check(!make_stage_numa_plan({0, 8192}, 1, {8192, 8192}, {{0, 0}, {0, 1}}).ready, "one socket preserves the existing path");
    check(!make_stage_numa_plan({0, 8192}, 1, {8192, 8192}, {{0, 0}, {1, -1}}).ready, "no allowed toucher refused");
    const auto sparse = make_stage_numa_plan({0, 8192}, 1, {8192, 8192}, {{3, 8}, {7, 32}});
    check(sparse.ready && sparse.nodes[0] == 3 && sparse.nodes[1] == 7, "sparse NUMA node IDs supported");
    std::printf("numa_stage_test: %d checks passed\n", checks);
}
