#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::detail {

struct NumaLayerLocation {
    int node = -1;
    int cpu = -1;
};

struct StageNumaPlan {
    bool ready = false;
    std::string reason;
    std::array<int, 2> nodes{-1, -1}, cores{-1, -1};
    std::array<std::vector<std::pair<uint64_t, uint64_t>>, 2> ranges;
    std::array<uint64_t, 2> bytes{};
};

// Place only complete 4 KiB pages owned by one expert. Boundary pages remain
// with the normal toucher, so adjacent layers never fault the same page from
// two sockets. Offsets name the compact RAM complement, not the GGUF mapping.
inline StageNumaPlan make_stage_numa_plan(const std::vector<uint64_t>& offsets, int64_t experts,
                                         const std::vector<uint64_t>& blobs,
                                         const std::vector<NumaLayerLocation>& layers) {
    StageNumaPlan out;
    auto fail = [&](const char* why) { out.reason = why; return out; };
    if (experts <= 0 || layers.empty() || layers.size() != blobs.size() ||
        offsets.size() % (uint64_t) experts || offsets.size() / (uint64_t) experts != layers.size())
        return fail("layer and complement geometry do not match");
    std::vector<int> nodes;
    for (const auto& layer : layers) {
        if (layer.node < 0 || layer.node >= 64 || layer.cpu < 0)
            return fail("a layer has no usable NUMA node and allowed CPU");
        if (std::find(nodes.begin(), nodes.end(), layer.node) == nodes.end()) nodes.push_back(layer.node);
    }
    std::sort(nodes.begin(), nodes.end());
    if (nodes.size() != 2) return fail("stage placement requires exactly two GPU NUMA nodes");
    out.nodes = {nodes[0], nodes[1]};
    constexpr uint64_t page = 4096, absent = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < offsets.size(); ++i) {
        const uint64_t off = offsets[i];
        if (off == absent) continue;
        const size_t layer = i / (uint64_t) experts;
        const uint64_t blob = blobs[layer];
        if (!blob || off > absent - blob) return fail("expert byte range overflows");
        const size_t k = layers[layer].node == nodes[0] ? 0 : 1;
        if (out.cores[k] < 0) out.cores[k] = layers[layer].cpu;
        if (off > absent - (page - 1)) continue;
        const uint64_t begin = (off + page - 1) / page * page;
        const uint64_t end = (off + blob) / page * page;
        if (end > begin) out.ranges[k].emplace_back(begin, end);
    }
    // A node can own no complement slots when its experts all fit in VRAM.
    for (size_t k = 0; k < 2; ++k) {
        if (out.cores[k] < 0)
            for (const auto& layer : layers) if (layer.node == nodes[k]) { out.cores[k] = layer.cpu; break; }
        auto& ranges = out.ranges[k];
        std::sort(ranges.begin(), ranges.end());
        size_t used = 0;
        for (const auto& range : ranges) {
            if (used && range.first <= ranges[used - 1].second)
                ranges[used - 1].second = std::max(ranges[used - 1].second, range.second);
            else ranges[used++] = range;
        }
        ranges.resize(used);
        for (const auto& range : ranges) out.bytes[k] += range.second - range.first;
    }
    size_t a = 0, b = 0;
    while (a < out.ranges[0].size() && b < out.ranges[1].size()) {
        const auto& x = out.ranges[0][a];
        const auto& y = out.ranges[1][b];
        if (std::max(x.first, y.first) < std::min(x.second, y.second))
            return fail("different GPU nodes claim the same expert pages");
        if (x.second <= y.second) ++a; else ++b;
    }
    out.ready = true;
    return out;
}

} // namespace strata::core::detail
