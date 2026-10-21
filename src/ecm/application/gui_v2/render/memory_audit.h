#pragma once
#include "vulkan_context.h"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace ecmgui_v2 {
// Optional acceptance probe for this process's explicit VkDeviceMemory allocations.
// This is not total GPU/driver memory. The single-window renderer owns one UI thread.
class MemoryAudit {
public:
    explicit MemoryAudit(bool enabled) : enabled_(enabled) {
        if (!enabled_) return;
        if (owner_) throw std::runtime_error("Only one Vulkan allocation audit may be active");
        allocate_ = vkAllocateMemory; free_ = vkFreeMemory; owner_ = this;
        vkAllocateMemory = allocate; vkFreeMemory = release;
    }
    ~MemoryAudit() {
        if (!enabled_) return;
        vkAllocateMemory = allocate_; vkFreeMemory = free_; owner_ = nullptr;
    }
    MemoryAudit(const MemoryAudit&) = delete;
    MemoryAudit& operator=(const MemoryAudit&) = delete;
    VkDeviceSize bytes() const { return bytes_; }
    size_t allocations() const { return memory_.size(); }
    void describe(FILE* log) const {
        std::vector<VkDeviceSize> blocks;
        for (const auto& block : memory_) blocks.push_back(block.second);
        std::sort(blocks.begin(), blocks.end());
        std::fprintf(log, "gpu_blocks:");
        for (const auto size : blocks) std::fprintf(log, " %llu", static_cast<unsigned long long>(size));
        std::fprintf(log, "\n");
    }
private:
    bool enabled_;
    VkDeviceSize bytes_ = 0;
    std::unordered_map<VkDeviceMemory, VkDeviceSize> memory_;
    PFN_vkAllocateMemory allocate_ = nullptr;
    PFN_vkFreeMemory free_ = nullptr;
    inline static MemoryAudit* owner_ = nullptr;
    static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo* info,
        const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
        auto& self = *owner_;
        const auto result = self.allocate_(device, info, callbacks, memory);
        if (result == VK_SUCCESS) {
            try { self.memory_.emplace(*memory, info->allocationSize); }
            catch (...) { self.free_(device, *memory, callbacks); *memory = VK_NULL_HANDLE; return VK_ERROR_OUT_OF_HOST_MEMORY; }
            self.bytes_ += info->allocationSize;
        }
        return result;
    }
    static VKAPI_ATTR void VKAPI_CALL release(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
        auto& self = *owner_;
        const auto found = self.memory_.find(memory);
        if (found != self.memory_.end()) { self.bytes_ -= found->second; self.memory_.erase(found); }
        self.free_(device, memory, callbacks);
    }
};
}
