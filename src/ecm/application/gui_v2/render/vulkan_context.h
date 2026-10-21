#pragma once
#include <windows.h>
#include <volk.h>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <string>

namespace ecmgui_v2 {
void check_vk(VkResult result, const char* operation);
void backend_check(VkResult result);

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    void initialize(HWND window, bool validation, bool no_loader, FILE* log);
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    uint32_t api_version = VK_API_VERSION_1_1;
    VkPhysicalDeviceProperties properties{};
    std::atomic<unsigned> validation_errors{0};
    std::atomic<unsigned> validation_warnings{0};
    std::atomic<unsigned> environment_warnings{0};
private:
    bool loader_ = false;
    FILE* log_ = nullptr;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void* owner);
};
}
