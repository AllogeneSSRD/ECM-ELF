#pragma once
#include "vulkan_context.h"
#include <vector>

struct ImDrawData;
namespace ecmgui_v2 {
class VulkanWindow {
public:
    explicit VulkanWindow(VulkanContext& context, FILE* log) : context_(context), log_(log) {}
    ~VulkanWindow();
    VulkanWindow(const VulkanWindow&) = delete;
    VulkanWindow& operator=(const VulkanWindow&) = delete;
    void rebuild(uint32_t width, uint32_t height);
    bool render(ImDrawData* draw);
    void shutdown();
    bool needs_rebuild = true;
    unsigned rebuilds = 0;
    float clear_color[4]{.14f, .14f, .14f, 1};
private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkSemaphore complete = VK_NULL_HANDLE;
    };
    VulkanContext& context_;
    FILE* log_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkRenderPass pass_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkSemaphore acquired_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    std::vector<Image> images_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    void destroy_resources(bool full = true);
};
}
