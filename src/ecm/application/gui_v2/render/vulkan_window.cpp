#include "vulkan_window.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include <algorithm>
#include <stdexcept>

namespace ecmgui_v2 {
namespace {
int create_surface(ImGuiViewport* viewport, ImU64 instance, const void* allocator, ImU64* output) {
    VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = static_cast<HWND>(viewport->PlatformHandleRaw ? viewport->PlatformHandleRaw : viewport->PlatformHandle);
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const auto result = vkCreateWin32SurfaceKHR(reinterpret_cast<VkInstance>(instance), &info,
        static_cast<const VkAllocationCallbacks*>(allocator), &surface);
    *output = reinterpret_cast<ImU64>(surface);
    return result;
}
}
void VulkanWindow::destroy_resources() {
    auto device = context_.device;
    if (ImGui::GetIO().BackendRendererUserData) ImGui_ImplVulkan_Shutdown();
    for (const auto& image : images_) {
        if (image.framebuffer) vkDestroyFramebuffer(device, image.framebuffer, nullptr);
        if (image.view) vkDestroyImageView(device, image.view, nullptr);
        if (image.complete) vkDestroySemaphore(device, image.complete, nullptr);
    }
    images_.clear();
    if (pool_) vkDestroyCommandPool(device, pool_, nullptr);
    if (acquired_) vkDestroySemaphore(device, acquired_, nullptr);
    if (fence_) vkDestroyFence(device, fence_, nullptr);
    if (pass_) vkDestroyRenderPass(device, pass_, nullptr);
    if (swapchain_) vkDestroySwapchainKHR(device, swapchain_, nullptr);
    pool_ = VK_NULL_HANDLE; command_ = VK_NULL_HANDLE; acquired_ = VK_NULL_HANDLE;
    fence_ = VK_NULL_HANDLE; pass_ = VK_NULL_HANDLE; swapchain_ = VK_NULL_HANDLE;
}
void VulkanWindow::shutdown() {
    if (!context_.device) return;
    vkDeviceWaitIdle(context_.device);
    destroy_resources();
}
VulkanWindow::~VulkanWindow() { shutdown(); }
void VulkanWindow::rebuild(uint32_t width, uint32_t height) {
    if (!width || !height) return;
    auto& c = context_;
    check_vk(vkDeviceWaitIdle(c.device), "wait before swapchain rebuild");
    destroy_resources();
    VkSurfaceCapabilitiesKHR caps{};
    check_vk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c.physical, c.surface, &caps), "surface capabilities");
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
        throw std::runtime_error("Vulkan surface does not support color attachments");
    uint32_t count = 0;
    check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(c.physical, c.surface, &count, nullptr), "surface formats");
    if (!count) throw std::runtime_error("No Vulkan surface format available");
    std::vector<VkSurfaceFormatKHR> formats(count);
    check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(c.physical, c.surface, &count, formats.data()), "surface formats");
    auto format = formats.front();
    for (auto candidate : formats)
        if (candidate.format == VK_FORMAT_B8G8R8A8_UNORM && candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            format = candidate;
    if (format.format == VK_FORMAT_UNDEFINED) format.format = VK_FORMAT_B8G8R8A8_UNORM;
    extent_ = caps.currentExtent.width == UINT32_MAX ? VkExtent2D{
        std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height)} : caps.currentExtent;
    if (!extent_.width || !extent_.height) { needs_rebuild = true; return; }
    count = std::max(2u, caps.minImageCount);
    if (caps.maxImageCount && count > caps.maxImageCount)
        throw std::runtime_error("Vulkan surface cannot provide two swapchain images");
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (auto candidate : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                          VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
        if (caps.supportedCompositeAlpha & candidate) { alpha = candidate; break; }
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = c.surface; info.minImageCount = count;
    info.imageFormat = format.format; info.imageColorSpace = format.colorSpace;
    info.imageExtent = extent_; info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform; info.compositeAlpha = alpha;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR; info.clipped = VK_TRUE;
    check_vk(vkCreateSwapchainKHR(c.device, &info, nullptr, &swapchain_), "create swapchain");
    check_vk(vkGetSwapchainImagesKHR(c.device, swapchain_, &count, nullptr), "swapchain images");
    std::vector<VkImage> raw(count);
    check_vk(vkGetSwapchainImagesKHR(c.device, swapchain_, &count, raw.data()), "swapchain images");
    VkAttachmentDescription attachment{};
    attachment.format = format.format; attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1; subpass.pColorAttachments = &reference;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL; dependency.dstSubpass = 0;
    dependency.srcStageMask = dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1; pass.pAttachments = &attachment;
    pass.subpassCount = 1; pass.pSubpasses = &subpass;
    pass.dependencyCount = 1; pass.pDependencies = &dependency;
    check_vk(vkCreateRenderPass(c.device, &pass, nullptr, &pass_), "create render pass");
    images_.resize(count);
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < count; ++i) {
        auto& image = images_[i]; image.image = raw[i];
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image.image; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = format.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check_vk(vkCreateImageView(c.device, &view, nullptr, &image.view), "create image view");
        VkFramebufferCreateInfo frame{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        frame.renderPass = pass_; frame.attachmentCount = 1; frame.pAttachments = &image.view;
        frame.width = extent_.width; frame.height = extent_.height; frame.layers = 1;
        check_vk(vkCreateFramebuffer(c.device, &frame, nullptr, &image.framebuffer), "create framebuffer");
        check_vk(vkCreateSemaphore(c.device, &semaphore, nullptr, &image.complete), "create image completion semaphore");
    }
    check_vk(vkCreateSemaphore(c.device, &semaphore, nullptr, &acquired_), "create acquire semaphore");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check_vk(vkCreateFence(c.device, &fence, nullptr, &fence_), "create frame fence");
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = c.queue_family; pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check_vk(vkCreateCommandPool(c.device, &pool, nullptr, &pool_), "create command pool");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool_; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount = 1;
    check_vk(vkAllocateCommandBuffers(c.device, &allocate, &command_), "allocate command buffer");
    ImGui::GetPlatformIO().Platform_CreateVkSurface = create_surface;
    ImGui_ImplVulkan_InitInfo backend{};
    backend.ApiVersion = c.api_version; backend.Instance = c.instance; backend.PhysicalDevice = c.physical;
    backend.Device = c.device; backend.Queue = c.queue; backend.QueueFamily = c.queue_family;
    backend.DescriptorPoolSize = 64; backend.MinImageCount = std::max(2u, caps.minImageCount);
    backend.ImageCount = count; backend.PipelineInfoMain.RenderPass = pass_;
    backend.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    backend.CheckVkResultFn = backend_check;
    if (!ImGui_ImplVulkan_Init(&backend)) throw std::runtime_error("ImGui Vulkan initialization failed");
    needs_rebuild = false; ++rebuilds;
    std::fprintf(log_, "swapchain: rebuild=%u extent=%ux%u images=%u format=%d present=FIFO\n", rebuilds, extent_.width, extent_.height, count, format.format);
    std::fflush(log_);
}
bool VulkanWindow::render(ImDrawData* draw) {
    if (needs_rebuild || !swapchain_) return false;
    auto& c = context_;
    auto result = vkWaitForFences(c.device, 1, &fence_, VK_TRUE, 1000000000);
    check_vk(result, "wait for frame fence");
    uint32_t index = 0;
    result = vkAcquireNextImageKHR(c.device, swapchain_, 10000000, acquired_, VK_NULL_HANDLE, &index);
    if (result == VK_TIMEOUT || result == VK_NOT_READY) return false;
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { needs_rebuild = true; return false; }
    if (result != VK_SUBOPTIMAL_KHR) check_vk(result, "acquire swapchain image");
    const bool suboptimal = result == VK_SUBOPTIMAL_KHR;
    check_vk(vkResetCommandPool(c.device, pool_, 0), "reset frame commands");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_, &begin), "begin frame");
    VkClearValue clear{}; clear.color = {{.14f, .14f, .14f, 1}};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = pass_; pass.framebuffer = images_[index].framebuffer;
    pass.renderArea.extent = extent_; pass.clearValueCount = 1; pass.pClearValues = &clear;
    vkCmdBeginRenderPass(command_, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw, command_);
    vkCmdEndRenderPass(command_);
    check_vk(vkEndCommandBuffer(command_), "end frame");
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &acquired_; submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command_;
    submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &images_[index].complete;
    check_vk(vkResetFences(c.device, 1, &fence_), "reset frame fence");
    check_vk(vkQueueSubmit(c.queue, 1, &submit, fence_), "submit frame");
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1; present.pWaitSemaphores = &images_[index].complete;
    present.swapchainCount = 1; present.pSwapchains = &swapchain_; present.pImageIndices = &index;
    result = vkQueuePresentKHR(c.queue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || suboptimal) needs_rebuild = true;
    if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) check_vk(result, "present frame");
    return true;
}
}
