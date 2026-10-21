#include "vulkan_context.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ecmgui_v2 {
void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: VkResult=" + std::to_string(result));
}
void backend_check(VkResult result) {
    if (result < 0) check_vk(result, "ImGui Vulkan backend");
}
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debug(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* owner) {
    auto& self = *static_cast<VulkanContext*>(owner);
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++self.validation_errors;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if (type & (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)) ++self.validation_warnings;
        else ++self.environment_warnings;
    }
    std::fprintf(self.log_, "validation: type=%u %s\n", type, data->pMessage);
    std::fflush(self.log_);
    return VK_FALSE;
}
void VulkanContext::initialize(HWND window, bool validation, bool no_loader, FILE* log) {
    log_ = log;
    if (no_loader || volkInitialize() != VK_SUCCESS)
        throw std::runtime_error("No Vulkan loader available. Use the unchanged ecm_gui.exe (DX11).");
    loader_ = true;
    if (volkGetInstanceVersion() < api_version)
        throw std::runtime_error("GUI v2 requires a Vulkan 1.1 loader and device");
    uint32_t count = 0;
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr), "enumerate instance extensions");
    std::vector<VkExtensionProperties> available(count);
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data()), "enumerate instance extensions");
    std::vector<const char*> extensions{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    if (validation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    for (const char* name : extensions)
        if (std::none_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; }))
            throw std::runtime_error(std::string("Missing Vulkan instance extension: ") + name);
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (validation) {
        check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr), "enumerate validation layers");
        std::vector<VkLayerProperties> layers(count);
        check_vk(vkEnumerateInstanceLayerProperties(&count, layers.data()), "enumerate validation layers");
        if (std::none_of(layers.begin(), layers.end(), [&](const auto& l) { return std::strcmp(l.layerName, layer) == 0; }))
            throw std::runtime_error("Requested VK_LAYER_KHRONOS_validation is unavailable");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ECM-ELF GUI v2";
    app.apiVersion = api_version;
    VkDebugUtilsMessengerCreateInfoEXT debug_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = debug;
    debug_info.pUserData = this;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    info.enabledLayerCount = validation ? 1u : 0u;
    info.ppEnabledLayerNames = validation ? &layer : nullptr;
    info.pNext = validation ? &debug_info : nullptr;
    check_vk(vkCreateInstance(&info, nullptr, &instance), "create Vulkan instance");
    volkLoadInstance(instance);
    if (validation) check_vk(vkCreateDebugUtilsMessengerEXT(instance, &debug_info, nullptr, &messenger_), "create validation messenger");
    VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface_info.hinstance = GetModuleHandleW(nullptr);
    surface_info.hwnd = window;
    check_vk(vkCreateWin32SurfaceKHR(instance, &surface_info, nullptr, &surface), "create Win32 surface");
    check_vk(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate graphics devices");
    std::vector<VkPhysicalDevice> devices(count);
    check_vk(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate graphics devices");
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties candidate_properties{};
        vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
        if (candidate_properties.apiVersion < api_version) continue;
        uint32_t n = 0;
        check_vk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &n, nullptr), "enumerate device extensions");
        std::vector<VkExtensionProperties> device_extensions(n);
        check_vk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &n, device_extensions.data()), "enumerate device extensions");
        if (std::none_of(device_extensions.begin(), device_extensions.end(), [](const auto& e) { return std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0; })) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> families(n);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, families.data());
        for (uint32_t i = 0; i < n; ++i) {
            VkBool32 present = false;
            check_vk(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &present), "query presentation support");
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                physical = candidate; queue_family = i; properties = candidate_properties; break;
            }
        }
        if (physical) break;
    }
    if (!physical) throw std::runtime_error("No Vulkan 1.1 graphics/present device with swapchain support");
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char* swapchain = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = &swapchain;
    check_vk(vkCreateDevice(physical, &device_info, nullptr, &device), "create graphics device");
    volkLoadDevice(device);
    vkGetDeviceQueue(device, queue_family, 0, &queue);
    std::fprintf(log_, "device: %s api=%u.%u.%u driver=%u queue=%u validation=%d\n", properties.deviceName,
        VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion),
        properties.driverVersion, queue_family, validation);
    std::fflush(log_);
}
VulkanContext::~VulkanContext() {
    if (device) { vkDeviceWaitIdle(device); vkDestroyDevice(device, nullptr); }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger_) vkDestroyDebugUtilsMessengerEXT(instance, messenger_, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    if (log_) { std::fprintf(log_, "validation_summary: errors=%u warnings=%u environment_warnings=%u\n", validation_errors, validation_warnings, environment_warnings); std::fflush(log_); }
    if (loader_) volkFinalize();
}
}
