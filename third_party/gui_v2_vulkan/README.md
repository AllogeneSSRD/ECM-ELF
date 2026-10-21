# GUI v2 Vulkan development dependencies

Vulkan-Headers `v1.3.290` (Apache-2.0 OR MIT as marked in each file) and Volk
`vulkan-sdk-1.3.290.0` (MIT). The headers are the Windows/core/video dependency subset;
Volk is unmodified. Source URLs and SHA256 hashes are in provenance.json.

These files are for the independent v2 target. Builds use no network and need no SDK
installation or Vulkan import library: Volk dynamically loads the system loader.
Running the application requires a Vulkan 1.1 loader and a compatible graphics/present
device. Validation layers are optional development dependencies; `--validation`
requires them explicitly and fails if absent.

The matching ImGui Vulkan backend has one v2-only local patch,
`third_party/imgui_vulkan/acquire-before-use.patch`: platform swapchain images
must not be transitioned before acquisition. The traditional render pass already
performs the initial transition after acquire; the dynamic path discards old
contents using UNDEFINED. The minimized detached-viewport regression records two
validation errors before the patch and zero afterwards. Original and patched
hashes are both recorded in provenance.json. The shared ImGui core and Win32/DX11
backends remain unmodified.
