# GUI v2 Vulkan development dependencies

Vulkan-Headers `v1.3.290` (Apache-2.0 OR MIT as marked in each file) and Volk
`vulkan-sdk-1.3.290.0` (MIT). The headers are the Windows/core/video dependency subset;
Volk is unmodified. Source URLs and SHA256 hashes are in provenance.json.

These files are for the independent v2 target. Builds use no network and need no SDK
installation or Vulkan import library: Volk dynamically loads the system loader.
Running the application requires a Vulkan 1.1 loader and a compatible graphics/present
device. Validation layers are optional development dependencies; `--validation`
requires them explicitly and fails if absent.
