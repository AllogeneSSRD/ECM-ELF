# GUI v2 ImGui Vulkan backend

Unmodified `backends/imgui_impl_vulkan.cpp/.h` from ocornut/imgui commit
`64944b4520b30772de8dbf0b37d0311746477a32`, matching the existing local ImGui core.
MIT license in LICENSE.txt. Source URLs and byte hashes are recorded in
`../gui_v2_vulkan/provenance.json`.

Only the v2 target compiles this backend. The legacy ImGui directory and DX11 target
are unchanged. Win32 Vulkan surface creation is implemented in the v2 host.
