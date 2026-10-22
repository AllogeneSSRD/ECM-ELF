cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ecm_gui_v2_standalone LANGUAGES C CXX)
    get_filename_component(ECM_GUI_V2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    include("${ECM_GUI_V2_ROOT}/tools/build/internal/check_ecm_config.cmake")
    add_custom_target(ecm_config_check
        COMMAND ${CMAKE_COMMAND} "-DECM_CONFIG_ROOT=${ECM_GUI_V2_ROOT}"
                -P "${ECM_GUI_V2_ROOT}/tools/build/internal/check_ecm_config.cmake")
else()
    set(ECM_GUI_V2_ROOT "${CMAKE_SOURCE_DIR}")
endif()
if(NOT WIN32)
    message(FATAL_ERROR "GUI v2 currently requires Windows")
endif()
set(ECM_GUI_V2_IMGUI "${ECM_GUI_V2_ROOT}/third_party/imgui" CACHE PATH "Unmodified pinned ImGui core")
set(ECM_GUI_V2_VULKAN "${ECM_GUI_V2_ROOT}/third_party/gui_v2_vulkan" CACHE PATH "Pinned Vulkan headers and Volk")
set(ECM_GUI_V2_BACKEND "${ECM_GUI_V2_ROOT}/third_party/imgui_vulkan" CACHE PATH "Matching ImGui Vulkan backend")
foreach(_file "${ECM_GUI_V2_IMGUI}/imgui.h" "${ECM_GUI_V2_VULKAN}/include/vulkan/vulkan_core.h"
              "${ECM_GUI_V2_VULKAN}/volk/volk.h" "${ECM_GUI_V2_BACKEND}/imgui_impl_vulkan.cpp")
    if(NOT EXISTS "${_file}")
        message(FATAL_ERROR "GUI v2 dependency missing: ${_file}")
    endif()
endforeach()
add_library(ecm_gui_v2_backend STATIC
    "${ECM_GUI_V2_IMGUI}/imgui.cpp" "${ECM_GUI_V2_IMGUI}/imgui_draw.cpp"
    "${ECM_GUI_V2_IMGUI}/imgui_tables.cpp" "${ECM_GUI_V2_IMGUI}/imgui_widgets.cpp"
    "${ECM_GUI_V2_IMGUI}/backends/imgui_impl_win32.cpp"
    "${ECM_GUI_V2_IMGUI}/backends/imgui_impl_dx11.cpp"
    "${ECM_GUI_V2_BACKEND}/imgui_impl_vulkan.cpp" "${ECM_GUI_V2_VULKAN}/volk/volk.c")
target_include_directories(ecm_gui_v2_backend PUBLIC "${ECM_GUI_V2_IMGUI}" "${ECM_GUI_V2_IMGUI}/backends"
    "${ECM_GUI_V2_BACKEND}" "${ECM_GUI_V2_VULKAN}/include" "${ECM_GUI_V2_VULKAN}/volk")
target_compile_definitions(ecm_gui_v2_backend PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK VK_USE_PLATFORM_WIN32_KHR
    WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
target_compile_features(ecm_gui_v2_backend PUBLIC cxx_std_17)
add_executable(ecm_gui_v2 WIN32 main_win32.cpp render/dx11_window.cpp render/vulkan_context.cpp render/vulkan_window.cpp
    ui/theme.cpp ui/fonts.cpp ui/widgets.cpp ui/i18n.cpp app.cpp views/pages.cpp views/settings_window.cpp ui_state.cpp pet/window.cpp gpu_details.cpp)
include("${ECM_GUI_V2_ROOT}/tools/build/internal/check_gui_v2_config.cmake")
add_custom_target(ecm_gui_v2_config_check
    COMMAND ${CMAKE_COMMAND} "-DECM_GUI_V2_ROOT=${ECM_GUI_V2_ROOT}" -P "${ECM_GUI_V2_ROOT}/tools/build/internal/check_gui_v2_config.cmake")
set(_old "${ECM_GUI_V2_ROOT}/src/ecm/application/gui")
add_library(ecm_gui_v2_models STATIC "${_old}/worker_controller.cpp" "${_old}/worker_proc.cpp"
    "${_old}/toml_file.cpp" "${_old}/log_parse.cpp" "${_old}/platform_win32.cpp"
    "${_old}/results.cpp" "${_old}/worktodo_gen.cpp" "${_old}/probability_model.cpp" "${_old}/gpu_monitor.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/ui/localization_xml.cpp")
target_include_directories(ecm_gui_v2_models PUBLIC "${ECM_GUI_V2_ROOT}/include" "${_old}")
target_compile_features(ecm_gui_v2_models PUBLIC cxx_std_17)
target_compile_definitions(ecm_gui_v2_models PUBLIC UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(ecm_gui_v2_models PRIVATE /W3 /utf-8 /EHsc)
target_link_libraries(ecm_gui_v2_models PUBLIC user32 gdi32 comdlg32 shell32)
target_link_libraries(ecm_gui_v2 PRIVATE ecm_gui_v2_backend ecm_gui_v2_models user32 gdi32 shell32 dwmapi ole32 gdiplus comctl32 d3d11 dxgi d3dcompiler)
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/assets/pet_ely.png"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ECM_GUI_V2_ROOT}/assets/gui_v2/pet_ely.png" "${CMAKE_BINARY_DIR}/assets/pet_ely.png"
    DEPENDS "${ECM_GUI_V2_ROOT}/assets/gui_v2/pet_ely.png")
add_custom_target(ecm_gui_v2_assets DEPENDS "${CMAKE_BINARY_DIR}/assets/pet_ely.png")
foreach(_notice CLASSIC-ELY-NOTICE.md ELYTerminal-LICENSE.txt)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/assets/${_notice}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ECM_GUI_V2_ROOT}/assets/gui_v2/${_notice}" "${CMAKE_BINARY_DIR}/assets/${_notice}"
        DEPENDS "${ECM_GUI_V2_ROOT}/assets/gui_v2/${_notice}")
endforeach()
add_custom_target(ecm_gui_v2_pet_notices DEPENDS "${CMAKE_BINARY_DIR}/assets/CLASSIC-ELY-NOTICE.md" "${CMAKE_BINARY_DIR}/assets/ELYTerminal-LICENSE.txt")
add_dependencies(ecm_gui_v2_assets ecm_gui_v2_pet_notices)
foreach(_skin miss_pink_elf herrscher_of_human_ego hi_love_elf)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/assets/pet_ely_${_skin}.png"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ECM_GUI_V2_ROOT}/assets/gui_v2/pet_ely_${_skin}.png" "${CMAKE_BINARY_DIR}/assets/pet_ely_${_skin}.png"
        DEPENDS "${ECM_GUI_V2_ROOT}/assets/gui_v2/pet_ely_${_skin}.png")
    add_custom_target(ecm_gui_v2_skin_${_skin} DEPENDS "${CMAKE_BINARY_DIR}/assets/pet_ely_${_skin}.png")
    add_dependencies(ecm_gui_v2_assets ecm_gui_v2_skin_${_skin})
endforeach()
add_dependencies(ecm_gui_v2 ecm_gui_v2_assets)
add_custom_target(ecm_gui_v2_localization ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/localization" "${CMAKE_BINARY_DIR}/localization/v2")
add_dependencies(ecm_gui_v2 ecm_gui_v2_localization)
target_compile_options(ecm_gui_v2 PRIVATE /W4 /utf-8 /EHsc)
target_link_options(ecm_gui_v2 PRIVATE /MANIFEST:EMBED "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
set_property(TARGET ecm_gui_v2 APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
target_compile_options(ecm_gui_v2_backend PRIVATE /W3 /utf-8)
set_target_properties(ecm_gui_v2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_dependencies(ecm_gui_v2 ecm_config_check ecm_gui_v2_config_check)
message(STATUS "gui_v2: independent Win32 target; DX11 default, Vulkan diagnostic opt-in via Volk")
add_executable(ecm_gui_v2_fake_worker EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/fake_worker.cpp")
target_link_libraries(ecm_gui_v2_fake_worker PRIVATE ecm_gui_v2_models)
target_link_options(ecm_gui_v2_fake_worker PRIVATE /MANIFEST:EMBED "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
set_property(TARGET ecm_gui_v2_fake_worker APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
target_compile_options(ecm_gui_v2_fake_worker PRIVATE /utf-8 /EHsc)
add_executable(ecm_gui_v2_owner_probe EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/owner_probe.cpp")
target_compile_features(ecm_gui_v2_owner_probe PRIVATE cxx_std_17)
target_compile_options(ecm_gui_v2_owner_probe PRIVATE /utf-8 /EHsc)
add_executable(ecm_gui_v2_state_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/state_test.cpp" ui_state.cpp pet/window.cpp)
target_link_libraries(ecm_gui_v2_state_test PRIVATE ecm_gui_v2_models gdiplus ole32)
target_compile_options(ecm_gui_v2_state_test PRIVATE /utf-8 /EHsc /UNDEBUG)
add_executable(ecm_gui_v2_app_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/app_test.cpp"
    app.cpp views/pages.cpp views/settings_window.cpp ui_state.cpp ui/theme.cpp ui/fonts.cpp ui/widgets.cpp ui/i18n.cpp gpu_details.cpp)
add_dependencies(ecm_gui_v2_app_test ecm_gui_v2_localization)
target_link_libraries(ecm_gui_v2_app_test PRIVATE ecm_gui_v2_models ecm_gui_v2_backend dwmapi)
target_compile_options(ecm_gui_v2_app_test PRIVATE /utf-8 /EHsc /UNDEBUG)
target_link_options(ecm_gui_v2_app_test PRIVATE /MANIFEST:EMBED "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
set_property(TARGET ecm_gui_v2_app_test APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
