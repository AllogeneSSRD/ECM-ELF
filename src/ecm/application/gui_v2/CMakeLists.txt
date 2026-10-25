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
    ui/theme.cpp ui/fonts.cpp ui/widgets.cpp ui/i18n.cpp app.cpp views/pages.cpp views/settings_window.cpp views/gpu_clock_control.cpp views/group_order.cpp group_order_job.cpp ui_state.cpp pet/window.cpp gpu_details.cpp gpu_clocks.cpp)
option(ECM_GUI_V2_BASE_PET_ONLY "Only expose the distributed Classic ELY pet" OFF)
if(ECM_GUI_V2_BASE_PET_ONLY)
    target_compile_definitions(ecm_gui_v2 PRIVATE ECM_GUI_V2_BASE_PET_ONLY)
endif()
include("${ECM_GUI_V2_ROOT}/tools/build/internal/check_gui_v2_config.cmake")
add_custom_target(ecm_gui_v2_config_check
    COMMAND ${CMAKE_COMMAND} "-DECM_GUI_V2_ROOT=${ECM_GUI_V2_ROOT}" -P "${ECM_GUI_V2_ROOT}/tools/build/internal/check_gui_v2_config.cmake")
set(_old "${ECM_GUI_V2_ROOT}/src/ecm/application/gui")
add_library(ecm_gui_v2_models STATIC "${_old}/worker_controller.cpp" "${_old}/worker_proc.cpp"
    "${_old}/toml_file.cpp" "${_old}/log_parse.cpp" "${_old}/platform_win32.cpp"
    "${_old}/results.cpp" "${_old}/worktodo_gen.cpp" "${_old}/probability_model.cpp" "${_old}/gpu_monitor.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/ui/localization_xml.cpp")
target_sources(ecm_gui_v2_models PRIVATE "${ECM_GUI_V2_ROOT}/src/io/files.cpp")
target_include_directories(ecm_gui_v2_models PUBLIC "${ECM_GUI_V2_ROOT}/include" "${ECM_GUI_V2_ROOT}/src" "${_old}")
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
target_link_libraries(ecm_gui_v2_owner_probe PRIVATE ecm_gui_v2_models)
target_compile_features(ecm_gui_v2_owner_probe PRIVATE cxx_std_17)
target_compile_options(ecm_gui_v2_owner_probe PRIVATE /utf-8 /EHsc)
add_executable(ecm_gui_v2_state_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/state_test.cpp" ui_state.cpp pet/window.cpp)
target_link_libraries(ecm_gui_v2_state_test PRIVATE ecm_gui_v2_models gdiplus ole32)
target_compile_options(ecm_gui_v2_state_test PRIVATE /utf-8 /EHsc /UNDEBUG)
add_executable(ecm_gui_v2_app_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/app_test.cpp"
    app.cpp views/pages.cpp views/settings_window.cpp views/gpu_clock_control.cpp views/group_order.cpp group_order_job.cpp ui_state.cpp ui/theme.cpp ui/fonts.cpp ui/widgets.cpp ui/i18n.cpp gpu_details.cpp gpu_clocks.cpp)
add_dependencies(ecm_gui_v2_app_test ecm_gui_v2_localization)
target_link_libraries(ecm_gui_v2_app_test PRIVATE ecm_gui_v2_models ecm_gui_v2_backend dwmapi)
target_compile_options(ecm_gui_v2_app_test PRIVATE /utf-8 /EHsc /UNDEBUG)
target_link_options(ecm_gui_v2_app_test PRIVATE /MANIFEST:EMBED "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
set_property(TARGET ecm_gui_v2_app_test APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
add_library(ecm_gui_v2_fake_nvml SHARED EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/fake_nvml_clocks.cpp")
add_library(ecm_gui_v2_fake_nvml_readonly SHARED EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/fake_nvml_clocks.cpp")
add_executable(ecm_gui_v2_fake_clock_query EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/fake_clock_query.cpp")
target_compile_definitions(ecm_gui_v2_fake_nvml_readonly PRIVATE NO_LOCK_EXPORTS)
add_executable(ecm_gui_v2_gpu_clock_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/gpu_clock_test.cpp"
    app.cpp views/pages.cpp views/settings_window.cpp views/gpu_clock_control.cpp views/group_order.cpp group_order_job.cpp ui_state.cpp ui/theme.cpp ui/fonts.cpp ui/widgets.cpp ui/i18n.cpp gpu_details.cpp gpu_clocks.cpp)
add_dependencies(ecm_gui_v2_gpu_clock_test ecm_gui_v2_localization ecm_gui_v2_fake_nvml ecm_gui_v2_fake_nvml_readonly ecm_gui_v2_fake_clock_query)
target_link_libraries(ecm_gui_v2_gpu_clock_test PRIVATE ecm_gui_v2_models ecm_gui_v2_backend dwmapi)
target_compile_options(ecm_gui_v2_gpu_clock_test PRIVATE /utf-8 /EHsc /UNDEBUG)

# CPU diagnostics live in a separate cancellable process; the GUI stays GMP-free.
if(NOT TARGET ecm_group_order)
    add_subdirectory("${ECM_GUI_V2_ROOT}/src/ecm/core/group_order" "${CMAKE_BINARY_DIR}/group_order_core")
endif()
add_executable(ecm_group_order_ui group_order_runner.cpp group_order_job.cpp
    "${ECM_GUI_V2_ROOT}/src/ecm/service/diagnostics/order_analysis.cpp")
target_compile_features(ecm_group_order_ui PRIVATE cxx_std_17)
target_compile_definitions(ecm_group_order_ui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(ecm_group_order_ui PRIVATE /utf-8 /EHsc)
target_link_options(ecm_group_order_ui PRIVATE /MANIFEST:EMBED "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
set_property(TARGET ecm_group_order_ui APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
target_link_libraries(ecm_group_order_ui PRIVATE ecm_group_order)
add_custom_command(TARGET ecm_group_order_ui POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ECM_GROUP_ORDER_GMP_ROOT}/bin/gmp-10.dll" "$<TARGET_FILE_DIR:ecm_group_order_ui>")
add_dependencies(ecm_gui_v2 ecm_group_order_ui)

option(ECM_GUI_V2_LOCAL_PARI "Bundle the native PARI library and SEA tables" ON)
set(ECM_DIAGNOSTICS_LOCAL_PARI ${ECM_GUI_V2_LOCAL_PARI} CACHE BOOL "Bundle local PARI" FORCE)
set(ECM_DIAGNOSTICS_GMP_ROOT "${ECM_GROUP_ORDER_GMP_ROOT}" CACHE PATH "Diagnostic host GMP prefix" FORCE)
if(NOT TARGET ecm_diagnostics_host)
    add_subdirectory("${ECM_GUI_V2_ROOT}/src/ecm/service/diagnostics" "${CMAKE_BINARY_DIR}/diagnostics")
endif()
add_dependencies(ecm_group_order_ui ecm_diagnostics_runtime)
add_executable(ecm_gui_v2_group_order_test EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/group_order_test.cpp" group_order_job.cpp)
target_compile_features(ecm_gui_v2_group_order_test PRIVATE cxx_std_17)
target_include_directories(ecm_gui_v2_group_order_test PRIVATE "${ECM_GUI_V2_ROOT}/include" "${ECM_GUI_V2_ROOT}/src")
target_compile_definitions(ecm_gui_v2_group_order_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(ecm_gui_v2_group_order_test PRIVATE /utf-8 /EHsc /UNDEBUG)
add_dependencies(ecm_gui_v2_group_order_test ecm_group_order_ui)
add_executable(ecm_gui_v2_native_timeout_fixture EXCLUDE_FROM_ALL "${ECM_GUI_V2_ROOT}/tools/test/gui_v2/native_order_timeout_fixture.cpp")
target_compile_features(ecm_gui_v2_native_timeout_fixture PRIVATE cxx_std_17)
