# ecm_gui -- the graphical front-end (docs/DEV_ECM_GUI.md).
#
# Reached from the root CMakeLists.txt only when
#   ECM_BUILD_GUI=ON  AND  WIN32  AND  ${ECM_IMGUI_DIR}/imgui.h exists
# so a machine without the vendored ImGui (or a non-Windows host) simply does not
# get this target -- it can never break the build of ecm / ecm_cuda.
#
# The GUI links NO project code: it is a process supervisor + log/config viewer,
# so no GMP, no OpenCL, no CUDA. System libraries only.

set(ECM_GUI_IMGUI_SOURCES
    ${ECM_IMGUI_DIR}/imgui.cpp
    ${ECM_IMGUI_DIR}/imgui_draw.cpp
    ${ECM_IMGUI_DIR}/imgui_tables.cpp
    ${ECM_IMGUI_DIR}/imgui_widgets.cpp
    ${ECM_IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${ECM_IMGUI_DIR}/backends/imgui_impl_dx11.cpp)
# imgui_demo.cpp is intentionally NOT vendored (see third_party/imgui/README.md):
# copy it in and append it here to get Tools -> ImGui demo window.

add_library(ecm_gui_imgui STATIC ${ECM_GUI_IMGUI_SOURCES})
target_include_directories(ecm_gui_imgui PUBLIC
    ${ECM_IMGUI_DIR}
    ${ECM_IMGUI_DIR}/backends)
target_compile_definitions(ecm_gui_imgui PUBLIC
    UNICODE _UNICODE)
target_compile_options(ecm_gui_imgui PRIVATE /W3)

add_executable(ecm_gui WIN32
    main_win32.cpp
    app.cpp
    gpu_monitor.cpp
    gpu_selftest.cpp
    ini_file.cpp
    localization.cpp
    log_parse.cpp
    platform_win32.cpp
    results.cpp
    worker_proc.cpp
    worker_selftest.cpp
    ${ECM_GUI_SOURCES})
target_include_directories(ecm_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ECM_IMGUI_DIR})
target_link_libraries(ecm_gui PRIVATE
    ecm_gui_imgui
    d3d11 dxgi d3dcompiler dwmapi shell32)
target_compile_options(ecm_gui PRIVATE /W3 /utf-8)
set_target_properties(ecm_gui PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
# Same output rule as every other helper/deliverable: the build root.  The
# localization files are copied next to it so a build-dir run finds them.
ecm_tool_output_root(ecm_gui)
# Localization files are copied next to the executable so a build-dir run finds them.
# This is a dependency-tracked rule, NOT a POST_BUILD step: POST_BUILD only runs when the
# target relinks, so editing an XML alone left the copy beside the exe stale and the GUI
# kept showing the old strings (measured 2026-09-29: a fixed translation did not appear
# after `cmake --build`, and --selftest failed on the stale copy).
# NOTE: add_custom_command(OUTPUT) rejects generator expressions, and ecm_gui lands in the
# build root anyway (ecm_tool_output_root below), so the path is CMAKE_BINARY_DIR.
set(ECM_GUI_LOC_FILES
    ${CMAKE_SOURCE_DIR}/src/gui/localization/english.xml
    ${CMAKE_SOURCE_DIR}/src/gui/localization/chineseSimplified.xml
    ${CMAKE_SOURCE_DIR}/src/gui/localization/README.md)
set(ECM_GUI_LOC_DIR ${CMAKE_BINARY_DIR}/localization)
add_custom_command(OUTPUT ${ECM_GUI_LOC_DIR}/english.xml
                          ${ECM_GUI_LOC_DIR}/chineseSimplified.xml
                          ${ECM_GUI_LOC_DIR}/README.md
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/src/gui/localization ${ECM_GUI_LOC_DIR}
    DEPENDS ${ECM_GUI_LOC_FILES}
    COMMENT "ecm_gui: sync localization/ next to the executable")
add_custom_target(ecm_gui_localization DEPENDS
    ${ECM_GUI_LOC_DIR}/english.xml
    ${ECM_GUI_LOC_DIR}/chineseSimplified.xml
    ${ECM_GUI_LOC_DIR}/README.md)
add_dependencies(ecm_gui ecm_gui_localization)

# ---------------------------------------------------------------------------
# Test fixtures/helpers of the GUI. They are tiny and have no dependency on ImGui
# or on any project library, so they build with plain system headers.
#
#   ecm_gui_fake_worker  : a stand-in for ecm_cuda.exe (scripted output, crashes,
#                          hangs, spawns a grandchild) used by --worker-selftest
#                          and by tools/test/test_gui_workers.ps1
#   ecm_gui_log_parse_test : unit test for the pure parsing layer
# ---------------------------------------------------------------------------

add_executable(ecm_gui_fake_worker fake_worker.cpp)
target_compile_options(ecm_gui_fake_worker PRIVATE /W3 /utf-8)
set_target_properties(ecm_gui_fake_worker PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
ecm_tool_output_root(ecm_gui_fake_worker)

add_executable(ecm_gui_results_test results_test.cpp results.cpp)
target_include_directories(ecm_gui_results_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ecm_gui_results_test PRIVATE /W3 /utf-8)
set_target_properties(ecm_gui_results_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
ecm_tool_output_root(ecm_gui_results_test)

add_executable(ecm_gui_log_parse_test log_parse_test.cpp log_parse.cpp)
target_include_directories(ecm_gui_log_parse_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ecm_gui_log_parse_test PRIVATE /W3 /utf-8)
set_target_properties(ecm_gui_log_parse_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
ecm_tool_output_root(ecm_gui_log_parse_test)
