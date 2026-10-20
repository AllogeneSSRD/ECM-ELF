# Authoritative translation-unit inventory. This file is also runnable with
# cmake -DECM_CUDA_SOURCES_OUTPUT=<build>/cuda_sources.json -P <this file>.
# Keep source order: it is the existing compile/link order, not a glob.
cmake_minimum_required(VERSION 3.20)
set(ECM_STAGE1_COMMON_SOURCES
    src/core/ecm_driver.cpp
    src/core/ecm_params_init.cpp
    src/core/ecm_checkpoint.cpp
    src/core/ecm_stage1_checkpoint_runtime.cpp
    src/core/ecm_save.cpp
    src/core/ecm_gpu_common.cpp
    src/core/ecm_queue_config.cpp
    src/core/ecm_worktodo.cpp
    src/core/ecm_stage2_transfer.cpp
    src/core/ecm_expr.cpp
    src/core/p95_transfer.cpp
    src/core/p95_worktodo.cpp
    src/opencl_ecm_log.cpp
    src/cpu/ecm_edwards_cpu.cpp
    src/cpu/ecm_edwards_save.cpp
    src/cpu/simd_edwards.cpp
    src/cpu/simd_mont_ifma.cpp
    src/cpu/ecm_mont_cpu.cpp
    src/cpu/simd_mont_curve.cpp
    src/core/ecm_mont_ckpt.cpp
    src/core/ecm_stage1_exp.cpp
    src/core/ecm_stage1_exp_cache.cpp)
set(ECM_STAGE1_CUDA_SOURCES
    src/opencl_ecm_runtime_config.cpp
    src/cuda/ecm_cuda_backend.cu
    src/cuda/ecm_cuda_edwards.cu
    src/cuda/ecm_cuda_edwards_naf.cu
    kernels/cuda/cgbn_edwards_instances_tpi4.cu
    kernels/cuda/cgbn_edwards_instances_tpi8.cu
    kernels/cuda/cgbn_edwards_instances_tpi16_2560_4096.cu
    kernels/cuda/cgbn_edwards_instances_tpi16_4608_6144.cu
    kernels/cuda/cgbn_edwards_instances_tpi16_6656_8192.cu
    kernels/cuda/cgbn_edwards_instances_tpi32_9216_12288.cu
    kernels/cuda/cgbn_edwards_instances_tpi32_13312_17408.cu
    kernels/cuda/cgbn_stage1.cu
    kernels/cuda/cgbn_stage1_kernels_tpi4.cu
    kernels/cuda/cgbn_stage1_kernels_tpi8.cu
    kernels/cuda/cgbn_stage1_kernels_tpi16.cu
    kernels/cuda/cgbn_stage1_kernels_tpi32.cu
    kernels/cuda/cgbn_stage1_kernels_suyama.cu
    kernels/cuda/cgbn_stage1_kernels_param2.cu
    kernels/cuda/cgbn_stage1_kernels_suyama_tpi16.cu
    kernels/cuda/cgbn_stage1_kernels_suyama_tpi32.cu
    kernels/cuda/cgbn_stage1_kernels_param2_tpi16.cu
    kernels/cuda/cgbn_stage1_kernels_param2_tpi32.cu)
set(ECM_STAGE1_PRAC_SOURCES
    kernels/cuda/cgbn_stage1_prac_kernels.cu
    kernels/cuda/cgbn_stage1_prac_small.cu
    kernels/cuda/cgbn_stage1_prac_tpi16.cu
    kernels/cuda/cgbn_stage1_prac_tpi32.cu
    kernels/cuda/cgbn_stage1_prac_tpi32_9216_10240.cu
    kernels/cuda/cgbn_stage1_prac_tpi32_11264_12288.cu
    kernels/cuda/cgbn_stage1_prac_tpi32_13312_14336.cu
    kernels/cuda/cgbn_stage1_prac_tpi32_15360_16384.cu
    kernels/cuda/cgbn_stage1_prac_alternate32.cu
    kernels/cuda/cgbn_stage1_prac_single_add.cu
    kernels/cuda/cgbn_stage1_prac_constants.cu
    src/core/ecm_prac_plan.cpp)
set(ECM_STAGE2_HOST_SOURCES
    src/core/ecm_cuda_stage2_main.cpp
    src/core/ecm_expr.cpp
    src/core/ecm_worktodo.cpp
    src/core/ecm_queue_config.cpp)
set(ECM_STAGE2_PRODUCTION_SOURCES src/cuda/ecm_cuda_stage2.cu)
set(ECM_STAGE2_DEVELOPMENT_SOURCES tools/bench/ecm_cuda_stage2_dev.cu)
set(ECM_CUDA_SOURCE_GROUPS
    STAGE1_COMMON STAGE1_CUDA STAGE1_PRAC
    STAGE2_HOST STAGE2_PRODUCTION STAGE2_DEVELOPMENT)
set(ECM_CUDA_BUILD_INPUTS
    CMakeLists.txt
    tools/build/internal/cuda_sources.cmake
    tools/build/internal/cuda_build_state.ps1
    tools/build/internal/check_ecm_config.cmake
    tools/build/internal/cuda_toolchain.ps1
    tools/build/internal/gmp_dependency.ps1
    tools/gen/generate_ecm_config.py
    config/ecm_options.json
    config/ecm_config.generated.json)

function(ecm_cuda_write_inventory output)
    set(_json "{\n  \"schema\": 1,\n  \"groups\": {")
    set(_separator "")
    foreach(_group IN LISTS ECM_CUDA_SOURCE_GROUPS)
        string(APPEND _json "${_separator}\n    \"${_group}\": [")
        set(_comma "")
        foreach(_source IN LISTS ECM_${_group}_SOURCES)
            string(APPEND _json "${_comma}\"${_source}\"")
            set(_comma ", ")
        endforeach()
        string(APPEND _json "]")
        set(_separator ",")
    endforeach()
    string(APPEND _json "\n  },\n  \"build_inputs\": [")
    set(_comma "")
    foreach(_source IN LISTS ECM_CUDA_BUILD_INPUTS)
        string(APPEND _json "${_comma}\"${_source}\"")
        set(_comma ", ")
    endforeach()
    string(APPEND _json "]\n}\n")
    # Avoid making an unchanged generated inventory appear newer than objects.
    if(EXISTS "${output}")
        file(READ "${output}" _previous)
        if(_previous STREQUAL _json)
            return()
        endif()
    endif()
    file(WRITE "${output}" "${_json}")
endfunction()

# Export the *actual* target graph, including linked object/static modules and
# $<TARGET_OBJECTS:...> sources. Imported libraries are separately identified by
# the toolchain/dependency manifests, not mistaken for project object owners.
function(ecm_cuda_collect_targets target output)
    set(_pending "${target}")
    set(_visited "")
    while(_pending)
        list(POP_FRONT _pending _target)
        if(_target IN_LIST _visited OR NOT TARGET "${_target}")
            continue()
        endif()
        get_target_property(_imported "${_target}" IMPORTED)
        if(_imported)
            continue()
        endif()
        list(APPEND _visited "${_target}")
        foreach(_property SOURCES INTERFACE_SOURCES LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_items "${_target}" "${_property}")
            foreach(_item IN LISTS _items)
                if(TARGET "${_item}")
                    list(APPEND _pending "${_item}")
                endif()
                string(REGEX MATCHALL "\\$<TARGET_OBJECTS:[^>]+>" _objects "${_item}")
                foreach(_object IN LISTS _objects)
                    string(REGEX REPLACE "^\\$<TARGET_OBJECTS:([^>]+)>$" "\\1" _name "${_object}")
                    list(APPEND _pending "${_name}")
                endforeach()
                # Non-imported target names can also be inside conditional link
                # expressions. Keep a conservative superset, never omit objects.
                string(REGEX REPLACE "(LINK_ONLY|BUILD_INTERFACE|INSTALL_INTERFACE|TARGET_NAME_IF_EXISTS|TARGET_OBJECTS):" ";" _link_tokens "${_item}")
                string(REGEX MATCHALL "[A-Za-z_][A-Za-z0-9_.:+-]*" _tokens "${_link_tokens}")
                foreach(_token IN LISTS _tokens)
                    if(TARGET "${_token}")
                        list(APPEND _pending "${_token}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endwhile()
    set(${output} "${_visited}" PARENT_SCOPE)
endfunction()

function(ecm_cuda_write_target_graph)
    set(_json "{\n  \"schema\": 1,\n  \"targets\": {")
    set(_comma "")
    foreach(_root ecm_cuda ecm_cuda_stage2)
        if(TARGET "${_root}")
            ecm_cuda_collect_targets("${_root}" _targets)
            string(JOIN "\", \"" _names ${_targets})
            string(APPEND _json "${_comma}\n    \"${_root}\": [\"${_names}\"]")
            set(_comma ",")
        endif()
    endforeach()
    string(APPEND _json "\n  }\n}\n")
    file(WRITE "${CMAKE_BINARY_DIR}/cuda_target_graph.json" "${_json}")
endfunction()

if(DEFINED ECM_CUDA_SOURCES_OUTPUT)
    ecm_cuda_write_inventory("${ECM_CUDA_SOURCES_OUTPUT}")
endif()
