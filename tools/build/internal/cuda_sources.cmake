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
set(ECM_STAGE1_EDWARDS_SOURCES
    src/cuda/stage1/edwards.cpp
    src/cuda/stage1/edwards_naf.cpp
    src/cuda/stage1/edwards_runtime.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi4.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi8.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi16_2560_4096.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi16_4608_6144.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi16_6656_8192.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi32_9216_12288.cu
    kernels/cuda/stage1/edwards/cgbn_edwards_instances_tpi32_13312_17408.cu)
set(ECM_STAGE1_MONT_SOURCES
    src/cuda/stage1/montgomery.cu
    kernels/cuda/stage1/montgomery/registry.cu
    kernels/cuda/stage1/montgomery/instances_param3_g0.cu
    kernels/cuda/stage1/montgomery/instances_param3_g1.cu
    kernels/cuda/stage1/montgomery/instances_param3_g2.cu
    kernels/cuda/stage1/montgomery/instances_param3_g3.cu
    kernels/cuda/stage1/montgomery/instances_param3_g4.cu
    kernels/cuda/stage1/montgomery/instances_param3_g5.cu
    kernels/cuda/stage1/montgomery/instances_param3_g6.cu
    kernels/cuda/stage1/montgomery/instances_param0_g0.cu
    kernels/cuda/stage1/montgomery/instances_param0_g1.cu
    kernels/cuda/stage1/montgomery/instances_param0_g2.cu
    kernels/cuda/stage1/montgomery/instances_param0_g3.cu
    kernels/cuda/stage1/montgomery/instances_param0_g4.cu
    kernels/cuda/stage1/montgomery/instances_param0_g5.cu
    kernels/cuda/stage1/montgomery/instances_param0_g6.cu
    kernels/cuda/stage1/montgomery/instances_param2_g0.cu
    kernels/cuda/stage1/montgomery/instances_param2_g1.cu
    kernels/cuda/stage1/montgomery/instances_param2_g2.cu
    kernels/cuda/stage1/montgomery/instances_param2_g3.cu
    kernels/cuda/stage1/montgomery/instances_param2_g4.cu
    kernels/cuda/stage1/montgomery/instances_param2_g5.cu
    kernels/cuda/stage1/montgomery/instances_param2_g6.cu)
set(ECM_STAGE1_CUDA_SOURCES
    src/opencl_ecm_runtime_config.cpp
    src/cuda/backend.cpp
    ${ECM_STAGE1_EDWARDS_SOURCES}
    ${ECM_STAGE1_MONT_SOURCES})
set(ECM_STAGE1_PRAC_SOURCES
    kernels/cuda/optional/prac/cgbn_stage1_prac_kernels.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_small.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi16.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi32.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi32_9216_10240.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi32_11264_12288.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi32_13312_14336.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_tpi32_15360_16384.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_alternate32.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_single_add.cu
    kernels/cuda/optional/prac/cgbn_stage1_prac_constants.cu
    src/core/ecm_prac_plan.cpp)
set(ECM_STAGE2_HOST_SOURCES
    src/core/ecm_cuda_stage2_main.cpp
    src/core/ecm_expr.cpp
    src/core/ecm_worktodo.cpp
    src/core/ecm_queue_config.cpp)
set(ECM_STAGE2_ARITHMETIC_SOURCES kernels/cuda/stage2/arithmetic_kernels.cu)
set(ECM_STAGE2_NTT_SOURCES
    src/cuda/ntt/shape.cpp
    src/cuda/ntt/runtime.cpp
    src/cuda/ntt/graph.cpp
    src/cuda/ntt/checks.cpp
    kernels/cuda/ntt/transform_kernels.cu)
set(ECM_STAGE2_ENGINE_SOURCES
    src/cuda/stage2/context.cpp
    src/cuda/stage2/options.cpp
    src/cuda/stage2/resources.cpp
    src/cuda/stage2/plan.cpp
    src/cuda/stage2/engine.cpp
    src/cuda/stage2/batch.cpp
    src/cuda/stage2/points.cpp
    src/cuda/stage2/product.cpp
    src/cuda/stage2/tree.cpp
    src/cuda/stage2/remainder.cpp
    src/cuda/stage2/s4.cpp
    src/cuda/stage2/factors.cpp
    src/cuda/stage2/checks.cpp
    src/cuda/stage2/diagnostics.cpp
    src/cuda/stage2/tune.cpp)
set(ECM_STAGE2_PRODUCTION_SOURCES
    ${ECM_STAGE2_ENGINE_SOURCES}
    ${ECM_STAGE2_ARITHMETIC_SOURCES}
    ${ECM_STAGE2_NTT_SOURCES})
set(ECM_STAGE2_DEVELOPMENT_SOURCES tools/experiments/cuda/stage2_tree/worker.cu)
set(ECM_EXPERIMENT_STAGE2_BSGS_SOURCES
    tools/experiments/cuda/stage2_bsgs/cgbn_stage2.cu
    tools/experiments/cuda/stage2_bsgs/cgbn_stage2_kernels_tpi4.cu
    tools/experiments/cuda/stage2_bsgs/cgbn_stage2_kernels_tpi8.cu)
set(ECM_EXPERIMENT_STAGE2_TREE_SOURCES tools/experiments/cuda/stage2_tree/main.cu)
set(ECM_EXPERIMENT_NTT_SOURCES tools/experiments/cuda/ntt_variants/main.cu)
set(ECM_EXPERIMENT_CUFFT_SOURCES tools/experiments/cuda/ntt_variants/cufft_probe.cu)
set(ECM_EXPERIMENT_GIANT_SOURCES tools/experiments/cuda/stage2_tree/giant_variant.cu)
set(ECM_CUDA_SOURCE_GROUPS
    STAGE1_COMMON STAGE1_CUDA STAGE1_EDWARDS STAGE1_MONT STAGE1_PRAC
    STAGE2_HOST STAGE2_PRODUCTION STAGE2_DEVELOPMENT STAGE2_ENGINE STAGE2_ARITHMETIC STAGE2_NTT
    EXPERIMENT_STAGE2_BSGS EXPERIMENT_STAGE2_TREE EXPERIMENT_NTT EXPERIMENT_CUFFT EXPERIMENT_GIANT)
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
