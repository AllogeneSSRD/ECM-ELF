# Authoritative translation-unit inventory. This file is also runnable with
# cmake -DECM_CUDA_SOURCES_OUTPUT=<build>/cuda_sources.json -P <this file>.
# Keep source order: it is the existing compile/link order, not a glob.
cmake_minimum_required(VERSION 3.20)
set(ECM_STAGE1_COMMON_SOURCES
    src/ecm/core/group_order/group_order.cpp
    src/ecm/service/diagnostics/order_analysis.cpp
    src/ecm/core/stage1/selection.cpp
    src/io/results.cpp
    src/ecm/service/stage1/task.cpp
    src/ecm/service/stage1/execution_contract.cpp
    src/ecm/service/stage1/preparation.cpp
    src/ecm/service/stage1/execution.cpp
    src/ecm/backend/cpu/stage1/edwards/workflow.cpp src/ecm/backend/cpu/stage1/montgomery/workflow.cpp src/ecm/backend/cpu/stage1/execution_support.cpp
    src/io/normalize.cpp
    src/ecm/application/cli/group_order.cpp
    src/io/cli.cpp
    src/io/files.cpp
    src/sync/backup.cpp
    src/sync/p95.cpp
    src/io/save.cpp
    src/main.cpp
    src/ecm/service/execution/ecm_params_init.cpp
    src/ecm/backend/opencl/stage1/montgomery/checkpoint.cpp
    src/io/stage1_checkpoint.cpp src/ecm/platform/stage1_stop.cpp
    src/ecm/service/stage1/ecm_gpu_common.cpp
    src/io/config.cpp
    src/io/worktodo.cpp
    src/sync/stage2.cpp
    src/ecm/application/configuration/ecm_expr.cpp
    src/io/prime95.cpp
    src/ecm/platform/logging.cpp
    src/ecm/platform/stage1_progress.cpp
    src/ecm/backend/cpu/stage1/edwards/ecm_edwards_cpu.cpp
    src/io/prime95_save.cpp
    src/ecm/backend/cpu/stage1/edwards/simd_edwards.cpp
    src/ecm/backend/cpu/arithmetic/ifma/simd_mont_ifma.cpp
    src/ecm/backend/cpu/stage1/montgomery/ecm_mont_cpu.cpp
    src/ecm/backend/cpu/stage1/montgomery/simd_mont_curve.cpp
    src/ecm/backend/cpu/stage1/montgomery/checkpoint.cpp
    src/ecm/core/stage1/ecm_stage1_exp.cpp
    src/io/exponent_cache.cpp)
set(ECM_STAGE1_EDWARDS_SOURCES
    src/ecm/backend/cuda/stage1/edwards/runtime.cpp
    src/ecm/backend/cuda/stage1/edwards/edwards_runtime.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi4.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi8.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi16_2560_4096.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi16_4608_6144.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi16_6656_8192.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi32_9216_12288.cu
    src/ecm/backend/cuda/stage1/edwards/instances/cgbn_edwards_instances_tpi32_13312_17408.cu)
set(ECM_STAGE1_MONT_SOURCES
    src/ecm/backend/cuda/stage1/montgomery/runtime.cu
    src/ecm/backend/cuda/stage1/montgomery/registry.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g0.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g1.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g2.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g3.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g4.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g5.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g6.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g0.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g1.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g2.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g3.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g4.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g5.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g6.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g0.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g1.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g2.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g3.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g4.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g5.cu
    src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g6.cu)
set(ECM_STAGE1_CUDA_SOURCES
    src/ecm/application/configuration/runtime.cpp
    src/ecm/backend/cuda/factory.cpp
    src/ecm/backend/cuda/stage1/montgomery/workflow.cpp
    src/ecm/backend/cuda/stage1/edwards/workflow.cpp
    src/ecm/backend/cuda/stage1/edwards/checkpoint.cpp
    src/ecm/backend/cuda/stage1/edwards/host_math.cpp
    src/ecm/backend/cuda/stage1/montgomery/host_math.cpp
    src/ecm/backend/cuda/stage1/montgomery/checkpoint.cpp src/ecm/backend/cuda/stage1/montgomery/output.cpp
    src/ecm/backend/cuda/stage1/montgomery/host_policy.cpp
    ${ECM_STAGE1_EDWARDS_SOURCES}
    ${ECM_STAGE1_MONT_SOURCES})
set(ECM_STAGE1_PRAC_SOURCES
    src/ecm/backend/cuda/stage1/montgomery/prac/runtime.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_kernels.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_small.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi16.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi32.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi32_9216_10240.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi32_11264_12288.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi32_13312_14336.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_tpi32_15360_16384.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_alternate32.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_single_add.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/cgbn_stage1_prac_constants.cu
    src/ecm/backend/cuda/stage1/montgomery/prac/ecm_prac_plan.cpp)
set(ECM_STAGE2_HOST_SOURCES
    src/io/files.cpp
    src/io/save.cpp
    src/ecm/platform/logging.cpp
    src/ecm/application/configuration/runtime.cpp
    src/ecm/core/group_order/group_order.cpp
    src/ecm/service/diagnostics/order_analysis.cpp
    src/ecm/platform/fingerprint.cpp
    src/io/stage2_checkpoint.cpp
    src/io/results.cpp
    src/ecm/core/stage1/selection.cpp
    src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp
    src/ecm/service/planning/stage2.cpp
    src/ecm/service/stage2/task.cpp
    src/ecm/service/stage2/input.cpp
    src/ecm/platform/process/stage2_curve_process.cpp
    src/ecm/backend/cuda/stage2/host_policy.cpp
    src/ecm/application/configuration/stage2_identity.cpp
    src/ecm/application/configuration/ecm_expr.cpp
    src/io/worktodo.cpp
    src/io/config.cpp)
set(ECM_STAGE2_ARITHMETIC_SOURCES src/ecm/backend/cuda/stage2/device/arithmetic_kernels.cu)
set(ECM_STAGE2_NTT_SOURCES
    src/ecm/backend/cuda/ntt/shape.cpp
    src/ecm/backend/cuda/ntt/runtime.cpp
    src/ecm/backend/cuda/ntt/graph.cpp
    src/ecm/backend/cuda/ntt/checks.cpp
    src/ecm/backend/cuda/ntt/device/transform_kernels.cu)
set(ECM_STAGE2_ENGINE_SOURCES
    src/ecm/backend/cuda/stage2/constants.cpp
    src/ecm/backend/cuda/ntt/module_dispatch.cpp
    src/ecm/backend/cuda/stage2/context.cpp
    src/ecm/backend/cuda/stage2/host_wait.cpp
    src/ecm/backend/cuda/stage2/options.cpp
    src/ecm/backend/cuda/stage2/resources.cpp
    src/ecm/backend/cuda/stage2/plan.cpp
    src/ecm/backend/cuda/stage2/engine.cpp
    src/ecm/backend/cuda/stage2/batch.cpp
    src/ecm/backend/cuda/stage2/points.cpp
    src/ecm/backend/cuda/stage2/product.cpp
    src/ecm/backend/cuda/stage2/tree.cpp
    src/ecm/backend/cuda/stage2/remainder.cpp
    src/ecm/backend/cuda/stage2/s4.cpp
    src/ecm/backend/cuda/stage2/factors.cpp
    src/ecm/backend/cuda/stage2/checks.cpp
    src/ecm/backend/cuda/stage2/diagnostics.cpp
    src/ecm/backend/cuda/stage2/tune.cpp)
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
    tools/build/internal/build_directory.ps1
    CMakeLists.txt
    tools/build/internal/cuda_modules.cmake
    tools/build/internal/generate_cuda_modules.py
    tools/build/internal/run_compile_jobs.py
    tools/build/internal/cuda_timing.ps1
    tools/build/internal/cuda_timing.cmake
    tools/build/internal/timed_nvcc.py
    tools/build/internal/release_profiles.ps1
    tools/build/internal/cuda_sources.cmake
    tools/build/internal/cuda_build_state.ps1
    tools/build/internal/cuda_build_verify.ps1
    tools/build/internal/cuda_build_profile.ps1
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
                if(_property MATCHES "^(INTERFACE_)?LINK_LIBRARIES$" AND TARGET "${_item}")
                    list(APPEND _pending "${_item}")
                endif()
                string(REGEX MATCHALL "\\$<TARGET_OBJECTS:[^>]+>" _objects "${_item}")
                foreach(_object IN LISTS _objects)
                    string(REGEX REPLACE "^\\$<TARGET_OBJECTS:([^>]+)>$" "\\1" _name "${_object}")
                    list(APPEND _pending "${_name}")
                endforeach()
                # Only link expressions name further targets. Ordinary source
                # paths such as src/ecm/... must not turn their directory names
                # into unrelated executable targets. Explicit TARGET_OBJECTS
                # references above remain valid in both source and link lists.
                if(NOT _property MATCHES "^(INTERFACE_)?LINK_LIBRARIES$" OR NOT _item MATCHES "^\\$<")
                    continue()
                endif()
                # Conditional link expressions retain a conservative superset.
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
