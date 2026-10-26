#pragma once
#include "runtime.h"
#include "../../cuda/arithmetic/cgbn/types.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace ecm_cuda_stage1 {
inline void check_cuda(cudaError_t rc,const char *operation) {
    if(rc!=cudaSuccess)throw RunError(Failure::Device,
        std::string(operation)+": "+cudaGetErrorString(rc));
}
inline void check_cgbn(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
  // check for cgbn errors

  if(cgbn_error_report_check(report)) {
    fprintf (stderr, "\n");
    fprintf (stderr, "CGBN error occurred: %s\n", cgbn_error_string(report));

    if(report->_instance!=0xFFFFFFFF) {
      fprintf (stderr, "Error reported by instance %d", report->_instance);
      if(report->_blockIdx.x!=0xFFFFFFFF)
        fprintf (stderr, ", blockIdx=(%d, %d, %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      if(report->_threadIdx.x!=0xFFFFFFFF)
        fprintf (stderr, ", threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
      fprintf (stderr, "\n");
    }
    else {
      fprintf (stderr, "Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      fprintf (stderr, "threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
    }
    if(file!=NULL)
      fprintf (stderr, "file %s, line %d\n", file, line);
    throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,"CGBN Stage1 arithmetic error");
  }
}

// Every resource is adopted immediately; cleanup also covers failed subsequent
// allocations, rejected restored state, arithmetic errors and host exceptions.
struct MontResources {
    uint32_t *scalar=nullptr,*host=nullptr,*dump=nullptr;
    uint32_t *device_scalar=nullptr,*device_data=nullptr;
    cgbn_error_report_t *report=nullptr;
    cudaEvent_t start=nullptr,batch=nullptr,stop=nullptr;
    FILE *dump_file=nullptr;
    MontResources()=default;
    MontResources(const MontResources &)=delete;
    MontResources &operator=(const MontResources &)=delete;
    ~MontResources() {
        if(device_scalar)cudaFree(device_scalar);
        if(device_data)cudaFree(device_data);
        if(report)cgbn_error_report_free(report);
        if(start)cudaEventDestroy(start);
        if(batch)cudaEventDestroy(batch);
        if(stop)cudaEventDestroy(stop);
        if(dump_file)std::fclose(dump_file);
        std::free(scalar);std::free(host);std::free(dump);
    }
};
}
