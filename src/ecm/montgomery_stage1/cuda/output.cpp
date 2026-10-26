/* cgbn_stage1.h: header for CGBN (GPU) based ecm stage 1.

Copyright 2021 Seth Troisi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include "output.h"
#include "host_policy.h"
#include <stdexcept>
#include "../../backend/cuda/runtime/cuda_ecm_shim.h"
#include "ecm/stage1/progress.h"
#include "ecm/platform/logging.h"
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#if ECM_CUDA_ENABLE_PRAC
#include "prac/ecm_prac_plan.h"
#include <sstream>
#include <stdexcept>
#endif
namespace ecm_cuda_mont_host {
bool stdout_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

void print_progress(double pct, uint64_t s_partial, uint64_t this_batch,
                           double per_curve_s, double elapsed_ms, double remaining_s,
                           bool newline, const char *unit) {
    const auto text=ecm_stage1_gpu_progress::line(pct,s_partial,this_batch,unit,
                                                per_curve_s,elapsed_ms/1000.0,remaining_s);

    if (newline) {
        // Redirected / log mode: a full timestamped line, mirrored to screen.log
        // via the outputf → ecm_ts_vfprintf path. No ANSI colour in the log.
        outputf(OUTPUT_ALWAYS,"%s\n",text.c_str());
    } else {
        // Interactive terminal: in-place update, coloured (ANSI; colour is
        // configurable via ecm.toml progress_color).
        fprintf(stdout,
                "\r%s%s%s",
                ecm_log_progress_color_code(),text.c_str(),
                ecm_log_progress_color_reset());
        fflush(stdout);
    }
}

static bool emit_progress_line(int n) {
    return ((n < 3) ||
            (n < 30 && n % 10 == 0) ||
            (n < 500 && n % 100 == 0) ||
            (n < 5000 && n % 1000 == 0) ||
            (n % 10000 == 0));
}

static const double kProgressLineGapMs = 200.0;
bool should_emit_progress_line(int n, double now_ms, double last_ms) {
    if (now_ms - last_ms >= kProgressLineGapMs) return true;
    return emit_progress_line(n);
}



FILE* open_dump_csv(const char *filename) {
  FILE *f = fopen(filename, "w");
  if (!f) {
    outputf(OUTPUT_ALWAYS, "Warning: Could not open dump file '%s' for writing\n", filename);
    return NULL;
  }

  fprintf(f,
          "stage,batch_index,s_partial,batch_size,sigma,curve_index,BITS,TPI,word0,word1,word2,word3,word4\n");
  return f;
}

void dump_curve_state_csv(FILE *f,
                          const char *stage,
                          int batch_index,
                          uint64_t s_partial,
                          uint64_t batch_size,
                          uint32_t sigma,
                          uint32_t BITS,
                          uint32_t TPI,
                          const uint32_t *data,
                          uint32_t curves,
                          size_t limbs_per) {
  if (!f) {
    return;
  }

  for (uint32_t i = 0; i < curves; i++) {
    const uint32_t *datum = data + (5 * i * limbs_per);
    fprintf(f, "%s,%d," "%" PRIu64 "," "%" PRIu64 ",%u,%u,%u,%u,",
            stage, batch_index, s_partial, batch_size, sigma, i, BITS, TPI);
    for (uint32_t word = 0; word < 5; word++) {
      if (word != 0) {
        fputc(',', f);
      }
      fputs("0x", f);
      const uint32_t *field = datum + (word * limbs_per);
      for (size_t limb = limbs_per; limb > 0; limb--) {
        fprintf(f, "%08x", field[limb - 1]);
      }
    }
    fputc('\n', f);
  }

  fflush(f);
}


void write_prac_window_csv(const PracWindow &w,uint64_t work,uint64_t b1,uint64_t sigma,uint32_t curves,uint32_t bits,const std::vector<uint32_t> &data) {
        FILE *dump = fopen("window_q.csv", "wb");
        if (!dump) throw std::runtime_error("cannot write PRAC window diagnostic CSV");
        bool failed = fprintf(dump, "kind,b1,first,count,work,sigma,x,z\n") < 0;
        size_t limbs = bits / 32;
        for (uint32_t i = 0; i < curves; ++i) {
            failed |= fprintf(dump, "partial_product,%llu,%llu,%llu,%llu,%llu",
                (unsigned long long)b1, (unsigned long long)w.first, (unsigned long long)w.count,
                (unsigned long long)work, (unsigned long long)(sigma + i)) < 0;
            for (size_t word : {size_t(3), size_t(4)}) {
                failed |= fputs(",0x", dump) < 0;
                const uint32_t *field = data.data() + (7 * size_t(i) + word) * limbs;
                for (size_t j = limbs; j > 0; --j) failed |= fprintf(dump, "%08x", field[j - 1]) < 0;
            }
            failed |= fputc('\n', dump) == EOF;
        }
        failed |= fclose(dump) != 0;
        if (failed) throw std::runtime_error("PRAC window diagnostic CSV write failed");
}

}
