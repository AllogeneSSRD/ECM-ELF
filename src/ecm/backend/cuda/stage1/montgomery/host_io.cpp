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

#include "host_io.h"
#include "../../runtime/cuda_ecm_shim.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "ecm/platform/stage1_progress.h"
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



/**
 * Get checkpoint filename for given N (more readable format)
 * Uses bit length and first/last few hex chars as identifier
 */
char* get_checkpoint_filename(const mpz_t N) {
  static char filename[1024];
  const auto path=ecm_stage1_checkpoint_path().empty()?std::string(".mont_stage1.ckpt"):ecm_stage1_curve_record(0);
  snprintf(filename,sizeof(filename),"%s",path.c_str());(void)N;return filename;
}

/**
 * Save checkpoint to file
 */
int save_checkpoint(const char *filename,
                    const checkpoint_header_t *header,
                    const uint32_t *data,
                    size_t data_size) {
  uint64_t sum=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,header,sizeof(*header));
  sum=ecm_stage1_checksum(sum,data,data_size);
  std::string bytes(reinterpret_cast<const char *>(header),sizeof(*header));
  bytes.append(reinterpret_cast<const char *>(data),data_size);
  bytes.append(reinterpret_cast<const char *>(&sum),sizeof(sum));
  if(!ecm_stage1_write_blob(filename,bytes))return ECM_ERROR;
  outputf(OUTPUT_VERBOSE, "Checkpoint saved: s_partial=%lu/%lu (%.1f%%)\n",
          header->s_partial, header->s_num_bits,
          100.0 * header->s_partial / header->s_num_bits);
  return ECM_NO_FACTOR_FOUND;
}

/**
 * Load checkpoint from file if it exists
 * Returns 0 if successful, -1 if file doesn't exist or is invalid
 */
int load_checkpoint(const char *filename,
                    checkpoint_header_t *header,
                    uint32_t **data_ptr,
                    size_t *data_size_ptr) {
  std::string bytes;if(!ecm_stage1_read_blob(filename,bytes))return -1;
  checkpoint_header_t temp_header;
  if(bytes.size()<sizeof(temp_header)+sizeof(uint64_t))return -1;
  memcpy(&temp_header,bytes.data(),sizeof(temp_header));
  if(temp_header.magic!=CHECKPOINT_MAGIC || temp_header.version!=CHECKPOINT_VERSION || !temp_header.data_size ||
     temp_header.data_size>bytes.size()-sizeof(temp_header)-sizeof(uint64_t) ||
     bytes.size()!=sizeof(temp_header)+temp_header.data_size+sizeof(uint64_t))return -1;
  uint64_t sum;memcpy(&sum,bytes.data()+bytes.size()-sizeof(sum),sizeof(sum));
  if(sum!=ecm_stage1_checksum(ECM_STAGE1_CHECKSUM_SEED,bytes.data(),bytes.size()-sizeof(sum)))return -1;
  auto *data=static_cast<uint32_t *>(malloc(temp_header.data_size));if(!data)return -1;
  memcpy(data,bytes.data()+sizeof(temp_header),temp_header.data_size);
  *header = temp_header;
  *data_ptr = data;
  *data_size_ptr = temp_header.data_size;

  time_t now = time(NULL);
  time_t age = now - temp_header.timestamp;
  outputf(OUTPUT_NORMAL, "Checkpoint loaded: s_partial=%lu/%lu (%.1f%%), age=%ld seconds\n",
          temp_header.s_partial, temp_header.s_num_bits,
          100.0 * temp_header.s_partial / temp_header.s_num_bits, age);

  return ECM_NO_FACTOR_FOUND;
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


#if ECM_CUDA_ENABLE_PRAC
bool resident_checkpoint_write(const std::string &path, ResidentCheckpoint h,
                                      const std::vector<uint32_t> &data) {
    h.checksum = ecm_prac_hash(data.data(), data.size() * sizeof(uint32_t));
    std::ostringstream out(std::ios::binary|std::ios::out);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(data.data()), h.bytes);
    return out && ecm_stage1_write_blob(path,out.str());
}
bool resident_checkpoint_restore(const std::string &path,ResidentCheckpoint &h,
                                 std::vector<uint32_t> &data,uint64_t *sigma,bool prac) {
    const auto total=h.total,bytes=h.bytes;
    const auto bits=h.bits,tpi=h.tpi,curves=h.curves,torsion=h.torsion;
    bool resumed=false;
    std::string saved;ecm_stage1_read_blob(path,saved);
    std::istringstream in(saved,std::ios::binary|std::ios::in); ResidentCheckpoint old{};
    if (in.read(reinterpret_cast<char *>(&old), sizeof(old))) {
        bool compatible = old.magic == h.magic && old.version == h.version && old.b1 == h.b1 &&
            old.scalar == h.scalar && old.modulus == h.modulus && old.identity == h.identity &&
            old.total == total && old.bytes == bytes && old.algorithm == h.algorithm && old.domain == 1 &&
            old.bits == bits && old.tpi == tpi && old.curves == curves && old.torsion == torsion &&
            old.next <= total && (prac || old.next >= 1) && old.sigma <= UINT64_MAX - curves;
        if (compatible && in.read(reinterpret_cast<char *>(data.data()), bytes) &&
            in.peek() == std::char_traits<char>::eof() && ecm_prac_hash(data.data(), bytes) == old.checksum) {
            h = old; *sigma = h.sigma; resumed = true;
            outputf(OUTPUT_ALWAYS, "GPU: resident checkpoint resumed (next=%llu/%llu, sigma=%llu)\n",
                (unsigned long long)h.next, (unsigned long long)total, (unsigned long long)*sigma);
        } else throw std::runtime_error("resident checkpoint parameters/payload mismatch");
    }
    if(!saved.empty() && !resumed)throw std::runtime_error("invalid resident checkpoint payload");
    return resumed;
}
#endif
} // namespace ecm_cuda_mont_host
