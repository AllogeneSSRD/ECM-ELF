#pragma once
namespace stage2_pack {
struct Stats {
    unsigned long long contiguous=0,gather=0,reverse=0,launches=0,words=0,avoided_memsets=0;
};
extern Stats stats;
bool enabled();
}
