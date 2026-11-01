#pragma once

// TPI16 is the frozen 16k control. TPI32 limits padding in the common wide
// ranges; all capacities are whole CGBN groups of 32 * 32-bit words.
constexpr int s2g_coop_capacity_words(int nw,int tpi)
{
    return tpi!=32 ? 256 : nw<=160 ? 160 : nw<=192 ? 192 : nw<=224 ? 224 : 256;
}
