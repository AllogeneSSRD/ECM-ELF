#pragma once
struct PlainScaleStats {
    bool requested=false,enabled=false;
    unsigned long long coefficients=0,h2d_bytes=0,check_d2h_bytes=0,checked_words=0;
    double seconds=0;
    const char *fallback="none";
};
