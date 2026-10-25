#pragma once
#include <cstdio>
#include <string>
// Owned application mirror; logging.h remains the small backend logging API.
class EcmLogMirror {
    FILE *file_=nullptr;
public:
    EcmLogMirror()=default;
    ~EcmLogMirror();
    bool open(const std::string &path);
    EcmLogMirror(const EcmLogMirror &)=delete;
    EcmLogMirror &operator=(const EcmLogMirror &)=delete;
};
