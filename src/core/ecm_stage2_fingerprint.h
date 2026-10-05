#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Windows CNG loaded dynamically: no new link dependency for existing builders.
namespace ecm_stage2 {
inline std::string sha256_file(const std::filesystem::path &path) {
    struct Cng {
        HMODULE library=nullptr;
        BCRYPT_ALG_HANDLE algorithm=nullptr;
        BCRYPT_HASH_HANDLE hash=nullptr;
        decltype(&BCryptOpenAlgorithmProvider) open=nullptr;
        decltype(&BCryptGetProperty) property=nullptr;
        decltype(&BCryptCreateHash) create=nullptr;
        decltype(&BCryptHashData) update=nullptr;
        decltype(&BCryptFinishHash) finish=nullptr;
        decltype(&BCryptDestroyHash) destroy=nullptr;
        decltype(&BCryptCloseAlgorithmProvider) close=nullptr;
        ~Cng() {
            if(hash && destroy)destroy(hash);
            if(algorithm && close)close(algorithm,0);
            if(library)FreeLibrary(library);
        }
    } api;
    // Object storage must outlive the CNG handle, including on exceptions.
    std::vector<unsigned char> object;
    struct HashStorage {
        Cng &api;
        ~HashStorage(){if(api.hash){api.destroy(api.hash);api.hash=nullptr;}}
    } storage{api};
    api.library=LoadLibraryW(L"bcrypt.dll");
    if(!api.library)throw std::runtime_error("Windows SHA256 provider unavailable");
#define ECM_CNG_LOAD(field, symbol) \
    api.field=reinterpret_cast<decltype(api.field)>(GetProcAddress(api.library,#symbol)); \
    if(!api.field)throw std::runtime_error("Windows SHA256 API unavailable: " #symbol)
    ECM_CNG_LOAD(open,BCryptOpenAlgorithmProvider);
    ECM_CNG_LOAD(property,BCryptGetProperty);
    ECM_CNG_LOAD(create,BCryptCreateHash);
    ECM_CNG_LOAD(update,BCryptHashData);
    ECM_CNG_LOAD(finish,BCryptFinishHash);
    ECM_CNG_LOAD(destroy,BCryptDestroyHash);
    ECM_CNG_LOAD(close,BCryptCloseAlgorithmProvider);
#undef ECM_CNG_LOAD
    auto check=[](NTSTATUS status){if(status<0)throw std::runtime_error("Windows SHA256 operation failed");};
    check(api.open(&api.algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    ULONG length=0,received=0;
    check(api.property(api.algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&length),sizeof(length),&received,0));
    object.resize(length);
    check(api.create(api.algorithm,&api.hash,object.data(),length,nullptr,0,0));
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("cannot fingerprint: "+path.string());
    std::vector<unsigned char> buffer(65536);
    while(input) {
        input.read(reinterpret_cast<char*>(buffer.data()),buffer.size());
        if(input.gcount())check(api.update(api.hash,buffer.data(),static_cast<ULONG>(input.gcount()),0));
    }
    if(input.bad())throw std::runtime_error("fingerprint read failed: "+path.string());
    unsigned char digest[32];check(api.finish(api.hash,digest,sizeof(digest),0));
    const char *hex="0123456789abcdef";std::string result;
    for(auto c:digest){result+=hex[c>>4];result+=hex[c&15];}
    return result;
}
} // namespace ecm_stage2
