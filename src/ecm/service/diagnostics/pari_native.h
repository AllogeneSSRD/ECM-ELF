#pragma once
#include "../../core/group_order/group_order.h"
#include <filesystem>
#include <functional>
#include <memory>

namespace ecm_diagnostics {
// Isolated diagnostic host only. UI and compute workers do not load PARI.
// Optional local PARI 2.17.3 / Win64 C ABI; no GP interpreter or child process.
class NativeOrder {
public:
    using GroupCheckpoint=std::function<void(const ecm_group_order::Analysis&)>;
    using FactorCheckpoint=std::function<void(const ecm_group_order::Factorization&)>;
    explicit NativeOrder(const std::filesystem::path& directory);
    ~NativeOrder();
    NativeOrder(const NativeOrder&)=delete;
    NativeOrder& operator=(const NativeOrder&)=delete;
    bool available()const;
    const std::string& failure()const;
    ecm_group_order::Analysis group(const ecm_group_order::CurveRequest&,const GroupCheckpoint&);
    ecm_group_order::Factorization factor(const ecm_group_order::Integer&,const FactorCheckpoint&);
private:
    struct Library;
    std::unique_ptr<Library> library_;
};
}
