#include "stage2_identity.h"
#include "generated/ecm_config_generated.h"
namespace s2prod {
ResultIdentity read_result_identity(const std::filesystem::path &identity_ini,int worker,bool explicit_config) {
    namespace fs=std::filesystem;
    std::vector<ecm_config::TomlLine> identity_lines;
    if(!ecm_config::read_toml(identity_ini.string(),identity_lines,ecm_config::TomlReadMode::Computation) && (explicit_config || fs::exists(identity_ini)))
        throw std::runtime_error("cannot read result identity TOML");
    const auto identity=ecm_config::read_stage2(ecm_config::cli_entries(identity_lines,worker,true));
    return {identity.result_user,identity.result_computer,identity.submission};
}
}
