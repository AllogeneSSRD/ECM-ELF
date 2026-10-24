#pragma once
#ifdef ECM_GUI_V2_BASE_PET_ONLY
#include <array>
#include <string_view>
namespace ecmgui_v2::pet {
struct Skin {const char *id,*english,*chinese,*atlas;};
inline constexpr std::array<Skin,1> skins{{
    {"classic","Classic ELY","\u7ecf\u5178 ELY","pet_ely.png"}
}};
inline const Skin* find_skin(std::string_view id) {
    return id==skins[0].id?&skins[0]:nullptr;
}
}
#else
#include "../generated/pet_skins.h"
#endif
