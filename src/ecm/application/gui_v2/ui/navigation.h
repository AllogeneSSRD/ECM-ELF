#include "i18n.h"
#pragma once
#include <string>
#include <algorithm>
namespace ecmgui_v2::ui {
inline const char* page_name(int page, bool zh) {
    static const char* keys[]{"worker.title","results","generator","page.probability","gpu","settings","components","go.title"};
    return page>=0&&page<8?text(keys[page],zh):"";
}
inline bool pinned(int mask,int page) { return page>=0&&page<8&&(mask&(1<<page))!=0; }
// Reinsert before/after the target; move hidden right-side pages as stable IDs.
inline bool reorder_page(std::string& order,int source,int target,bool after) {
    if(source<0||source>=8||target<0||target>=8||source==target)return false;
    const auto from=order.find(static_cast<char>('0'+source)),to=order.find(static_cast<char>('0'+target));
    if(from==order.npos||to==order.npos)return false;
    auto next=order;next.erase(from,1);const auto at=next.find(static_cast<char>('0'+target));
    next.insert(at+(after?1:0),1,static_cast<char>('0'+source));if(next==order)return false;
    order=std::move(next);return true;
}
inline std::string right_window(int page,bool zh){return std::string(page_name(page,zh))+"###v2.page."+std::to_string(page);}
// Keep one central page available; reject invalid persisted or drag payload values.
inline bool move_page(int& mask,int& center,int& right,int page,bool to_right) {
    if(page<0||page>=8||page==5||mask<0||mask>=255)return false;
    const int next=to_right?mask|(1<<page):mask&~(1<<page);
    if((next&223)==223)return false;
    mask=next;
    if(to_right){right=page;if(pinned(mask,center))for(int i=0;i<8;++i)if(i!=5&&!pinned(mask,i)){center=i;break;}}
    else {if(!pinned(mask,right))for(int i=0;i<8;++i)if(pinned(mask,i)){right=i;break;}}
    return true;
}
}
