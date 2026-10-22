#pragma once
namespace ecmgui_v2::ui {
inline const char* page_name(int page, bool zh) {
    static const char* en[]{"Workers","Results","Generator","Probability","GPU","Settings","Components"};
    static const char* cn[]{"任务","结果","生成器","概率","GPU","设置","组件"};
    return page>=0&&page<7?(zh?cn[page]:en[page]):"";
}
inline bool pinned(int mask,int page) { return page>=0&&page<7&&(mask&(1<<page))!=0; }
// Keep one central page available; reject invalid persisted or drag payload values.
inline bool move_page(int& mask,int& center,int& right,int page,bool to_right) {
    if(page<0||page>=7||mask<0||mask>=127)return false;
    const int next=to_right?mask|(1<<page):mask&~(1<<page);
    if(next==127)return false;
    mask=next;
    if(to_right){right=page;if(pinned(mask,center))for(int i=0;i<7;++i)if(!pinned(mask,i)){center=i;break;}}
    else {center=page;if(!pinned(mask,right))for(int i=0;i<7;++i)if(pinned(mask,i)){right=i;break;}}
    return true;
}
}
