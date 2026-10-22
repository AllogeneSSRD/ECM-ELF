#pragma once
#include <string>
#include <utility>
namespace ecmgui_v2::ui {
// Only UI-owned messages are translated. Raw worker diagnostics retain their source text.
inline std::string message(const std::string& text,bool zh) {
    if(!zh)return text;
    static const std::pair<const char*,const char*> exact[]{
        {"Open a workflow TOML to manage workers.","请在设置中打开工作流 TOML，再管理计算任务。"},
        {"Stop all workers before changing workflow.","请先停止所有任务，再切换工作流。"},
        {"Workflow opened; worker control requires queue-owner protocol 1.","已打开工作流；计算程序须支持队列拥有权协议 1。"},
        {"Save or reload workflow changes before starting workers.","请先保存或重载工作流修改，再启动任务。"},
        {"Stop workers and save workflow before generating.","请先停止任务并保存工作流，再生成队列。"},
        {"Select a valid target worker.","请选择有效的目标 worker。"},
        {"CPU supports e/m0 with explicit curve count; GPU recommendation and m2/m3 are unavailable.","CPU 仅支持 e/m0 与显式曲线数；无法使用 GPU 推荐和 m2/m3。"},
        {"Cannot read target queue","无法读取目标队列"},
        {"Preview expired; generate again.","预览已失效，请重新生成。"},
        {"Stop workers before editing workflow.","请先停止任务，再编辑工作流。"},
        {"Stop workers before saving workflow.","请先停止任务，再保存工作流。"},
        {"Legacy GUI preferences are preserved; use v2 appearance controls.","旧 GUI 偏好保持不变，请使用 v2 外观设置。"},
        {"Unsaved workflow edit","工作流修改尚未保存"},
        {"Workflow changed externally; reload before saving.","工作流已被外部修改，请重载后再保存。"},
        {"UI preferences saved","已保存界面偏好"},
        {"Moved to right tools","已移到右侧工具栏"},{"Moved to left navigation","已移回左侧导航"},
        {"Probability snapshot ready","概率快照已就绪"},{"Snapshot CSV copied","已复制快照 CSV"},
        {"Keep at least one central page","至少保留一个中央页面"},
        {"Running","运行中"},{"Idle","空闲"},{"Starting","启动中"},{"Stopping","停止中"},
        {"Stopped","已停止"},{"Restarting","重启中"},{"Error","错误"},{"Queue empty","队列为空"},{"QueueEmpty","队列为空"}
    };
    for(const auto& v:exact)if(text==v.first)return v.second;
    static const std::pair<const char*,const char*> prefixes[]{
        {"Worker executable missing: ","找不到计算程序："},
        {"Worker lacks queue ownership protocol 1; install the updated worker: ","计算程序不支持队列拥有权协议 1，请更新："},
        {"Preview ready: ","预览已就绪："},{"Queue appended: ","已追加队列："},
        {"Worker started: ","已启动 worker："},{"Safe stop requested: ","已请求安全停止："},
        {"GPU profile unavailable: ","无法获取 GPU 配置："}
    };
    for(const auto& v:prefixes)if(text.rfind(v.first,0)==0)return std::string(v.second)+text.substr(std::char_traits<char>::length(v.first));
    return text;
}
}
