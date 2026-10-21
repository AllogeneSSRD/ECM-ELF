from pathlib import Path
import html
import json
import re

root = Path(__file__).resolve().parents[2]
source = root / 'docs/development/GUI_MODERNIZATION_DRAFT.md'
target = root / 'docs/web/gui-v2-vulkan-plan-20261021.html'
text = source.read_text(encoding='utf-8')


def inline(value):
    slots = []
    def code(match):
        slots.append('<code>' + html.escape(match.group(1)) + '</code>')
        return f'@@CODE{len(slots)-1}@@'
    value = re.sub(r'`([^`]+)`', code, value)
    value = html.escape(value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
    for i, slot in enumerate(slots):
        value = value.replace(f'@@CODE{i}@@', slot)
    return value


def markdown(value):
    out, paragraph, list_type = [], [], None
    def flush():
        if paragraph:
            out.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()
    def close_list():
        nonlocal list_type
        if list_type:
            out.append(f'</{list_type}>')
            list_type = None
    for line in value.splitlines():
        if not line.strip():
            flush()
            close_list()
            continue
        heading = re.match(r'^(#{1,4}) (.+)$', line)
        bullet = re.match(r'^(- |\d+\. )(.+)$', line)
        if heading:
            flush()
            close_list()
            level = min(4, len(heading[1]) + 1)
            out.append(f'<h{level}>' + inline(heading[2]) + f'</h{level}>')
        elif bullet:
            flush()
            kind = 'ul' if bullet[1] == '- ' else 'ol'
            if list_type != kind:
                close_list()
                out.append(f'<{kind}>')
                list_type = kind
            out.append('<li>' + inline(bullet[2]) + '</li>')
        else:
            close_list()
            paragraph.append(line)
    flush()
    close_list()
    return '\n'.join(out)


phases = []
for match in re.finditer(r'^### (P[0-6]) ([^\n]+)\n\n(.*?)(?=^### P[0-6] |^## 7\.)', text, re.M | re.S):
    body = match[3].strip()
    work, _, gate = body.partition('\n\n退出条件：')
    if '\n\n依赖：' in gate:
        gate = gate.split('\n\n依赖：')[0]
    phases.append({'id': match[1], 'title': match[2], 'work': work, 'gate': gate})
assert len(phases) == 7
progress = text.split('## 实施进度', 1)[1].split('## 1.', 1)[0].strip() if '## 实施进度' in text else '尚未开始实施。'
status = re.search(r'^状态：([^。]+)', text, re.M).group(1)
for phase in phases:
    phase['state'] = '已完成' if re.search(r'^- ' + phase['id'] + r'：已完成', progress, re.M) else '待实施'

phase_buttons = ''.join(f'<button class="phase-button" data-phase="{i}" data-state="{p["state"]}" aria-pressed="{str(i==0).lower()}"><span>{p["id"]} · {p["state"]}</span>{html.escape(p["title"])}</button>' for i,p in enumerate(phases))
page = r'''<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>ECM-ELF · GUI v2 / Vulkan 规划</title>
<style>
:root{color-scheme:light;--bg:#fcfaf7;--surface:#fff;--sunken:#f6f4f1;--fg:#181613;--muted:#605d5a;--border:#e1dfdd;--hover:#f2f0ed;--active:#e9e7e5;--focus:#3772bb;--accent:#181613;--on-accent:#fcfaf7;--info:#2e69b2;--info-bg:#ebf4ff;--warn:#a25f12;--warn-bg:#fef2dd}
[data-theme=dark]{color-scheme:dark;--bg:#252525;--surface:#2c2c2c;--sunken:#202020;--fg:#f3f1f0;--muted:#c4c2c0;--border:#3f3f3f;--hover:#383838;--active:#3f3f3f;--focus:#69a1e8;--accent:#f3f1f0;--on-accent:#252525;--info:#77abec;--info-bg:#19273a;--warn:#e8b45e;--warn-bg:#342611}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.75 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}button{font:inherit;color:inherit;cursor:pointer;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 12px;min-height:34px}button:hover{background:var(--hover)}button[aria-pressed=true]{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}button:focus-visible,a:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:3px}a{color:var(--info);text-underline-offset:3px}code{font:0.9em/1.6 Consolas,"Cascadia Mono",monospace;background:var(--sunken);padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}h1,h2,h3,h4,p{margin:0}h1{font-size:34px;line-height:1.3;font-weight:650;letter-spacing:-.5px}h2{font-size:23px;line-height:1.4;margin-bottom:14px}h3{font-size:17px;line-height:1.5;margin-bottom:10px}h4{font-size:15px;margin:20px 0 8px}p+p{margin-top:12px}small,.caption{font-size:13px;color:var(--muted)}.eyebrow{font-size:12px;letter-spacing:1.8px;color:var(--muted);margin-bottom:14px}.frame{display:grid;grid-template-columns:206px minmax(0,1fr);max-width:1430px;margin:auto;min-height:100vh}.sidebar{border-right:1px solid var(--border);padding:35px 22px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:24px}.brand{font-size:17px;font-weight:650}.brand small{display:block;font-weight:400;font-size:12px;letter-spacing:.7px;margin-top:2px}.sidebar nav{display:flex;flex-direction:column;gap:5px}.sidebar nav a{color:var(--muted);text-decoration:none;border-radius:6px;padding:8px 10px}.sidebar nav a:hover{color:var(--fg);background:var(--hover)}.sidebar-foot{margin-top:auto;font-size:12px;color:var(--muted)}main{min-width:0;padding:44px 46px 80px}.hero{margin-bottom:36px}.hero-top{display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap}.badge{display:inline-block;border:1px solid var(--border);border-radius:5px;padding:2px 8px;font-size:12px;color:var(--muted)}.intro{font-size:17px;max-width:790px;margin-top:18px}.meta{display:flex;gap:10px 18px;flex-wrap:wrap;margin-top:18px;font-size:13px;color:var(--muted)}section{padding:30px 0;border-top:1px solid var(--border)}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:28px}.label{font-size:12px;color:var(--muted);margin-bottom:8px}.identity{padding:16px 0 18px;border-bottom:1px solid var(--border)}.identity strong{display:block;font-size:19px;margin-bottom:7px}.identity p{font-size:14px;color:var(--muted)}.table-wrap{overflow:auto;max-width:100%;margin-top:16px}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}th{color:var(--muted);font-weight:500}th,td{border-bottom:1px solid var(--border);padding:12px 12px 12px 0;vertical-align:top;overflow-wrap:anywhere}th:first-child,td:first-child{min-width:90px;font-weight:500}table td code{font-size:12px}.notice{border-left:3px solid var(--focus);padding:12px 16px;background:var(--info-bg);margin:18px 0}.notice strong{display:block;margin-bottom:4px}.warn{border-left-color:var(--warn);background:var(--warn-bg)}ul,ol{padding-left:22px;margin:12px 0}li+li{margin-top:6px}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:18px 0}.controls .caption{margin-right:auto}.mock{border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--surface)}.mock-title{padding:9px 15px;background:var(--sunken);border-bottom:1px solid var(--border);font-size:13px;color:var(--muted)}.commandbar{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;gap:8px;flex-wrap:wrap;font-size:13px}.commandbar span{padding-right:14px}.board{display:grid;grid-template-columns:150px minmax(0,1fr)205px;min-height:330px}.mock-nav{padding:14px 10px;background:var(--sunken);border-right:1px solid var(--border);display:flex;flex-direction:column;align-items:stretch;gap:7px}.mock-nav button{border-color:transparent;text-align:left;background:transparent}.mock-nav button[aria-pressed=true]{background:var(--active);color:var(--fg)}.mock-nav .settings{margin-top:auto;font-size:12px;color:var(--muted);padding-left:10px}.mock-center{padding:20px;min-width:0;display:flex;flex-direction:column;gap:12px}.mock-center ul{margin:0;padding-left:18px;font-size:14px}.mock-rule{font-size:13px;color:var(--info)}.mock-log{border-top:1px solid var(--border);padding-top:12px;margin-top:auto;font-size:13px}.inspector{padding:20px 16px;border-left:1px solid var(--border);font-size:14px}.inspector .caption{display:block;margin-top:12px}.mock[data-width=narrow] .board{grid-template-columns:76px minmax(0,1fr)}.mock[data-width=narrow] .inspector{display:none}.mock[data-width=narrow] .mock-nav{padding:12px 6px}.mock[data-width=narrow] .mock-nav button{font-size:12px;padding:6px}.mock[data-width=narrow] .mock-nav .full{display:none}.mock-nav .short{display:none}.mock[data-width=narrow] .mock-nav .short{display:inline}.caption-line{margin-top:12px}.tokens{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:18px 0}.token{border-top:4px solid var(--border);padding-top:8px;min-width:0;font-size:13px}.token code{display:block;margin-top:4px;width:fit-content}.phase-grid{display:grid;grid-template-columns:200px minmax(0,1fr);gap:28px;margin-top:20px}.phase-nav{display:flex;flex-direction:column;gap:6px}.phase-button{text-align:left;font-size:13px;padding:8px 12px}.phase-button span{display:inline-block;width:30px;font:12px Consolas,monospace}.phase-detail{border-top:2px solid var(--accent);padding:18px 0}.phase-detail .label{margin-top:22px}.phase-detail .label:first-child{margin-top:0}.phase-detail h3{font-size:20px}.phase-detail p{white-space:pre-line}.check-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.check-group{border-top:1px solid var(--border);padding-top:14px}.full-plan{margin-top:18px}.full-plan>summary{cursor:pointer;font-weight:500;padding:16px 0}.full-body{max-width:900px}.full-body h2{font-size:24px;margin:24px 0}.full-body h3{font-size:21px;border-top:1px solid var(--border);padding-top:25px;margin-top:32px}.full-body h4{font-size:17px;margin-top:22px}.full-body p,.full-body li{font-size:14px;line-height:1.85}.full-body p{margin:12px 0}.footer{margin-top:28px;color:var(--muted);font-size:13px}.source-links{display:flex;gap:10px 20px;flex-wrap:wrap;margin-top:14px}.flow{font:14px/1.9 Consolas,monospace;color:var(--muted);margin:14px 0}.hidden{display:none!important}
@media(max-width:1100px){main{padding:34px 28px 60px}.frame{grid-template-columns:175px minmax(0,1fr)}.sidebar{padding:30px 16px}.board{grid-template-columns:120px minmax(0,1fr)160px}}
@media(max-width:850px){.frame{display:block}.sidebar{position:static;height:auto;border-right:0;border-bottom:1px solid var(--border);padding:18px 22px;gap:14px}.sidebar nav{flex-direction:row;flex-wrap:wrap;font-size:13px;gap:2px}.sidebar nav a{padding:5px 8px}.sidebar-foot{display:none}.brand small{display:inline;margin-left:12px}.board{grid-template-columns:110px minmax(0,1fr)}.inspector{display:none}.phase-grid{grid-template-columns:1fr}.phase-nav{flex-direction:row;flex-wrap:wrap}.phase-button{flex:1 1 145px}.tokens{grid-template-columns:repeat(3,1fr)}}
@media(max-width:580px){main{padding:28px 18px 45px}h1{font-size:28px}.intro{font-size:15px}.twocol,.check-grid{grid-template-columns:1fr;gap:14px}.tokens{grid-template-columns:1fr 1fr}.board,.mock[data-width=narrow] .board{grid-template-columns:65px minmax(0,1fr)}.mock-nav{padding:10px 4px}.mock-nav button{font-size:12px;padding:6px 3px}.mock-nav .full{display:none}.mock-nav .short{display:inline}.mock-nav .settings{padding-left:4px}.mock-center{padding:14px}.mock-title{font-size:12px}.commandbar{padding:8px 12px}.phase-grid{gap:16px}table{font-size:13px}th,td{padding:10px 8px 10px 0}.hero{margin-bottom:25px}}
@media print{.frame{display:block}.sidebar,.controls,#theme-toggle,.phase-nav{display:none}main{padding:0}.full-plan>.full-body{display:block}section{break-inside:avoid}html{scroll-behavior:auto}}
</style>
</head>
<body>
<div class="frame">
<aside class="sidebar"><div class="brand">ECM-ELF<small>GUI v2 · DESIGN PLAN</small></div><nav aria-label="章节"><a href="#boundaries">01 · 双版本边界</a><a href="#current">02 · 现状与设计规则</a><a href="#vulkan">03 · Vulkan 接入</a><a href="#layout">04 · 目标布局</a><a href="#phases">05 · 开发阶段</a><a href="#checks">06 · 验收与交付</a><a href="#full-plan">完整规划与依据</a></nav><div class="sidebar-foot">2026-10-21<br>离线阅读 · 无外部依赖<br>规划状态，尚未实施</div></aside>
<main>
<header class="hero"><div class="eyebrow">C++ / DEAR IMGUI / WIN32 / VULKAN</div><div class="hero-top"><h1>独立构建 GUI v2</h1><button id="theme-toggle" type="button" aria-pressed="false">深色阅读</button></div><p class="intro">保留原 GUI，在独立前端中实现 Ely 风格的布局、配色、字体和控件。先证明 Vulkan 宿主可靠，再逐页接入业务功能。</p><div class="meta"><span class="badge">规划 · 尚未实施</span><span>原版 DX11 保留</span><span>新版 Vulkan 独立构建</span><span>v2 为前端名称，不是公共版本 2.x</span></div></header>

<section id="boundaries"><h2>01 · 两个前端，独立演进</h2><div class="twocol"><div class="identity"><div class="label">原 GUI · 保持不变</div><strong>ecm_gui.exe</strong><p>Win32 + ImGui + Direct3D 11<br>原目录、入口、默认值、布局与测试期望保留。</p></div><div class="identity"><div class="label">GUI v2 · 新增</div><strong>ecm_gui_v2.exe</strong><p>Win32 + ImGui + Vulkan<br>新增工程、构建入口、输出目录和 UI 状态。</p></div></div><div class="table-wrap"><table><thead><tr><th>范围</th><th>原 GUI</th><th>GUI v2 规划</th></tr></thead><tbody>
<tr><td>实现</td><td><code>src/ecm/application/gui/</code></td><td><code>src/ecm/application/gui_v2/</code></td></tr><tr><td>构建入口</td><td><code>tools/build/build_gui.bat</code></td><td>新增 <code>tools/build/build_gui_v2.bat</code></td></tr><tr><td>默认输出</td><td><code>build_gui_local/</code></td><td><code>build_gui_v2_local/</code></td></tr><tr><td>界面状态</td><td>原 <code>[gui]</code> 及停靠布局</td><td>独立 <code>ecm_gui_v2.ui.toml</code>（候选）</td></tr><tr><td>依赖</td><td>不增加 Vulkan 要求</td><td><code>ECM_BUILD_GUI_V2</code> 默认 OFF，显式构建时检查依赖</td></tr><tr><td>业务模块</td><td>原实现和链接关系不变</td><td>明确列源，只读复用 Controller、结果、生成器和概率等模块</td></tr>
</tbody></table></div><p class="caption-line caption">v2 保存外观和窗口状态不改写旧 UI 配置；业务 TOML 与 UI 状态分开。两个 GUI 对同一 worker / 队列组的控制拥有权必须在启动前验证。</p></section>

<section id="current"><h2>02 · 保留成熟能力，重建界面规则</h2><div class="twocol"><div><h3>现有布局的空间约束</h3><p>停靠区左 60% / 右 40%，两边再上下对半。主页面约占停靠区 30%，11 列结果表位于右下小区。</p><p class="caption">以上为源码几何推算。没有把它当作实际可读性或性能测量。</p></div><div><h3>可以继续复用的能力</h3><p>独立进程控制、任务和结果合同、日志可见行裁剪，以及分开的界面刷新与 NVML 采样。</p><p class="caption">旧版参考配置：界面 10 Hz，GPU 采样 500 ms，日志最多 4000 行。</p></div></div><h3 style="margin-top:24px">Ely 风格的核心：中性背景、统一尺寸、清晰状态</h3><div class="tokens"><div class="token">背景<code>#252525</code></div><div class="token">表面<code>#2c2c2c</code></div><div class="token">内嵌区域<code>#202020</code></div><div class="token">正文<code>#f3f1f0</code></div><div class="token">结构边框<code>#3f3f3f</code></div></div><div class="table-wrap"><table><thead><tr><th>规则</th><th>Ely 组件库</th><th>v2 适配</th></tr></thead><tbody><tr><td>语义颜色</td><td>背景 / 表面 / 文字 / 悬停 / 选中 / 焦点 / 状态</td><td>所有控件和手绘图表走统一明暗主题；阶段色与成功/失败色分开</td></tr><tr><td>尺寸</td><td>控件高 24 / 28 / 32；圆角 4 / 6 / 8 / 12</td><td>逻辑尺寸随 DPI 计算，实际高度不能裁切中文</td></tr><tr><td>字体</td><td>Inter / JetBrains Mono，默认正文 13</td><td>先采用自动正文 15px，新增字体角色；保留中文救援</td></tr><tr><td>组件</td><td>Button / PathInput / ListItem / ColumnShell</td><td>封装项目需要的控件；输入、焦点和停靠继续交给 ImGui</td></tr></tbody></table></div></section>

<section id="vulkan"><h2>03 · Vulkan 是独立的宿主工作</h2><p>Win32 平台后端处理窗口、输入、IME 和剪贴板；Vulkan 渲染后端绘制 ImGui。设备、主 surface、swapchain、帧同步、重建和资源回收仍由 v2 宿主管理。</p><div class="notice"><strong>P1 就要注册 Win32 surface 回调</strong>固定 Vulkan 后端要求 <code>Platform_CreateVkSurface</code>，同提交 Win32 后端没有提供。v2 需要自己的 HWND → Vulkan surface 桥接，不能把初始化接线推迟到多窗口阶段。</div><div class="table-wrap"><table><thead><tr><th>接入项</th><th>规划要求</th></tr></thead><tbody><tr><td>固定版本</td><td>新增同提交 <code>64944b4</code> 的 Vulkan 后端到专用目录，不混用最新后端与旧核心</td></tr><tr><td>设备与队列</td><td>按 surface 的 graphics / present 能力选择；渲染 GPU 不绑定 worker 的设备编号</td></tr><tr><td>初始路径</td><td>传统 render pass / framebuffer，先不强制 dynamic rendering 或额外引擎</td></tr><tr><td>生命周期</td><td>处理 resize、最小化零尺寸、OUT_OF_DATE / SUBOPTIMAL、surface / device lost</td></tr><tr><td>同步与纹理</td><td>明确帧资源拥有权和延迟释放；字体 / DPI 切换无悬空 descriptor；不每帧等待整个 device 空闲</td></tr><tr><td>运行环境</td><td>开发阶段 validation；分发机不要求开发 SDK，但必须有可用 loader / 驱动</td></tr></tbody></table></div><div class="notice warn"><strong>多窗口是完整交付门槛</strong>先做单窗口里程碑，再验收浮动窗的创建、跨屏、重建和销毁。固定后端标有新 viewport 闪烁问题，以及图像数量变化的 API 限制，必须以实际验证收口。</div><p class="caption">环境核对仅发现未设置 VULKAN_SDK、无 C:/VulkanSDK；尚未完成其他路径和运行能力检测。没有 Vulkan 时明确提示使用原 GUI，v2 内不暗中回退 DX11。</p></section>

<section id="layout"><h2>04 · 导航稳定，主区获得更多空间</h2><p>左侧导航 + 中央主区 + 右侧检查区 + 底部日志。导航聚焦同一个窗口实例，保留自由停靠和弹出能力。</p><div class="controls"><span class="caption">结构示意 · 不是实际运行截图</span><button id="wide-toggle" type="button" aria-pressed="true">宽屏结构</button><button id="narrow-toggle" type="button" aria-pressed="false">窄屏结构</button></div><div class="mock" id="layout-mock" data-width="wide"><div class="mock-title">ECM-ELF GUI v2 · Windows 系统标题栏保留</div><div class="commandbar"><span>当前配置与全局动作</span><span>通知区域预留空间</span></div><div class="board"><div class="mock-nav" aria-label="示意页面"><button type="button" data-page="0" aria-pressed="true"><span class="full">工作台</span><span class="short">工作</span></button><button type="button" data-page="1" aria-pressed="false"><span class="full">结果</span><span class="short">结果</span></button><button type="button" data-page="2" aria-pressed="false"><span class="full">任务生成</span><span class="short">生成</span></button><button type="button" data-page="3" aria-pressed="false"><span class="full">ECM 概率</span><span class="short">概率</span></button><span class="settings">设置 / 语言</span></div><div class="mock-center"><h3 id="mock-heading">工作台 · 中央主区</h3><div id="mock-id" class="caption">###v2_workers</div><ul id="mock-items"><li>worker 状态、进度、速度与 ETA</li><li>保留表格宽度保护与任务摘要</li><li>低频配置移到配置区域</li></ul><p class="mock-rule" id="mock-rule">继续使用原 Controller / Proc 业务合同。</p><div class="mock-log">底部日志 <code>###v2_logN</code><br><small>初始高度候选 25%；收起时仍消费日志与推进进程监管。</small></div></div><div class="inspector"><h3>GPU / 详情</h3><p>跟随所选 worker 更新上下文，不自动抢焦点。</p><span class="caption">右区候选 300 逻辑单位，可调可收起。</span></div></div></div><p id="layout-caption" class="caption caption-line">宽屏候选：左导航 200 / 中央自适应 / 右区 300 逻辑单位。本图按容器宽度示意，并非像素规格。</p><p class="caption caption-line">v2 首次建立现代布局，重置只影响 v2。首版使用不透明主题与系统标题栏；磨砂和自绘系统按钮后置。</p></section>

<section id="phases"><h2>05 · 七个阶段，独立退出条件</h2><p>首批范围 P0–P2：先冻结原版保护边界，完成 Vulkan 最小宿主，再建立主题、字体和控件样板。</p><div class="flow">P0 → P1 → P2 → P3 → P4 → P5 → P6</div><div class="phase-grid"><div class="phase-nav" aria-label="开发阶段">__PHASE_BUTTONS__</div><article class="phase-detail" aria-live="polite"><div class="label" id="phase-index">P0 · 基线</div><h3 id="phase-title">冻结原版与并存合同</h3><div class="label">工作范围</div><p id="phase-work">__P0_WORK__</p><div class="label">退出条件</div><p id="phase-gate">__P0_GATE__</p></article></div><noscript><p>阶段详情也完整保存在下方“完整规划与依据”中。</p></noscript></section>

<section id="checks"><h2>06 · 外观、业务、后端与原版保护分别验收</h2><div class="check-grid"><div class="check-group"><h3>原 GUI 不变</h3><p>源码与后端差异、原入口、UI 默认值和旧布局；v2 关闭或缺 SDK 时，旧版仍可配置与构建。</p></div><div class="check-group"><h3>Vulkan 生命周期</h3><p>主次窗口、呈现队列、同步、纹理、重建及失败清理；validation 输出和跨屏实际行为。</p></div><div class="check-group"><h3>业务功能等价</h3><p>安全停止、退出、原始 JSON / checksum 复制、生成器预览与 mtime、概率快照及监控采样。</p></div><div class="check-group"><h3>可读性与低占用</h3><p>中英、明暗、四档缩放、长标签和窄屏；同条件至少 5 次采样，性能门限事先冻结。</p></div></div><p class="caption caption-line">CPU、working set / private bytes、显存、绘制时间和输入延迟分别报告。worker 吞吐独立计时；不因使用 Vulkan 就宣称更低占用。</p><div class="source-links"><a href="../development/GUI_MODERNIZATION_DRAFT.md">权威 Markdown 草稿</a><a href="../usage/GUI.md">原 GUI 当前说明</a><a href="../README.md">文档入口</a></div></section>

<section id="full-plan"><h2>完整规划与依据</h2><p class="caption">以下是权威草稿的完整阅读内容。源码依据、固定第三方提交、业务合同和待验证事项均保留。</p><details class="full-plan" id="full-details"><summary>展开完整规划</summary><div class="full-body">__FULL_PLAN__</div></details></section><footer class="footer">本页为离线规划阅读版，内嵌样式与交互，不依赖外部脚本或字体。未构建 GUI v2、未测量运行性能、未执行发布。<br>来源：GUI_MODERNIZATION_DRAFT.md · 2026-10-21</footer>
</main></div>
<script>
"use strict";
const phases = __PHASES_JSON__;
const pages = [
 {name:"工作台",id:"###v2_workers",items:["worker 状态、进度、速度与 ETA","保留表格宽度保护与任务摘要","低频配置移到配置区域"],rule:"继续使用原 Controller / Proc 业务合同。"},
 {name:"结果",id:"###v2_results",items:["来源、筛选、搜索与复制工具栏","11 列结果表使用中央主区域","保留原始 JSON 与来源诊断"],rule:"复制保留原始 JSON / checksum 字节，筛选变化清空选择。"},
 {name:"任务生成",id:"###v2_gen",items:["输入 → 曲线/输出选项 → 预览 → 追加","宽屏可并排输入与预览，窄屏单列","保留目标 worker、分区和 B1 / B2 含义"],rule:"选项变化使旧预览失效，追加前重新核对目标文件。"},
 {name:"ECM 概率",id:"###v2_prob",items:["统一输入表单与结果摘要","图表、图例和坐标适配明暗主题","宽窄窗口按可用空间切换列数"],rule:"不改数值模型、后台输入快照、显式计算和反算口径。"}
];
document.getElementById("theme-toggle").addEventListener("click",function(){const dark=document.documentElement.dataset.theme!=="dark";document.documentElement.dataset.theme=dark?"dark":"light";this.setAttribute("aria-pressed",String(dark));this.textContent=dark?"浅色阅读":"深色阅读";});
function widthMode(narrow){document.getElementById("layout-mock").dataset.width=narrow?"narrow":"wide";document.getElementById("wide-toggle").setAttribute("aria-pressed",String(!narrow));document.getElementById("narrow-toggle").setAttribute("aria-pressed",String(narrow));document.getElementById("layout-caption").textContent=narrow?"窄屏候选：导航折叠为 56 逻辑单位，默认收起右检查区；表单单列，表格保留横向滚动。":"宽屏候选：左导航 200 / 中央自适应 / 右区 300 逻辑单位。本图按容器宽度示意，并非像素规格。";}
document.getElementById("wide-toggle").addEventListener("click",()=>widthMode(false));document.getElementById("narrow-toggle").addEventListener("click",()=>widthMode(true));
document.querySelectorAll("[data-page]").forEach(button=>button.addEventListener("click",()=>{const page=pages[Number(button.dataset.page)];document.querySelectorAll("[data-page]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));document.getElementById("mock-heading").textContent=page.name+" · 中央主区";document.getElementById("mock-id").textContent=page.id;const list=document.getElementById("mock-items");list.replaceChildren(...page.items.map(value=>{const li=document.createElement("li");li.textContent=value;return li;}));document.getElementById("mock-rule").textContent=page.rule;}));
document.querySelectorAll("[data-phase]").forEach(button=>button.addEventListener("click",()=>{const phase=phases[Number(button.dataset.phase)];document.querySelectorAll("[data-phase]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));document.getElementById("phase-index").textContent=phase.id+" · 规划阶段";document.getElementById("phase-title").textContent=phase.title;document.getElementById("phase-work").textContent=phase.work;document.getElementById("phase-gate").textContent=phase.gate;}));
document.querySelector('a[href="#full-plan"]').addEventListener("click",()=>{document.getElementById("full-details").open=true;});
window.addEventListener("beforeprint",()=>{document.getElementById("full-details").open=true;});
</script>
</body></html>
'''
page = page.replace('环境核对仅发现未设置 VULKAN_SDK、无 C:/VulkanSDK；尚未完成其他路径和运行能力检测。', 'P1 已在 RTX 4070 Ti 上通过单窗口 Vulkan / validation 验证。构建使用固定 headers 和 Volk，无开发 SDK 依赖；开发验证层仅解压在证据目录。')
page = page.replace('规划 · 尚未实施', html.escape(status))
page = page.replace('规划状态，尚未实施', html.escape(status))
page = page.replace('P0 · 基线', 'P0 · ' + phases[0]['state'])
page = page.replace('phase.id+" · 规划阶段"', 'phase.id+" · "+phase.state')
page = page.replace('<a href="../usage/GUI.md">原 GUI 当前说明</a>', '<a href="../usage/GUI.md">原 GUI 当前说明</a><a href="../usage/GUI_V2.md">GUI v2 开发预览说明</a>')
page = page.replace('<section id="boundaries">', '<section id="progress"><h2>实施进度</h2>' + markdown(progress) + '</section><section id="boundaries">')
page = page.replace('<nav aria-label="章节">', '<nav aria-label="章节"><a href="#progress">实施进度</a>')
page = page.replace('未构建 GUI v2、未测量运行性能、未执行发布。', '实际完成范围与验证见实施进度；后续阶段不视为已完成，未执行发布。')
page = page.replace('__PHASE_BUTTONS__', phase_buttons)
page = page.replace('__P0_WORK__', html.escape(phases[0]['work']))
page = page.replace('__P0_GATE__', html.escape(phases[0]['gate']))
page = page.replace('__FULL_PLAN__', markdown(text))
page = page.replace('__PHASES_JSON__', json.dumps(phases, ensure_ascii=False).replace('</', '<\\/'))
target.write_text(page, encoding='utf-8', newline='\n')
print(f'Created {target} ({len(page.encode("utf-8"))} bytes), {len(phases)} phases, full source text included.')
