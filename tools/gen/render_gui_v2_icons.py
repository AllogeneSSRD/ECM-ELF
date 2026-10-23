"""Export the actual C++ line_icon commands into a self-contained offline gallery.

Requires Windows + MSVC (or --vcvars to select its environment). No ImGui runtime
or renderer is started. The recorder translates drawing primitives to SVG;
browser rasterization is illustrative, not a native pixel comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/ecm/application/gui_v2/ui/icons.h'
TARGET = ROOT / 'docs/web/gui-v2-icons.html'
TEMPLATE = Path(__file__).with_name('templates') / 'gui_v2_icons.html'

# Labels and current uses only. All geometry comes from icons.h, not this map.
CATALOG = {
    'None': ('无图标', ['其他'], '不绘制图案；用于纯文字按钮。'),
    'Play': ('开始', ['操作'], '圆角空心三角形；全部开始、单个 worker 开始。'),
    'Stop': ('停止', ['操作'], '全部安全停止、单个 worker 停止与停止中。'),
    'Grid': ('网格', ['导航', '布局'], '组件页、切换到双列耗时图。'),
    'List': ('列表', ['布局'], '切换到每行一个的耗时图。'),
    'Sliders': ('滑杆', ['预留'], '当前主界面未绑定，保留用于筛选或参数入口。'),
    'Check': ('确认', ['预留'], '当前主界面未绑定，保留用于完成或确认状态。'),
    'Folder': ('文件夹', ['设置'], '设置分类：工作流与引擎。'),
    'Cpu': ('计算', ['导航', '设置'], '计算任务导航；设置分类：计算设置。'),
    'Results': ('结果', ['导航'], '结果页面导航。'),
    'Generate': ('生成', ['导航'], '队列生成页面导航。'),
    'Chart': ('图表', ['导航', '设置'], 'ECM 概率导航；设置分类：监控。'),
    'Gpu': ('GPU 监控', ['导航'], 'GPU 页面导航；图案为显示器与曲线。'),
    'Settings': ('设置', ['导航', '设置'], '导航底部设置入口及设置弹窗标题。'),
    'Palette': ('外观', ['设置'], '设置分类：外观。'),
    'Pet': ('宠物', ['设置'], '设置分类：ELY 宠物。'),
    'Calculate': ('运算', ['导航', '操作'], '两行加、减、乘、等号，无外框；群阶分解导航和计算群阶与点阶按钮。'),
    'Layout': ('布局', ['设置'], '设置分类：布局与交互。'),
}

RECORDER = r'''#pragma once
#include <cmath>
#include <sstream>
#include <stdexcept>
using ImU32 = unsigned int;
enum ImDrawFlags_ { ImDrawFlags_Closed = 1 << 9 };
struct ImVec2 { float x,y; };
struct ImDrawList {
    std::ostringstream out;
    std::ostringstream path;
    void width(float w) { out << " stroke-width=\"" << w << "\""; }
    void pair(ImVec2 p) { out << p.x << ',' << p.y; }
    void AddLine(ImVec2 a,ImVec2 b,ImU32,float w) {
        out << "<line x1=\"" << a.x << "\" y1=\"" << a.y << "\" x2=\"" << b.x << "\" y2=\"" << b.y << "\""; width(w);out << "/>";
    }
    void AddRect(ImVec2 a,ImVec2 b,ImU32,float r,int flags,float w) {
        if(flags) throw std::runtime_error("Recorder does not support rectangle flags");
        out << "<rect x=\"" << a.x << "\" y=\"" << a.y << "\" width=\"" << b.x-a.x << "\" height=\"" << b.y-a.y << "\" rx=\"" << r << "\"";width(w);out << "/>";
    }
    void AddCircle(ImVec2 a,float r,ImU32,int segments,float w) {
        if(segments<3) throw std::runtime_error("Recorder requires explicit circle segments");
        out << "<polygon points=\"";
        for(int i=0;i<segments;++i) {if(i)out << ' ';const float t=float(i)*6.28318530718f/segments;pair({a.x+r*std::cos(t),a.y+r*std::sin(t)});}
        out << "\"";width(w);out << "/>";
    }
    void AddTriangle(ImVec2 a,ImVec2 b,ImVec2 c,ImU32,float w) {
        out << "<polygon points=\"";pair(a);out << ' ';pair(b);out << ' ';pair(c);out << "\"";width(w);out << "/>";
    }
    void AddBezierCubic(ImVec2 a,ImVec2 b,ImVec2 c,ImVec2 d,ImU32,float w) {
        out << "<path d=\"M";pair(a);out << " C";pair(b);out << ' ';pair(c);out << ' ';pair(d);out << "\"";width(w);out << "/>";
    }
    void PathLineTo(ImVec2 p) {
        path << (path.str().empty()?"M":" L") << p.x << ',' << p.y;
    }
    void PathBezierCubicCurveTo(ImVec2 b,ImVec2 c,ImVec2 d) {
        if(path.str().empty()) throw std::runtime_error("Bezier path needs a start point");
        path << " C" << b.x << ',' << b.y << ' ' << c.x << ',' << c.y << ' ' << d.x << ',' << d.y;
    }
    void PathStroke(ImU32,float w,int flags=0) {
        if(flags & ~ImDrawFlags_Closed) throw std::runtime_error("Unsupported path stroke flags");
        out << "<path d=\"" << path.str() << ((flags & ImDrawFlags_Closed)?" Z":"") << "\"";
        width(w);out << "/>";path.str("");path.clear();
    }
};
'''


def generate(vcvars):
    source = SOURCE.read_text(encoding='utf-8')
    enum = re.search(r'enum class Icon\s*\{([^}]+)\}', source)
    if not enum:
        raise ValueError('Icon enum not found')
    names = [name.strip() for name in enum[1].split(',') if name.strip()]
    if len(names) != len(set(names)) or set(names) != set(CATALOG):
        raise ValueError('Update gallery labels to cover exactly the current Icon enum')
    scratch = ROOT / 'data/experiments/gui_v2/icon_gallery_build'
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / 'imgui.h').write_text(RECORDER, encoding='utf-8')
    calls = '\n'.join(f'emit("{name}", Icon::{name});' for name in names)
    unit = ('#include <iostream>\n#include "' + SOURCE.as_posix() + '"\n'
            'using namespace ecmgui_v2::ui;\n'
            'void emit(const char* name,Icon icon){ImDrawList draw;'
            'line_icon(&draw,icon,{0,0},24,0);'
            'std::cout<<name<<"\\t"<<draw.out.str()<<"\\n";}\n'
            'int main(){\n' + calls + '\n}\n')
    (scratch / 'export.cpp').write_text(unit, encoding='utf-8')
    command = (f'@echo off\ncall "{vcvars}" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
               f'cl /nologo /std:c++17 /utf-8 /EHsc /I"{scratch}" '
               f'"{scratch / "export.cpp"}" /Fe:"{scratch / "export.exe"}" '
               f'/Fo:"{scratch / "export.obj"}" >"{scratch / "compile.log"}" 2>&1\n'
               'exit /b %errorlevel%\n')
    script = scratch / 'compile.cmd'
    script.write_text(command, encoding='utf-8')
    subprocess.run(['cmd.exe', '/d', '/c', str(script)], cwd=scratch, check=True)
    exported = subprocess.check_output([str(scratch / 'export.exe')], text=True)
    geometry = dict(line.split('\t', 1) for line in exported.splitlines())
    if list(geometry) != names or geometry['None'] or any(not geometry[n] for n in names if n != 'None'):
        raise ValueError('Missing or unexpected drawing output')
    icons = [dict(name=n, label=CATALOG[n][0], groups=CATALOG[n][1],
                  usage=CATALOG[n][2], svg=geometry[n]) for n in names]
    page = TEMPLATE.read_text(encoding='utf-8')
    page = page.replace('__ICONS__', json.dumps(icons, ensure_ascii=False).replace('<', '\\u003c'))
    page = page.replace('__SOURCE_SHA__', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    page = page.replace('__COUNT__', str(len(names)-1))
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify committed HTML against current C++ source and template')
    parser.add_argument('--vcvars', type=Path, default=Path('C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat'))
    args = parser.parse_args()
    result = generate(args.vcvars)
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding='utf-8') != result:
            raise SystemExit('Icon gallery is stale; regenerate it')
        print('Icon gallery matches current C++ source, metadata and template')
    else:
        TARGET.write_text(result, encoding='utf-8', newline='\n')
        print(f'Created {TARGET}')
