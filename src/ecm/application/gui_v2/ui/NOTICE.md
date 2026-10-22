# GUI v2 design attribution

The semantic palette and size conventions in `theme.cpp` adapt Ely GPUI
Components, commit `17f936b98c8c8fbdd5454745ac70c22df8526689`, inspected in
`.refactor/Ely-GPUI-Components/src/theme/`. No GPUI runtime, font files or
Lucide assets are bundled. The small line icons are drawn by this application.
Windows system fonts are discovered at runtime and are not redistributed.

The non-docking settings window, category sidebar and theme preview cards also
reference `.refactor/ELYTerminal/src/settings/window.rs` and
`.refactor/Ely-GPUI-Components/src/settings/section.rs`. These are native ImGui
controls using this application's palette, line geometry and external XML text;
no upstream icon artwork or GPUI window/runtime code is copied into the build.

Upstream license:

MIT License

Copyright (c) 2026 Ely GPUI Component contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
