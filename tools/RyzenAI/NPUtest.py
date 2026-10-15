#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD Ryzen AI NPU Environment Detection Script (Windows)
Auto-detect NPU hardware model, driver version, SDK environment, etc.
"""

import subprocess
import sys
import os
import re


# === Color Output ===
class C:
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    B = '\033[96m'
    BOLD = '\033[1m'
    E = '\033[0m'


def ok(m):   print(f"  {C.G}[OK]{C.E} {m}")
def warn(m): print(f"  {C.Y}[!!]{C.E} {m}")
def fail(m): print(f"  {C.R}[XX]{C.E} {m}")
def info(m): print(f"  {C.B}[ii]{C.E} {m}")


def header(m):
    sep = '=' * 58
    print(f"\n{C.BOLD}{sep}{C.E}")
    print(f"{C.BOLD}  {m}{C.E}")
    print(f"{C.BOLD}{sep}{C.E}")


def run_cmd(cmd):
    """Run system command, auto-handle Windows Chinese encoding"""
    try:
        r = subprocess.run(cmd, capture_output=True, shell=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if not r.stdout:
            return ""
        try:
            return r.stdout.decode('utf-8')
        except UnicodeDecodeError:
            return r.stdout.decode('gbk', errors='ignore')
    except Exception as e:
        return f"ERROR: {e}"


# === NPU Device ID Mapping ===
NPU_MAP = {
    '1502': ('Ryzen AI PHX/HPT (NPU ~10 TOPS)', 'Phoenix/Hawk Point', 'XDNA 1'),
    '17F0': ('Ryzen AI STX (NPU ~40 TOPS)',     'Strix Point',        'XDNA 2'),
    '17F1': ('Ryzen AI STX (NPU ~40 TOPS)',     'Strix Point',        'XDNA 2'),
    '17F2': ('Ryzen AI STX (NPU ~40 TOPS)',     'Strix Point',        'XDNA 2'),
    '17F3': ('Ryzen AI STX (NPU ~40 TOPS)',     'Strix Point',        'XDNA 2'),
}


# ============================================================
#  1. NPU Hardware Detection
# ============================================================
def detect_npu_hardware():
    header("1. NPU Hardware Detection")
    found = False
    info_dict = {}

    # Method 1: pnputil search ProcessingAccelerators class
    out = run_cmd('pnputil /enum-devices /class ProcessingAccelerators')
    matches = re.findall(r'PCI\\VEN_1022&DEV_([0-9A-Fa-f]{4})', out)

    if matches:
        for did in set(matches):
            did_u = did.upper()
            if did_u in NPU_MAP:
                name, arch, gen = NPU_MAP[did_u]
                ok(f"NPU Device: {name}")
                info(f"Arch: {arch} | Gen: {gen} | Device ID: {did_u}")
                found = True
                info_dict = {'device_id': did_u, 'architecture': arch,
                             'generation': gen, 'name': name}
            else:
                warn(f"Unknown AMD device: VEN_1022&DEV_{did_u}")
                info_dict['device_id'] = did_u
    else:
        # Method 2: PowerShell WMI query
        ps_cmd = (
            "Get-CimInstance Win32_PnPEntity | Where-Object { "
            "$_.DeviceID -like '*VEN_1022*DEV_1502*' -or "
            "$_.DeviceID -like '*VEN_1022*DEV_17F*' -or "
            # "$_.Name -like '*AMD*AI*' -or "
            "$_.Name -like '*AMD*NPU*' "
            "} | Select-Object Name, DeviceID, Status, DriverVersion | Format-List"
        )
        ps_out = run_cmd(f'powershell -Command "{ps_cmd}"')
        if ps_out.strip() and any(k in ps_out for k in ['NPU', 'AI', '1502', '17F']):
            ok("NPU device found via WMI:")
            for ln in ps_out.strip().split('\n'):
                if ln.strip():
                    print(f"    {ln.strip()}")
            found = True
        else:
            # Method 3: list all AMD PCI devices
            all_amd = run_cmd('pnputil /enum-devices /instanceid "PCI\\VEN_1022*"')
            devs = re.findall(r'VEN_1022&DEV_([0-9A-Fa-f]{4})', all_amd)
            if devs:
                warn("No known NPU found. AMD PCI devices detected:")
                for d in sorted(set(devs)):
                    u = d.upper()
                    label = NPU_MAP[u][0] if u in NPU_MAP else 'Non-NPU device'
                    print(f"    VEN_1022&DEV_{u} - {label}")

    if not found:
        fail("AMD Ryzen AI NPU hardware NOT detected")
        info("Possible reasons:")
        print("    1. Device does not have a Ryzen AI processor")
        print("    2. NPU driver not installed (check Device Manager)")
        print("    3. NPU disabled in BIOS")
    return found, info_dict


# ============================================================
#  2. NPU Driver Status
# ============================================================
# 修改后的 detect_driver 函数
def detect_driver(npu_info):
    header("2. NPU Driver Status")
    
    # 方法1: 检查设备管理器（最可靠）
    ps_cmd1 = (
        'Get-PnpDevice | Where-Object { '
        "$_.FriendlyName -like '*AMD NPU*' -or "
        "$_.FriendlyName -like '*MCDM*' -or "
        "$_.InstanceId -like '*VEN_1022&DEV_1502*' -or "
        "$_.InstanceId -like '*VEN_1022&DEV_17F*' "
        '} | Select-Object Status, Class, FriendlyName, InstanceId'
    )
    dev_output = run_cmd(f'powershell -Command "{ps_cmd1}"')
    
    if dev_output.strip() and 'OK' in dev_output:
        ok("NPU设备在设备管理器中正常")
    else:
        fail("NPU设备未在设备管理器中找到或状态异常")
    
    # 方法2: 使用pnputil查询（更底层）
    pnputil_cmd = 'pnputil /enum-devices /instanceid "PCI\\VEN_1022&DEV_1502*"'
    pnputil_output = run_cmd(pnputil_cmd)
    if pnputil_output.strip():
        ok("通过pnputil找到NPU设备")
    
    # 方法3: 检查WMI性能计数器
    wmi_cmd = 'Get-CimInstance -ClassName Win32_PerfFormattedData_AMDNPUMCDM_AMDNPUMCDM -ErrorAction SilentlyContinue'
    wmi_output = run_cmd(f'powershell -Command "{wmi_cmd}"')
    if wmi_output.strip() and 'UsagePercent' in wmi_output:
        ok("WMI性能计数器已就绪")
    
    # 方法4: 检查驱动文件
    driver_paths = [
        r"C:\Windows\System32\drivers\amdxdna.sys",
        r"C:\Windows\System32\drivers\amdumd.sys",
    ]
    print()
    info("关键驱动文件检查:")
    for path in driver_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ok(f"    {path} ({size_mb:.1f} MB)")
        else:
            warn(f"    {path} 不存在")
    
    # 综合判断
    driver_ok = (
        ('OK' in dev_output) or 
        (pnputil_output.strip()) or 
        ('UsagePercent' in wmi_output)
    )
    
    if driver_ok:
        print()
        ok("NPU驱动检测成功（综合判断）")
    else:
        print()
        fail("NPU驱动检测失败")
    
    return driver_ok


# ============================================================
#  3. Ryzen AI SDK Environment
# ============================================================
def detect_sdk():
    header("3. Ryzen AI SDK Environment")
    pv = run_cmd('python --version')
    if pv.strip():
        ok(f"Python: {pv.strip()}")
    else:
        fail("Python not found in PATH")
        return

    pp = run_cmd('where python').strip()
    if pp:
        info(f"Python path: {pp.split(chr(10))[0].strip()}")

    pkgs = [
        ('onnxruntime',         'ONNX Runtime'),
        ('onnxruntime_vitisai', 'ONNX RT Vitis AI EP'),
        ('vaip',                'VAIP (Vitis AI Provider)'),
        ('voe',                 'VOE (Vitis ONNX Engine)'),
        ('torch',               'PyTorch'),
        ('onnx',                'ONNX'),
        ('numpy',               'NumPy'),
    ]
    print()
    info("Python packages:")
    for pkg, name in pkgs:
        try:
            r = subprocess.run(
                [sys.executable, '-c', f'import {pkg}; print({pkg}.__version__)'],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0:
                ok(f"    {name}: {r.stdout.strip()}")
            else:
                fail(f"    {name}: not installed")
        except Exception:
            fail(f"    {name}: check failed")

    print()
    info("ONNX Runtime Execution Providers:")
    try:
        r = subprocess.run(
            [sys.executable, '-c',
             'import onnxruntime as ort; print("\\n".join(ort.get_available_providers()))'],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode == 0:
            for p in r.stdout.strip().split('\n'):
                p = p.strip()
                if not p:
                    continue
                if 'VitisAI' in p:
                    ok(f"    {p} <-- NPU acceleration!")
                elif 'Dml' in p or 'CUDA' in p:
                    ok(f"    {p} <-- GPU acceleration")
                else:
                    info(f"    {p}")
        else:
            fail("    Cannot get EP list")
    except Exception:
        fail("    ONNX Runtime not installed")


# ============================================================
#  4. System Information
# ============================================================
def detect_system_info():
    header("4. System Information")
    cpu = run_cmd(
        'powershell -Command "Get-CimInstance Win32_Processor '
        '| Select-Object -ExpandProperty Name"'
    ).strip()
    if cpu:
        ok(f"CPU: {cpu}")

    os_out = run_cmd(
        'powershell -Command "Get-CimInstance Win32_OperatingSystem '
        '| Select-Object Caption, Version, BuildNumber, OSArchitecture | Format-List"'
    )
    for ln in os_out.strip().split('\n'):
        if ln.strip():
            print(f"    {ln.strip()}")

    mem = run_cmd(
        'powershell -Command "'
        '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB"'
    ).strip()
    if mem:
        try:
            info(f"Memory: {float(mem):.1f} GB")
        except ValueError:
            pass

    if cpu:
        markers = ['ryzen ai', '8945', '8845', '8840',
                   '8645', '8640', '7840', '7640']
        for m in markers:
            if m in cpu.lower():
                ok("Identified as Ryzen AI series processor")
                break


# ============================================================
#  5. NPU Environment Variables
# ============================================================
def detect_env_vars():
    header("5. NPU Environment Variables")
    targets = [
        'RYZEN_AI_INSTALLATION_PATH', 'XLNX_VART_FIRMWARE',
        'VAIP_DRIVERS_PATH', 'VART_FIRMWARE', 'XRT_PATH',
    ]
    found = False
    for v in targets:
        val = os.environ.get(v)
        if val:
            ok(f"    {v} = {val}")
            found = True
    for k, val in os.environ.items():
        if any(x in k.upper() for x in ['RYZEN', 'XDNA', 'XAIE']):
            if k not in targets:
                ok(f"    {k} = {val}")
                found = True
    if not found:
        warn("No NPU-related environment variables detected")


# ============================================================
#  6. Detection Summary
# ============================================================
def summary(npu_found, npu_info, driver_ok):
    header("6. Detection Summary")
    if npu_found:
        ok(f"NPU Hardware: {npu_info.get('name', 'Detected')}")
        if npu_info.get('device_id'):
            info(f"Device ID: {npu_info['device_id']} "
                 f"({npu_info.get('generation', '?')})")
    else:
        fail("NPU Hardware: NOT detected")

    st = "Installed" if driver_ok else "Not installed / Not detected"
    ic = "[OK]" if driver_ok else "[XX]"
    print(f"  {ic} MCDM Driver: {st}")
    print()

    if npu_found and driver_ok:
        print(f"  {C.G}{C.BOLD}"
              f"Status: NPU environment READY for development!"
              f"{C.E}")
        info("Next: Run Ryzen AI SDK examples to verify NPU inference")
    elif npu_found:
        print(f"  {C.Y}{C.BOLD}"
              f"Status: NPU hardware detected, but driver not ready"
              f"{C.E}")
        info("Next: Install NPU driver (MCDM) from "
             "https://ryzenai.docs.amd.com")
    else:
        print(f"  {C.R}{C.BOLD}"
              f"Status: NPU environment NOT ready"
              f"{C.E}")
        info("Next: Confirm device has a Ryzen AI processor")


# ============================================================
#  Main
# ============================================================
def main():
    os.system('')  # Enable ANSI colors on Windows Terminal
    print()
    print(f"  {C.BOLD}{C.B}+----------------------------------------------------+")
    print(f"  |   AMD Ryzen AI NPU Detection Tool v1.0             |")
    print(f"  |   Windows Native                                   |")
    print(f"  +----------------------------------------------------+{C.E}")
    print()

    detect_system_info()
    npu_found, npu_info = detect_npu_hardware()
    driver_ok = detect_driver(npu_info)
    detect_sdk()
    detect_env_vars()
    summary(npu_found, npu_info, driver_ok)

    print()


if __name__ == '__main__':
    main()
