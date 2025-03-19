import os
import subprocess
import sys

def check_architecture(file_path):
    """Kiểm tra kiến trúc của file .pyd hoặc .dll"""
    try:
        output = subprocess.check_output(["dumpbin", "/headers", file_path], stderr=subprocess.DEVNULL)
        output = output.decode(errors="ignore")
        if "machine (x86)" in output.lower():
            return "32-bit ❌"
        elif "machine (x64)" in output.lower():
            return "64-bit ✅"
        else:
            return "Unknown"
    except:
        return "Error"

def scan_libraries():
    """Quét toàn bộ thư viện trong site-packages"""
    lib_path = os.path.join(sys.prefix, "Lib", "site-packages")
    print(f"📂 Kiểm tra thư viện trong: {lib_path}\n")

    for root, dirs, files in os.walk(lib_path):
        for file in files:
            if file.endswith(".pyd") or file.endswith(".dll"):
                file_path = os.path.join(root, file)
                arch = check_architecture(file_path)
                print(f"{file_path}: {arch}")

scan_libraries()
