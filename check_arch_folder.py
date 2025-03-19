import os
import subprocess

# 🛠 HÀM KIỂM TRA FILE CÓ 32-BIT HAY 64-BIT
def check_architecture(file_path):
    try:
        output = subprocess.check_output(["dumpbin", "/headers", file_path], stderr=subprocess.DEVNULL)
        output = output.decode(errors="ignore").lower()
        
        if "machine (x86)" in output:
            return "32-bit ❌"
        elif "machine (x64)" in output:
            return "64-bit ✅"
        else:
            return "Unknown"
    except Exception as e:
        return f"Error ({e})"

# 🔎 KIỂM TRA TẤT CẢ FILE TRONG FOLDER
def scan_folder(folder_path):
    print(f"📂 Đang kiểm tra folder: {folder_path}\n")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".exe", ".dll", ".pyd", ".so")):  # Lọc file cần kiểm tra
                file_path = os.path.join(root, file)
                arch = check_architecture(file_path)
                print(f"{file_path}: {arch}")

# 🔥 CHẠY SCRIPT Ở FOLDER CẦN KIỂM TRA
folder_to_scan = "F:\\viXTTS"  # 🔄 Đổi thành thư mục bạn muốn kiểm tra
scan_folder(folder_to_scan)
