import os
from huggingface_hub import snapshot_download

# Tải Unidic
os.system("python -m unidic download")

# Tải mô hình từ Hugging Face
print(" > Tải mô hình...")
snapshot_download(
    repo_id="thinhlpg/viXTTS",
    repo_type="model",
    local_dir="model"
)

# Xóa output trên màn hình (nếu chạy trong Jupyter Notebook)
try:
    from IPython.display import clear_output
    clear_output()
except ImportError:
    pass  # Nếu không chạy trên Jupyter thì bỏ qua

print(" > ✅ Cài đặt hoàn tất, bạn hãy chạy tiếp các bước tiếp theo nhé!")
