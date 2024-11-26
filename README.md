# Yuka

YRCC instance segmentations based on YOLOv8

# Note
1. First do `rye add torch torchvision`, directly `rye sync` will fail, seems to be a bug for uv
2. Use `uv pip install .\assets\rs_utils-0.1.4-cp312-none-win_amd64.whl`
3. Use `uv pip install setuptools`
3. Use `uv pip install git+https://github.com/MrParosk/soft_nms.git --no-build-isolation`