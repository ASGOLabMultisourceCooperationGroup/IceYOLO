# Yuka

YRCC instance segmentations based on YOLOv8

# Note

1. First do `rye add torch torchvision`, directly `rye sync` will fail, seems to be a bug for uv
2. Cross compile to linux with command `maturin build --release --target x86_64-unknown-linux-gnu --zig -i 3.12`