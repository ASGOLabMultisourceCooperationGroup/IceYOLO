# IceYOLO

IceYOLO implementation of the paper [River Ice Fine-Grained Segmentation: A GF-2 Satellite Image Dataset and Deep Learning Benchmark](https://doi.org/10.1109/TGRS.2025.3604644)

# Note

1. First do `rye add torch torchvision`, directly `rye sync` will fail, seems to be a bug for uv
2. Cross compile to linux with command `maturin build --release --target x86_64-unknown-linux-gnu --zig -i 3.12`
