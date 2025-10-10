# ...existing code...
#!/usr/bin/env bash
set -euo pipefail

# download sherpa onnx models
mkdir -p sherpa-onnx
# x64
mkdir -p sherpa-onnx/x64
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.14/sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2
tar -xvf sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2 -C sherpa-onnx/x64
rm sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2
# aarch64
mkdir -p sherpa-onnx/aarch64
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.14/sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2
tar -xvf sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2 -C sherpa-onnx/aarch64
rm sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2

# macos aarch64
mkdir -p sherpa-onnx/macos-aarch64
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.14/sherpa-onnx-v1.12.14-osx-universal2-shared.tar.bz2
tar -xvf sherpa-onnx-v1.12.14-osx-universal2-shared.tar.bz2 -C sherpa-onnx/macos-aarch64
rm sherpa-onnx-v1.12.14-osx-universal2-shared.tar.bz2