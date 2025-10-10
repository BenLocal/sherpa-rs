# ...existing code...
#!/usr/bin/env bash
set -euo pipefail

# download sherpa onnx models
mkdir -p sherpa-onnx
# x64
mkdir -p sherpa-onnx/x64
wget http://1.1.1.1/mirror/sherpa-onnx/sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2
tar -xvf sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2 -C sherpa-onnx/x64
rm sherpa-onnx-v1.12.14-linux-x64-shared.tar.bz2
# aarch64
mkdir -p sherpa-onnx/aarch64
wget http://1.1.1.1/mirror/sherpa-onnx/sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2
tar -xvf sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2 -C sherpa-onnx/aarch64
rm sherpa-onnx-v1.12.14-linux-aarch64-shared-cpu.tar.bz2


