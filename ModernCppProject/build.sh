cmake --preset windows-debug          # 配置 Debug 构建目录
cmake --build --preset windows-debug  # 构建 Debug 版本
cmake --build --preset windows-debug --target install
# cmake --build --preset windows-debug --target docs