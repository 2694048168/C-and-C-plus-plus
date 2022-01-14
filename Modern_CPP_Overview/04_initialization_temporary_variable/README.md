# Compile Command for 04_initialization_temporary_variable

## Command line with g++ or clang++

```shell
g++ initialization_temporary_variable.cpp -std=c++2a -o initialization_temporary_variable

clang++ initialization_temporary_variable.cpp -std=c++2a -o initialization_temporary_variable

# VSCode Debug
# step 1. 选择需要进行调试的源代码文件
# step 2. [Run] --> [Start Debugging] 或者直接快捷键 F5 开启调试
# step 3. 然后选择调试的环境 [C++(GDB/LLDB)]
# step 4. 然后接着选择配置文件的生成 [g++/clang++]
# step 5. 主要自动生成的配置文件 (launch.json and tasks.json)
# step 6. launch.json 配置文件主要配置调试文件的，调试器 gdb, 需要进行调试的可执行文件路径
# step 7. tasks.json 配置文件主要配置编译器情况，需要编译选项 "-g" 才能进行调试
# step 8. 如果使用 CMake 进行构建编译, 那么 launch.json 文件里面指定调试可执行文件部分修改为 CMake Tools 插件提供的命令
# "preLaunchTask": "C/C++: clang++.exe 生成活动文件" 这一行需要编译的 tasks.json 就不需要了，直接由 CMake 完成
# "program": "${fileDirname}\\${fileBasenameNoExtension}.exe"
# "program": "${command:cmake.launchTargetPath}" 
```
