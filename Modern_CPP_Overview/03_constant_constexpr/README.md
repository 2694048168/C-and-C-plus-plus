# Compile Command for 03_constant_constexpr

## Command line with g++ or clang++

```shell
g++ constant_constexpr.cpp -std=c++2a -o constant_constexpr
clang++ constant_constexpr.cpp -std=c++2a -o constant_constexpr
```

```shell
# objdump 反汇编常用参数
# objdump --help
# objdump -d <file(s)>: 将代码段反汇编
# objdump -S <file(s)>: 将代码段反汇编的同时，将反汇编代码与源代码交替显示，编译时需要使用 -g 参数，即需要调试信息
# objdump -C <file(s)>: 将 C++ 符号名逆向解析
# objdump -l <file(s)>: 反汇编代码中插入文件名和行号
# objdump -j section <file(s)>: 仅反汇编指定的 section

# 从源代码编译为汇编文件
g++ -S constant_constexpr.cpp -o constant_constexpr.s -std=c++2a
clang++ -S constant_constexpr.cpp -o constant_constexpr.s -std=c++2a

# 从编译后的二进制文件反汇编文件
# objdump command

# step 1. 将源代码文件编译为目标文件
g++ -c constant_constexpr.cpp -o constant_constexpr.o -std=c++2a
clang++ -c constant_constexpr.cpp -o constant_constexpr.o -std=c++2a
# 将目标文件进行反汇编
objdump -C -s -d constant_constexpr.o

# step 2. 将目标文件(需要编译时有调试信息 -g)进行反汇编, 同时显示源码
g++ -g -c constant_constexpr.cpp -o constant_constexpr.o -std=c++2a
objdump -S -d constant_constexpr.o
# 显示源码的时候显示行号
objdump -j .text -ld -C -S constant_constexpr.o

# step 3. 将可执行的二进制文件进行反汇编
g++ constant_constexpr.cpp -o constant_constexpr -std=c++2a
clang++ constant_constexpr.cpp -o constant_constexpr -std=c++2a
objdump -s -d constant_constexpr > binary_asm_code.txt 
objdump -s -d constant_constexpr.exe > binary_asm_code.txt
```