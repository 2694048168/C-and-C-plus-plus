# Compile Command for 07_auto_decltype

## Command line with g++ or clang++

```shell
g++ auto_decltype.cpp -std=c++2a -Wall -o auto_decltype

clang++ auto_decltype.cpp -std=c++2a -Wall -o auto_decltype
```

```shell
# GCC/g++/clang++ 常用编译选项
# -S 生成汇编代码
g++ -S filename.cpp -o filename.s

# -c 生成可目标文件，但不进行链接
g++ -c filename.cpp -o filename.o

# -o 指定生成文件的文件名
# -std=c++2a 指定 C++ 所使用的标准
# -g 在目标文件中添加调试信息，便于 gdb 调试或 objdump 反汇编
g++ -g filename.cpp -o filename

# -Wall 显示所有的警告信息(建议使用)
# -Werror 视警告为错误，出现警告即放弃编译
# -w 不显示任何警告信息(不建议使用)
# -v 显示编译步骤
# -On (n=0,1,2,3) 设置编译器优化等级，O0为不优化，O3为最高等级优化，O1为默认优化等级
# -L 指定库文件的搜索目录
# -l (小写的L)链接某一库
# -I (大写的i)指定头文件路径
# -D 定义宏，例如 -DAAA=1, -DBBBB -DDebug
# -U 取消宏定义，例如 -UAAA -UDebug
```