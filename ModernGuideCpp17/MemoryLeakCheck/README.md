## Memory Leak Check for Modern C and C++

> Memory leakage is the magic and charm of C and C++

![image-pipeline]()

### Memory Leak Check
- Valgrind 工具进行内存泄漏检测 **ValgrindTool.cpp**
- log-message 方式进行内存泄漏检测 **MemoryLeak.cpp**
- bpf and bpftrace 方式进行内存泄漏检测

```shell
# sudo apt install bpftrace
# sudo nala install bpftrace
bpftrace memory.bt

# 然后启动线上进程
./memory

# memory.bt 检测的进程名称为 'memory'
BEGIN
{
    @count[comm] = 0;
}

uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc /comm == "memory"/
{
    @count[comm] ++;
    printf("memory ---> malloc\n);
}

uprobe:/lib/x86_64-linux-gnu/libc.so.6:free /comm == "memory"/
{
    @count[comm] --;
    printf("memory ---> free\n);
}

```
