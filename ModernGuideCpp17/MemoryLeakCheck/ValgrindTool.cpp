/**
 * @file ValgrindTool.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 利用 Valgrind 工具进行内存泄漏检测
 * @version 0.1
 * @date 2023-12-03
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>


/**
 * @brief [Valgrind](https://valgrind.org/)
 * 0. update: sudo nala/apt update
 * 1. install: sudo nala/apt install valgrind
 * 2. check version: valgrind --version
 * 3. command help: valgrind --help
 * 
 * 4. compile and link with Debug info: 
 *    g++ ValgrindTool.cpp -g -O0 -o main
 * 5. using Valgrind tool to check memory:
 *    valgrind ./main
 * 6. Check the output Info from Valgrind
 * 
 */


// 最简单的内存泄漏: 没有释放申请的 heap 内存
void func()
{ 
    char *ptr = new char; 
}

// 内存访问越界, 错误释放方式, 
// 直接使用未经处理的内存空间, 使用被释放的内存空间
// 多次释放指针(double delete)
void func1()
{
    int *p = new int[10];
    p[10] = 42; /* 内存访问越界 */
    delete p; /* 错误释放方式 */

    int* p2;
    int num = *p2; /* 直接使用未经处理的内存空间 */

    int *p3 = new int;
    delete p3;
    *p3 = 24;
    delete p3;
}

// ====================================
int main(int argc, const char **argv) 
{ 
    func();
    func1();

    return 0; 
}


/* =============== Info from Valgrind ===============

weili@LAPTOP-UG2EDDHM ~/development/Valgrind$ valgrind ./main                                             
==6397== Memcheck, a memory error detector
==6397== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==6397== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==6397== Command: ./main
==6397== 
==6397== Invalid write of size 4
==6397==    at 0x109208: func1() (ValgrindTool.cpp:43)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397==  Address 0x4dd4cf8 is 0 bytes after a block of size 40 alloc'd
==6397==    at 0x484A2F3: operator new[](unsigned long) (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==6397==    by 0x1091FB: func1() (ValgrindTool.cpp:42)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397== 
==6397== Mismatched free() / delete / delete []
==6397==    at 0x484BB6F: operator delete(void*, unsigned long) (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==6397==    by 0x109223: func1() (ValgrindTool.cpp:44)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397==  Address 0x4dd4cd0 is 0 bytes inside a block of size 40 alloc'd
==6397==    at 0x484A2F3: operator new[](unsigned long) (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==6397==    by 0x1091FB: func1() (ValgrindTool.cpp:42)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397== 
==6397== Use of uninitialised value of size 8
==6397==    at 0x109228: func1() (ValgrindTool.cpp:47)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397== 
==6397== Invalid read of size 4
==6397==    at 0x109228: func1() (ValgrindTool.cpp:47)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397==  Address 0x0 is not stack'd, malloc'd or (recently) free'd
==6397== 
==6397== 
==6397== Process terminating with default action of signal 11 (SIGSEGV)
==6397==  Access not within mapped region at address 0x0
==6397==    at 0x109228: func1() (ValgrindTool.cpp:47)
==6397==    by 0x109290: main (ValgrindTool.cpp:59)
==6397==  If you believe this happened as a result of a stack
==6397==  overflow in your program's main thread (unlikely but
==6397==  possible), you can try to increase the size of the
==6397==  main thread stack using the --main-stacksize= flag.
==6397==  The main thread stack size used in this run was 8388608.
==6397== 
==6397== HEAP SUMMARY:
==6397==     in use at exit: 72,705 bytes in 2 blocks
==6397==   total heap usage: 3 allocs, 1 frees, 72,745 bytes allocated
==6397== 
==6397== LEAK SUMMARY:
==6397==    definitely lost: 1 bytes in 1 blocks
==6397==    indirectly lost: 0 bytes in 0 blocks
==6397==      possibly lost: 0 bytes in 0 blocks
==6397==    still reachable: 72,704 bytes in 1 blocks
==6397==         suppressed: 0 bytes in 0 blocks
==6397== Rerun with --leak-check=full to see details of leaked memory
==6397== 
==6397== Use --track-origins=yes to see where uninitialised values come from
==6397== For lists of detected and suppressed errors, rerun with: -s
==6397== ERROR SUMMARY: 4 errors from 4 contexts (suppressed: 0 from 0)
[1]    6397 segmentation fault  valgrind ./main

*/