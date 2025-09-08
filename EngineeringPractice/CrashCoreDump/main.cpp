/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-08
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "CrashDumpHelper.h"

// --------------------------------------
int main(int argc, const char *argv[])
{
    //开始位置
    CrashCoreDump::Initialize(L"myapp_crash.dmp", 0x00000002 | 0x00001000
                              /*static_cast<MINIDUMP_TYPE>(
			MiniDumpWithFullMemory |
			MiniDumpWithThreadInfo)*/
    );

    //结束位置
    CrashCoreDump::Shutdown();

    return 0;
}
