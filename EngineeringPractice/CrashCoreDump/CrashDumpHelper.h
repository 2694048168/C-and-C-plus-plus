/**
 * @file CrashDumpHelper.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief windows下提供了core dump文件来分析定位该问题
 * @version 0.1
 * @date 2025-09-08
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <string>

/**
  * @brief 在软件实际运行时，由于内存越界访问、非法指针使用等情况会发生crash,
  * 遇到这种情况通常根据事件查看器/log看哪个函数crash了,
  * 然后一行行打下log最终定位，如果是偶发问题，需要花费大量时间测试复现.
  * Windows下提供了core dump文件来分析定位该问题,
  * dump文件相当于是一个快照, 包含堆信息的转储还包含该应用程序内存的快照.
  * 
  */
class CrashCoreDump
{
public:
    // 初始化模块（设置异常过滤器）    
    static void Initialize(const std::wstring &dumpFileName = L"CrashCore.dmp", unsigned long long dumpTypeMask = 0);

    // 卸载模块（恢复原异常过滤器）    
    static void Shutdown();
};
