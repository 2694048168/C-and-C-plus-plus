#include "CrashDumpHelper.h"

#include <Windows.h>
#include <dbghelp.h>

#include <iostream>
#include <mutex>
#pragma comment(lib, "DbgHelp.lib")

// 静态成员初始化
std::mutex                   gDbghelpLock;
LPTOP_LEVEL_EXCEPTION_FILTER gPreviousFilter = nullptr;
MINIDUMP_TYPE                gDumpType       = MiniDumpNormal;
std::wstring                 gDumpFileName   = L"CrashCore.dmp";

void WriteDump(MINIDUMP_EXCEPTION_INFORMATION *exception_info)
{
    HANDLE file_handle = ::CreateFileW(gDumpFileName.c_str(), GENERIC_WRITE,
                                       0, // 独占访问
                                       NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (file_handle == INVALID_HANDLE_VALUE)
    {
        std::cerr << "CreateFile failed: " << ::GetLastError() << std::endl;
        return;
    }

    BOOL result = FALSE;
    {
        // 线程安全锁（DbgHelp非线程安全）[[14]]
        std::lock_guard<std::mutex> lock(gDbghelpLock);
        result = ::MiniDumpWriteDump(::GetCurrentProcess(), ::GetCurrentProcessId(), file_handle,
                                     gDumpType, // 可配置的转储类型
                                     exception_info, NULL, NULL);
    }
    ::CloseHandle(file_handle);

    if (!result)
    {
        std::cerr << "MiniDumpWriteDump failed: " << ::GetLastError() << std::endl;
    }
    else
    {
        std::wcout << L"Dump created: " << gDumpFileName << std::endl;
    }
}

LONG WINAPI HandleExceptionFilter(EXCEPTION_POINTERS *exception_pointers)
{
    MINIDUMP_EXCEPTION_INFORMATION exception_info = {0};
    exception_info.ThreadId                       = ::GetCurrentThreadId();
    exception_info.ExceptionPointers              = exception_pointers;
    exception_info.ClientPointers                 = FALSE; // 使用进程内存空间

    WriteDump(&exception_info);

    // 链式调用原异常过滤器
    if (gPreviousFilter)
        return gPreviousFilter(exception_pointers);
    return EXCEPTION_CONTINUE_SEARCH;
}

void CrashCoreDump::Initialize(const std::wstring &dumpFileName, unsigned long long dumpTypeMask)
{
    gDumpFileName = dumpFileName;
    gDumpType     = static_cast<MINIDUMP_TYPE>(dumpTypeMask);

    // 设置异常过滤器并保存原指针
    gPreviousFilter = ::SetUnhandledExceptionFilter(HandleExceptionFilter);
}

void CrashCoreDump::Shutdown()
{
    // 恢复原异常过滤器 
    ::SetUnhandledExceptionFilter(gPreviousFilter);
}
