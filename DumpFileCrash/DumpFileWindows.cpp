#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <dbghelp.h>
// add_compile_options(/Zi)
// add_link_options(/ DEBUG)
// set(CMAKE_CXX_STANDARD 20)
#pragma comment(lib, "Dbghelp.lib")
#pragma comment(lib, "wininet.lib")
#include <string>
#include <chrono>
using namespace std::chrono_literals;

inline std::string get_current_date()
{
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    auto tNow = std::chrono::system_clock::now();
    auto tMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(tNow.time_since_epoch());
    auto tSeconds = std::chrono::duration_cast<std::chrono::seconds>(tNow.time_since_epoch());
    auto ms = tMilliseconds - tSeconds;
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

inline void CreateDumpFile(EXCEPTION_POINTERS *exceptionInfo)
{
    HANDLE hFile = CreateFile((LPCWSTR)(get_current_date() + ".dmp").c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE)
    {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = exceptionInfo;
        mdei.ClientPointers = FALSE;

        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpWithFullMemory, &mdei, NULL, NULL);
        CloseHandle(hFile);
    }
}

LONG WINAPI ExceptionHandler(EXCEPTION_POINTERS *exceptionInfo)
{
    CreateDumpFile(exceptionInfo);
    return EXCEPTION_EXECUTE_HANDLER; // 继续执行默认的崩溃处理
}

int main()
{
    SetUnhandledExceptionFilter(ExceptionHandler);
    int *p = nullptr;
    *p = 0;
}
