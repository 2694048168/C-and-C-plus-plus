/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Tabular printing of a list of processes
 *  Suppose you have a snapshot of the list of all processes in a system. 
 The information for each process includes name, identifier, 
 status (which can be either running or suspended),
account name (under which the process runs), memory size in bytes, 
and platform (which can be either 32-bit or 64-bit). 
Your task is to write a function that takes such a list of processes 
and prints them to the console alphabetically, in tabular format. 
All columns must be left-aligned, except for the memory column which must be right-aligned. 
The value of the memory size must be displayed in KB.
 */

/**
 * @brief Solution:
 To solve this problem, 
 we will consider the following class representing information about a process

 In order to print the status and platform as text and not as numerical values, we need
conversion functions from the enumerations to std::string
------------------------------------------------------ */
enum class processStatus
{
    suspended,
    running
};

std::string status_to_string(const processStatus status)
{
    if (status == processStatus::suspended)
        return "suspended";
    else
        return "running";
}

enum class platforms
{
    p32bit,
    p64bit
};

std::string platform_to_string(const platforms platform)
{
    if (platform == platforms::p32bit)
        return "32-bit";
    else
        return "64-bit";
}

struct procinfo
{
    int           id;
    std::string   name;
    processStatus status;
    std::string   account;
    size_t        memory;
    platforms     platform;
};

/**
 * @brief The processes are required to be sorted alphabetically by process name. 
 Therefore, the first step would be to sort the input range of processes. 
 For the printing itself, we should use the I/O manipulators:
 * 
 * @param processes 
 */
void print_processes(std::vector<procinfo> processes)
{
    std::sort(std::begin(processes), std::end(processes),
              [](const procinfo &p1, const procinfo &p2) { return p1.name < p2.name; });

    for (const auto &pi : processes)
    {
        std::cout << std::left << std::setw(15) << std::setfill(' ') << pi.name;
        std::cout << std::left << std::setw(8) << std::setfill(' ') << pi.id;
        std::cout << std::left << std::setw(12) << std::setfill(' ') << status_to_string(pi.status);
        std::cout << std::left << std::setw(15) << std::setfill(' ') << pi.account;
        std::cout << std::right << std::setw(10) << std::setfill(' ') << (int)(pi.memory / 1024);
        std::cout << std::left << ' ' << platform_to_string(pi.platform);
        std::cout << '\n';
    }
    std::cout << std::endl;
}

// The following program defines a list of processes
//  (you can actually retrieve the list of running processes
// using operating system-specific APIs)
// and prints it to the console in the requested format:
// ------------------------------
int main(int argc, char **argv)
{
    using namespace std::string_literals;
    std::vector<procinfo> processes{
        {  512,      "cmd.exe"s,   processStatus::running,     "SYSTEM"s,    148293, platforms::p64bit},
        { 1044,   "chrome.exe"s,   processStatus::running, "marius.ban"s,  25180454, platforms::p32bit},
        { 7108, "explorer.exe"s,   processStatus::running, "marius.ban"s,   2952943, platforms::p64bit},
        {10100,   "chrome.exe"s,   processStatus::running, "marius.ban"s, 227756123, platforms::p32bit},
        {22456,    "skype.exe"s, processStatus::suspended, "marius.ban"s,  16870123, platforms::p64bit},
    };

    std::cout << std::left << std::setw(15) << std::setfill(' ') << "Name";
    std::cout << std::left << std::setw(8) << std::setfill(' ') << "ID";
    std::cout << std::left << std::setw(12) << std::setfill(' ') << "Status";
    std::cout << std::left << std::setw(15) << std::setfill(' ') << "Account";
    std::cout << std::right << std::setw(10) << std::setfill(' ') << "Memory(KB)";
    std::cout << std::left << ' ' << "Platform";
    std::cout << '\n';
    print_processes(processes);

    return 0;
}
