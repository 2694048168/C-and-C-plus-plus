/**
 * @file 02_get_exePath.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <Windows.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

std::string GetExePath()
{
    char szFilePath[MAX_PATH + 1] = {0};
    GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
    /*
	strrchr:函数功能：查找一个字符c在另一个字符串str中末次出现的位置
    (也就是从str的右侧开始查找字符c首次出现的位置),
	并返回这个位置的地址。如果未能找到指定字符，那么函数将返回NULL.
	使用这个地址返回从最后一个字符c到str末尾的字符串.                                                     
	*/
    (strrchr(szFilePath, '\\'))[0] = 0; // 删除文件名,只获得路径字串
    std::string path               = szFilePath;
    return path;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::string str = GetExePath();
    std::cout << "The exe path: " << str << std::endl;

    return 0;
}

// g++ 02_get_exePath.cpp -std=c++17
