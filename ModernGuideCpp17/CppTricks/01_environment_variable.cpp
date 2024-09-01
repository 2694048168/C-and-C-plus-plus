/**
 * @file 01_environment_variable.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>

/**
 * @brief getenv: 取得环境变量内容
 * 头文件:  #include<stdlib.h>
 * 格式: char * getenv(const char *name);
 * 意义: getenv()用来取得参数name环境变量的内容.
 * 
 * @param name为环境变量的名称, 如果该变量存在则会返回指向该内容的指针, 环境变量的格式为name＝value.
 * @return 执行成功则返回指向该内容的指针, 找不到符合的环境变量名称则返回NULL.
 * 
 */

/**
  * @brief putenv: 改变或增加环境变量
  * header-file: #include<cstdlib>
  * signature:  int putenv(const char * string);
  * 
  * @param string的格式为name＝value,如果该环境变量原先存在,
  *        则变量内容会依参数string改变,否则此参数内容会成为新的环境变量 
  * @return 执行成功则返回0, 有错误发生则返回-1
  * @note 错误代码: ENOMEM 内存不足, 无法配置新的环境变量空间. 
  *  
  */
// setenv函数是用在linux中的，在Windows中没有setenv函数而用putenv来代替
/**
  * @brief setenv: 改变或增加环境变量
  * header-file: #include<cstdlib>
  * signature: int setenv(const char *name,const char * value,int overwrite);
  * 
  * @param name为环境变量名称字符串;
  * @param value则为变量内容;
  * @param overwrite用来决定是否要改变已存在的环境变量.
  *  如果overwrite不为0, 则改变环境变量原有内容, 原有内容会被改为参数value所指的变量内容;
  *  如果overwrite为0, 且该环境变量已有内容, 则参数value会被忽略.
  * @return 执行成功则返回0, 有错误发生则返回-1
  * @note 错误代码: ENOMEM 内存不足, 无法配置新的环境变量空间. 
  *  
  */

void checkLicense()
{
    const char *variable_name{"_LICENSE_FILE"};

    auto _LICENSE_FILE = std::getenv(variable_name);
    if (nullptr == _LICENSE_FILE)
    {
        std::cout << u8"未检测到License环境变量, 请配置\n";
    }
    else
    {
        std::cout << u8"检测到License环境变量: " << variable_name << '\n';
        if (!std::filesystem::exists(_LICENSE_FILE))
        {
            std::cout << u8"未检测到License本地文件或路径, 请检查\n";
        }
        else
        {
            std::cout << u8"检测到License本地文件或路径: " << _LICENSE_FILE << '\n';
        }
    }
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // ==================
    checkLicense();

    const char *variable_name{"_LICENSE_FILE"};
    const char *license_value{"license_value_path_file"};
    // int         flag = std::setenv(variable_name, license_value, 1);
    // int         flag = putenv(variable_name);
    int         flag = _putenv(variable_name);
    if (-1 == flag)
    {
        std::cout << u8"设置License环境失败\n";
    }

    std::string env = std::getenv("PATH");
    std::cout << "PATH: " << env << '\n';

    env += ";../../runtime/3rdParty/bin";
    std::string newEnv = "PATH=" + env;
    _putenv(newEnv.c_str());
    env = std::getenv("PATH");
    std::cout << "PATH: " << env << '\n';

    checkLicense();

    return 0;
}

// g++ 01_environment_variable.cpp -std=c++17 -finput-charset=utf-8 -fexec-charset=utf-8
// clang++ 01_environment_variable.cpp -std=c++17 -finput-charset=utf-8 -fexec-charset=utf-8
// cl 01_environment_variable.cpp /std:c++17 /utf-8
