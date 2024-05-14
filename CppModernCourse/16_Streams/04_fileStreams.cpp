/**
 * @file 04_fileStreams.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <fstream>
#include <iostream>
#include <limits>

/**
  * @brief File Streams 文件流
  * 文件流类为读取和写入字符序列提供了便利, 文件流类结构与字符串流类的结构类似.
  * *文件流类模板可用于输入、输出或同时用于输入和输出.
  * ?与使用本机系统调用与文件内容交互相比, 文件流类有以下主要优势:
  * 1. 将获得通用的流接口, 它提供了一组丰富的用于格式化和操纵输出的功能;
  * 2. 文件流类是围绕文件的 RAII 包装器, 这意味着不会出现资源泄漏;
  * 3. 文件流类支持移动语义, 因此可以严格控制文件在作用域内的位置;
  * 
  */

/**
* @brief 1. 使用流打开文件
* 有两种选择, 以使用流打开文件,
* 1. 第一个选择是 open 方法, 它接受一个 const char* filename 
* 和一个可选的 std::ios_base::openmode 位掩码参数, openmode参数可以是许多可能的值组合之一.
* ?Flag(位于 std::ios): in / out / app / in|out / in|app / out|app / out|trunc  
* ?in|out|app / in|out|trunc / 
* *可以将 binary 标志添加到任意组合中, 以将文件置于二进制模式,
* *在二进制模式下, 流不会转换特殊字符序列, 如行尾(例如 Windows 上的回车加换行符)或 EOF
* 
* 2. 指定要打开的文件的第二个选择是使用流的构造函数,
* 每个文件流都提供了一个构造函数, 其参数与 open 方法相同.
* 所有文件流类都是围绕它们拥有的文件句柄的 RAII 包装器, 因此当文件流销毁时,文件将被自动清理.
* 也可以手动调用不带参数的 close 方法, 当清楚地知道文件操作已经完成, 但是文件流类对象在一段时间内是不会销毁的.
* 文件流也有默认构造函数, 它不打开任何文件, 要检查文件是否打开, 
* 请调用 is_open 方法, 该方法不接受任何参数并返回一个布尔值.
* 
*/

std::ifstream open(const char *path, std::ios_base::openmode mode = std::ios_base::in)
{
    std::ifstream file{path, mode};
    if (!file.is_open())
    {
        std::string err{"Unable to open file "};
        err.append(path);
        throw std::runtime_error{err};
    }
    file.exceptions(std::ifstream::badbit);
    return file;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 2. 输出文件流
     * 输出文件流为字符序列提供输出流语义, 派生自＜fstream＞头文件中的
     * 类模板 std::basic_ofstream, 它提供以下特化:
     * *using ofstream = basic_ofstream＜char＞;
     * *using wofstream = basic_ofstream＜wchar_t＞;
     * basic_ofstream 的默认构造函数不打开文件, 非默认构造函数的第二个可选参数默认为 ios::out.
     * 每当将输入发送到文件流时，流都会将数据写入相应的文件
     * 
     */
    std::ofstream file{"lunchtime.txt", std::ios::out | std::ios::app};
    file << "Time is an illusion." << std::endl;
    file << "Lunch time, " << 2 << "x so." << std::endl;

    /**
     * @brief 3. 输入文件流
     * 输入文件流为字符序列提供输入流语义, 派生自＜fstream＞头文件中的
     * 类模板 std::basic_ifstream, 它提供以下特化:
     * *using ifstream = basic_ifstream＜char＞;
     * *using wifstream = basic_ifstream＜wchar_t＞;
     * basic_ifstream 的默认构造函数不打开文件, 非默认构造函数的第二个可选参数默认为 ios::in.
     * 每当从文件流中读取数据时, 该流都会从相应的文件中读取数据.
     * 
     */
    std::ifstream file_in{"numbers.txt"};

    auto maximum = std::numeric_limits<int>::min();
    int  value;
    while (file_in >> value)
    {
        maximum = maximum < value ? value : maximum;
    }
    std::cout << "Maximum found was " << maximum << std::endl;

    /**
     * @brief 4. 处理失败的情况
     * 与其他流一样,文件流以静默的方式失败. 如果使用文件流构造函数打开文件,
     * 则必须检查 is_open 方法以确定流是否成功打开文件.
     * 这种设计不同于大多数其他 stdlib 对象, 其中不变量由异常强制执行,
     * 很难说为什么库实现者选择了这种方法, 但事实是可以很容易地选择基于异常的方法.
     * ?可以创建自己的工厂函数来处理抛出异常的文件打开失败的情况.
     * 
     */

    // 5. 文件流操作总结
    // TODO: https://en.cppreference.com/w/cpp/io/basic_fstream

    return 0;
}
