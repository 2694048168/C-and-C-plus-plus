/**
 * @file 06_randomAccess.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <exception>
#include <fstream>
#include <iostream>

/**
 * @brief 随机访问 Random Access
 * 有时希望随机访问流(尤其是文件流), 输入和输出运算符显然不支持这种用例,
 * 因此 basic_istream 和 basic_ostream 提供了单独的随机访问方法,
 * 这些方法跟踪光标或位置, 即流当前字符的索引, 该位置指示输入流将读取的下一个字节或输出流将写入的下一个字节.
 * 
 * 对于输入流, 可以使用 tellg 和 seekg 两种方法, 
 * 1. tellg 方法不接受任何参数并返回光标位置;
 * 2. seekg 方法允许设置光标位置, 它有两个重载, 第一个重载需要提供一个pos_type 位置参数, 它设置读取位置;
 *   第二个重载需要提供一个 off_type 偏移参数和一个 ios_base::seekdir 方向参数.
 *   pos_type 和 off_type 由 basic_istream 或 basic_ostream 的模板参数确定,
 *   但通常这些会转换为整数类型或从整数类型转换而来.
 * 
 * seekdir 类型采用以下三个值之一:
 * 1. std::ios_base::beg 指定位置参数是相对于开头的;
 * 2. std::ios_base::cur 指定位置参数是相对于当前位置的;
 * 3. std::ios_base::end 指定位置参数是相对于结尾的;
 * 对于输出流, 可以使用 tellp 和 seekp 两种方法, 它们大致类似于输入流的 tellg 和 seekg 方法,
 * 其中 p 代表 put, g 代表 get.
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
    // 考虑包含以下内容的文件 introspection.txt:
    // The problem with introspection is that it has no end.
    try
    {
        auto intro = open("introspection.txt");

        std::cout << "Contents: " << intro.rdbuf() << std::endl;
        intro.seekg(0);

        std::cout << "Contents after seekg(0): " << intro.rdbuf() << std::endl;
        intro.seekg(-4, std::ios_base::end);

        std::cout << "tellg() after seekg(-4, ios_base::end): " << intro.tellg() << std::endl;
        std::cout << "Contents after seekg(-4, ios_base::end): " << intro.rdbuf() << std::endl;
    }
    catch (const std::exception &exp)
    {
        std::cerr << exp.what();
    }

    return 0;
}
