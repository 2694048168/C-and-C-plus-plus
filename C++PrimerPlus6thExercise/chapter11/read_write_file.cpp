/** C++ IO Class Inheritance:
 * . ios_base
 * |—— ios
 * |   |—— istream
 * |   |   |—— std::cin
 * |   |   |—— std::ifstream (read file from disk)
 * |   |—— ostream
 * |   |   |—— std::cout
 * |   |   |—— std::cerr
 * |   |   |—— std::clog
 * |   |   |—— std::ofstream (write file into disk)
 * |   |—— iostream
 * |   |   |—— std::fstream (read file from disk and write file into disk)
 *
 * https://cplusplus.com/doc/tutorial/files/
 *
 * 获取文件的当前位置指针, 字节为单位
 * - ofstream.tellp() 获取当前文件的位置指针
 * - ifstream.tellg() 获取当前文件的位置指针
 * - fstream.tellg() or fstream.tellp() 获取当前文件的位置指针
 *
 * 移动文件的当前位置指针, 字节为单位(两个重载函数版本)
 * - ofstream.seekp(128); ofstream.seekp(std::ios::beg); ofstream.seekp(std::ios::end)
 * - ifstream.seekg(128); ofstream.seekg(std::ios::beg); ofstream.seekg(std::ios::end)
 * - fstream.seekg(128) or fstream.seekp(128);
 *
 * 文件缓冲区以及对应的流的状态
 * fstream.flush 成员函数刷新缓冲区
 * std::end 刷新缓冲区
 * 流状态标志位: eofbit[eof() 成员函数]; badbit[bad() 成员函数]; failbit[fail() 成员函数]
 * good() 成员函数返回该三个标志的流状态
 * clear() 成员函数清除流状态
 * setstate() 成员函数重置流状态
 *
 */

#include <iostream>
#include <string>
#include <fstream>

// ------------------------------------
int main(int argc, char const *argv[])
{
    /* --------------------------------- */
    std::string filename = "./test.txt";
    std::ofstream file_writer;
    file_writer.open(filename, std::ios::out);
    // file_writer.open(filename, std::ios::trunc);
    // file_writer.open(filename, std::ios::app);
    if (!file_writer.is_open())
    {
        std::cout << "open file is not successfully, please check." << filename << "\n";
        return 0;
    }
    file_writer << "hello world!\n";
    file_writer << 42 << "\n";
    file_writer << 3.14 << "\n";

    file_writer.close();
    /* ------------------------------------- */

    /* ------------------------------------- */
    std::ifstream file_reader;
    file_reader.open(filename, std::ios::in);
    if (!file_reader.is_open())
    {
        std::cout << "read file is not successfully, please check." << filename << "\n";
        return 0;
    }
    std::string buffer;
    // while (file_reader >> buffer)
    // {
    //     std::cout << buffer << "\n";
    // }

    while (std::getline(file_reader, buffer))
    {
        std::cout << buffer << "\n";
    }

    file_reader.close();
    /* ------------------------------------- */

    return 0;
}
