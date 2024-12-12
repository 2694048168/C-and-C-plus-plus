/**
 * @file 08_file_reader.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// 将文件读取到字符串中的函数
std::string file_reader(const char *fname)
{
    std::ifstream file_handle;
    file_handle.open(fname);
    if (!file_handle)
    {
        std::cout << "Can't open " << fname << " for reading\n";
        return "";
    }

    std::stringstream ss_buffer;
    std::copy(std::istreambuf_iterator<char>(file_handle.rdbuf()), std::istreambuf_iterator<char>(),
              std::ostreambuf_iterator<char>(ss_buffer));
    return ss_buffer.str();
}

// 作为库函数,进一步优化该功能函数, 打开文件的处理交给调用者
void stream_read_streambuf_stringstream(std::istream &file_handle, std::string &result)
{
    std::stringstream ss_buffer;
    std::copy(std::istreambuf_iterator<char>(file_handle.rdbuf()), std::istreambuf_iterator<char>(),
              std::ostreambuf_iterator<char>(ss_buffer));
    // std::swap(result, ss_buffer.str());
    result = ss_buffer.str();
    /* 本来可以不进行交换, 将 s.str() 赋值给 result 即可,
    但是除非编译器和标准库实现都支持移动语义,否则这样做会导致内存分配和复制;
    std::swap() 对许多标准库类的特化实现都是调用它们的 swap() 成员函数;
    该成员函数会交换指针, 这远比内存分配和复制操作的开销小. */
}

// 复制流迭代器的文件读取函数
void stream_read_streambuf_string(std::istream &file_handle, std::string &result)
{
    result.assign(std::istreambuf_iterator<char>(file_handle.rdbuf()), std::istreambuf_iterator<char>());
}

// 缩短调用链
void stream_read_streambuf(std::istream &file_handle, std::string &result)
{
    std::stringstream ss_buffer;
    ss_buffer << file_handle.rdbuf();
    // std::swap(result, ss_buffer.str());
    result = ss_buffer.str();
}

// 减少重新分配, result 预先分配存储空间
void stream_read_string_reserve(std::istream &file_handle, std::string &result)
{
    file_handle.seekg(0, std::istream::end);
    std::streamoff len = file_handle.tellg();
    file_handle.seekg(0);
    if (len > 0)
        result.reserve(static_cast<std::string::size_type>(len));
    result.assign(std::istreambuf_iterator<char>(file_handle.rdbuf()), std::istreambuf_iterator<char>());
}

// 通用版本的 stream_read_string()
void stream_read_string_2(std::istream &file_handle, std::string &result, std::streamoff len = 0)
{
    if (len > 0)
        result.reserve(static_cast<std::string::size_type>(len));
    result.assign(std::istreambuf_iterator<char>(file_handle.rdbuf()), std::istreambuf_iterator<char>());
}

// 更大的吞吐量——使用更大的输入缓冲区
// C++ 流包含一个继承自 std::streambuf 的类, 用于改善从操作系统底层以更大块的数据单位读取文件时的性能

// 更大的吞吐量——一次读取一行
void stream_read_getline(std::istream &f, std::string &result)
{
    std::string line;
    result.clear();
    while (getline(f, line))
    {
        (result += line) += "\n";
    }
}

// 计算流长度并预先分配存储空间的技巧很实用
// 优秀的库设计总是会在它们自己的函数中复用这些工具
std::streamoff stream_size(std::istream &f)
{
    std::istream::pos_type current_pos = f.tellg();
    if (-1 == current_pos)
        return -1;
    f.seekg(0, std::istream::end);
    std::istream::pos_type end_pos = f.tellg();
    f.seekg(current_pos);
    return end_pos - current_pos;
}

// 提高吞吐量的方法是使用 std::streambuf 的成员函数 sgetn()
// 它能够获取任意数量的数据到缓冲区参数中
bool stream_read_sgetn(std::istream &f, std::string &result)
{
    std::streamoff len = stream_size(f);
    if (len == -1)
        return false;
    result.resize(static_cast<std::string::size_type>(len));
    f.rdbuf()->sgetn(&result[0], len);
    return true;
}

// 再次缩短函数调用链
// std::istream 提供了一个 read() 成员函数，它能够将字符直接复制到缓冲区中
bool stream_read_string(std::istream &f, std::string &result)
{
    std::streamoff len = stream_size(f);
    if (len == -1)
        return false;
    result.resize(static_cast<std::string::size_type>(len));
    f.read(&result[0], result.length());
    return true;
}

// C++11 标准在 21.4.1 节中首次清晰地要求字符串必须连续地存储字符
bool stream_read_array(std::istream &f, std::string &result)
{
    std::streamoff len = stream_size(f);
    if (len == -1)
        return false;
    std::unique_ptr<char> data(new char[static_cast<size_t>(len)]);
    f.read(data.get(), static_cast<std::streamsize>(len));
    result.assign(data.get(), static_cast<std::string::size_type>(len));
    return true;
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    std::cout << "The optimized for I/O\n";

    return 0;
}
