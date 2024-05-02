/**
 * @file 03_structuredBindings.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>

/**
 * @brief Structured Bindings 结构化绑定
 * 通过结构化绑定, 可以将对象解压缩为它们的组成元素;
 * 使用这种方式, 可以解压缩非静态数据成员为 public 的任何类型.
 * auto [object-1, object-2, ...] = plain-old-data;
 * ?这些对象从顶部到底部逐渐剥离 POD, 并且从左到右填充结构化绑定声明.
 * 
 * ?====Attributes
 * 属性(attribute)将实现定义的功能应用于表达式语句;
 * 可以使用双括号 [[]] 来引入属性, 该括号包含由一个或多个用逗号分隔的属性元素组成的列表;
 * !标准属性 https://en.cppreference.com/w/cpp/language/attributes
 * 
 * 编译器可以根据这些信息更全面地对代码进行推理(并有可能优化程序),
 * 很少有需要使用属性的情况, 尽管如此, 它们仍然可以向编译器传达有用的信息.
 * *最常见的是库函数里面提示该函数未来被移除的警告
 * 
 */

struct TextFile
{
    bool        success;  // 告知调用者函数调用是否成功
    const char *contents; // 文件内容
    size_t      n_bytes;  // 文件大小
};

TextFile read_text_file(const char *path)
{
    static const char contents[]{"Sometimes the goat is you."};
    return TextFile{true, contents, sizeof(contents)};
}

// [[noreturn]] 表示函数不返回
[[noreturn]] void pitcher()
{
    throw std::runtime_error{"Knuckleball."};
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // 使用结构化绑定声明将结果分解为三个不同的变量
    // ?结构化绑定声明中的类型不必匹配
    const auto [success, contents, length] = read_text_file("README.txt");

    if (success)
    {
        printf("Read %zd bytes: %s\n", length, contents);
    }
    else
    {
        printf("Failed to open README.txt.\n");
    }

    printf("========= Attributes =========\n");
    try
    {
        pitcher();
    }
    catch (const std::exception &e)
    {
        printf("exception: %s\n", e.what());
    }

    return 0;
}
