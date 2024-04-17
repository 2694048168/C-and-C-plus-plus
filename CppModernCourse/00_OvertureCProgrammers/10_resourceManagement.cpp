/**
 * @file 10_resourceManagement.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <cstring>
#include <iostream>
#include <system_error>

/**
 * @brief C++给系统编程带来的最大创新是对象生命周期,
 *  这个概念源于C语言, 对象具有不同的存储期, 这取决于它们在代码中的声明方式;
 *  C++在这个内存管理模型的基础上, 创造了构造函数和析构函数, 这些特殊函数是属于用户自定义类型的方法;
 *  用户自定义类型是C++应用程序的基本构建块, 可以把它想象成有函数的 struct 对象;
 *  对象的构造函数在其存储期开始后被调用, 而析构函数则在其存储期结束前被调用;
 * 
 * ========= Step 1 =========
 * 编译器会确保静态、本地和线程局部存储期的对象自动调用构造函数和析构函数;
 * 对于具有动态存储期的对象, 可以使用关键字 new 和 delete 来分别代替 malloc和 free;
 * 
 * ========= Step 2 =========
 * 如果构造函数无法达到一个好的状态(无论什么原因), 那么它通常会抛出异常;
 * 作为C语言程序员, 可能在使用一些操作系统API编程时处理过异常, 例如 Windows结构化异常处理;
 * 当抛出异常时, 堆栈会展开, 直到找到异常处理程序, 这时程序就会恢复; C++对异常有语言级的支持.
 * NOTE: 谨慎地使用异常可以使代码更干净, 因为只需要在有意义的地方检查错误条件.
 * 
 * 构造函数(constructor), 析构函数(destructor)和异常(exception)与另一个C++核心主题密切相关,
 * 该核心主题便是把对象的生命周期与它所拥有的资源联系起来;
 * 这就是资源获取即初始化(RAII)的概念(有时也称为构造函数获取, 析构函数释放)
 * 
 * 
 */

class CodeVersion
{
public:
    CodeVersion()
        : m_version(8888)
    {
        std::cout << "Constructor function: " << m_version << '\n';
    }

    ~CodeVersion()
    {
        std::cout << "Destructor function\n";
    }

    void printInfo()
    {
        std::cout << "CodeVersion.printInfo is called\n";
    }

private:
    const int m_version;
};

/**
 * @brief File 构造函数代码段中, 构造函数试图以读/写访问方式打开位于 path 路径的文件;
 *  如果有问题, 本次调用将把 file_pointer 设置为 nullptr, 这是一个特殊的C++值, 类似于0;
 *  当发生这种情况时, 将抛出一个 system_error, system_error 是封装了系统错误细节的对象;
 * 如果 file_pointer 不是 nullptr, 它就可以有效地使用, 这就是这个类的不变量.
 * 
 */
class File
{
public:
    File(const char *path, bool write)
    {
        auto file_mode = write ? "w" : "r";
        file_pointer   = fopen(path, file_mode);

        if (!file_pointer)
        {
            throw std::system_error(errno, std::system_category());
        }
    }

    ~File()
    {
        fclose(file_pointer);
    }

public:
    FILE *file_pointer;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // ========= Step 1 =========
    std::cout << "\n========= Step 1 =========\n";
    // Memory is allocated, then constructor is called
    auto p_obj = new CodeVersion{};
    p_obj->printInfo();

    if (nullptr != p_obj)
    {
        // Destructor is called, then memory is deallocated
        delete p_obj;
        p_obj = nullptr;
    }
    // ==================================================

    // ========= Step 2 =========
    std::cout << "\n========= Step 2 =========\n";
    int *ptr = nullptr;
    try
    {
        ptr    = new int[100000000ul];
        ptr[0] = 42;
        std::cout << "the first number value: " << *ptr << '\n';
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Allocation failed: " << e.what() << '\n';
    }

    if (nullptr != ptr)
    {
        delete[] ptr;
        ptr = nullptr;
    }
    // ==================================================

    // ========= Step 3 =========
    std::cout << "\n========= Step 3 =========\n";
    {
        File       file("last_message.txt", true);
        const auto message = "We apologize for the inconvenience.\n";
        fwrite(message, strlen(message), 1, file.file_pointer);
    } // last_message.txt is closed here!
    {
        File file("last_message.txt", false);
        char read_message[40]{};
        fread(read_message, sizeof(read_message), 1, file.file_pointer);
        printf("Read last message: %s\n", read_message);
    }
    /**
     * @brief 有时需要动态内存分配的灵活性, 但仍然想依靠C++的对象生命周期来
     * 确保不会泄漏内存或意外地“释放后使用”, 这正是智能指针的作用,
     * 它通过所有权模型来管理动态对象的生命周期, 
     * 一旦没有智能指针拥有动态对象(引用计数的方式), 该对象就会被销毁.
     * 
     */
    // ==================================================

    return 0;
}
