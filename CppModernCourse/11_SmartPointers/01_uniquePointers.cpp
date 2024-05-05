/**
 * @file 01_uniquePointers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <memory>

/**
* @brief 独占指针对单个动态对象具有可转移的专属所有权.
* 可以移动独占指针, 是可转移的, 还拥有对对象的专属所有权, 因此它们无法被复制;
* stdlib 在＜memory＞ 头文件中定义了一个 unique_ptr 作为独占指针.
* 
*/

// !删除器类型与 fclose 的类型匹配
using FileGuard = std::unique_ptr<FILE, int (*)(FILE *)>;

void say_hello(FileGuard file)
{
    fprintf(file.get(), "Hello Modern C++\n");
}

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("========= the Constructing =========\n");
    // std::unique_ptr 接受与指向的类型相对应的单个模板参数
    // 独占指针也有一个默认构造函数, 该构造函数将独占指针初始化为空
    // 还提供了一个接受原始指针的构造函数, 该指针获得所指向的动态对象的所有权
    std::unique_ptr<int> unique_ptr_int{new int{42}};

    // 使用 std::make_unique 函数, 一个模板接受所有参数并将它们传递给模板参数的适当构造函数
    auto unique_ptr_int_ = std::make_unique<int>(24);

    printf("the pointer value: %d\n", *unique_ptr_int);
    printf("the pointer value: %d\n", *unique_ptr_int_);

    // 可转移的专属所有权
    auto unique_ptr_move = std::move(unique_ptr_int);
    printf("the pointer value: %d\n", *unique_ptr_move);
    // printf("the pointer value: %d\n", *unique_ptr_int); //!runtime-error

    // 独占数组 Unique Arrays
    printf("UniquePtr to array supports operator[]\n");
    std::unique_ptr<int[]> squares{
        new int[5]{1, 4, 9, 16, 25}
    };
    squares[0] = 1;
    assert(squares[0] == 1);
    assert(squares[1] == 4);
    assert(squares[2] == 9);

    // 删除器 Deleter
    printf("删除器 Deleter\n");
    // std::unique_ptr 有第二个可选的模板参数, 称为删除器类型;
    // 当独占指针需要销毁拥有的对象时, 就会调用删除器;
    // 默认情况下, Deleter 是 std::default_delete＜T＞, 它在动态对象上调用 delete 或 delete[]
    auto my_deleter = [](int *x)
    {
        printf("[custom deleter]Deleting an int at %p.\n", x);
        delete x;
    };

    std::unique_ptr<int, decltype(my_deleter)> my_up{new int{}, my_deleter};
    printf("the custom deleter: %d\n\n", *my_up);

    /**
     * @brief 自定义删除器和系统编程 
     * 用 ＜cstdio＞ 头文件中的 fopen、fprintf 和 fclose 等底层 API 来管理文件
     * FILE* fopen(const char *filename, const char *mode);
     * void fclose(FILE* file);
     * *FILE* 文件句柄是对操作系统管理的文件的引用;
     * *句柄(handle)是对操作系统中某些非透明资源的抽象引用; 
     * 
     */
    auto file = fopen("Test_HAL.txt", "w");
    if (!file)
        return errno;

    // 文件已打开, 并且由于其自定义删除器, file_guard 会自动管理文件的生命周期
    FileGuard file_guard{file, fclose};
    // File open here
    // 要调用 say_hello, 需要将所有权转移到该函数中(因为它需要按值获取 FileGuard)
    say_hello(std::move(file_guard));
    // File closed here

    return 0;
}
