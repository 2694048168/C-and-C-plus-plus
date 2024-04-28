/**
 * @file 02_dynamicStorage.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>

/**
 * @brief 动态存储期
 * 具有动态存储期的对象会根据请求进行分配和释放, 可以手动控制动态对象生命周期何时开始, 何时结束.
 * 动态对象也被称为分配对象(allocated object), 分配动态对象的主要方式是使用 new 表达式.
 * new 表达式创建给定类型的对象, 然后返回指向新创建的对象的指针. 
 * 
 * 可以在 new 表达式中初始化动态对象, Initializes dynamic object to 42
 * 类型对象分配存储空间后, 动态对象将像往常一样被初始化, 初始化完成后, 动态对象的生命周期就开始了.
 *
 * 可以使用 delete 表达式来释放动态对象, delete 表达式由 delete 关键字和指向动态对象的指针组成,
 * delete 表达式总是返回 void.
 * 
 * 被删除对象所在的内存中所包含的值是未定义的, 这意味着编译器可以产生任意内容;
 * 在实践中, 主流编译器会尽量提高效率, 所以通常情况下对象的内存将保持不变, 直到程序将其重新用于其他目的;
 * 可能不得不实现自定义的析构函数来将一些敏感的内容清空.
 * NOTE: 因为编译器通常不会在对象被删除后清理内存, 所以可能会发生一种微妙而又潜在的严重错误,
 * 称为"释放后使用"(use after free); 如果在删除对象后不小心再次使用了它,
 * 程序在功能上可能看起来正确, 因为释放的内存可能仍然包含合理的值.
 * 在某些情况下, 这些问题直到程序在生产环境中运行了很长时间
 * 或者直到安全研究人员找到了利用这个bug的方法并将其公开才会显现出来.
 * 
 * -----动态数组
 * 动态数组是具有动态存储期的数组, 可以用数组 new 表达式创建动态数组.
 * new MyType[n_elements]{init-list };
 * 可选的 init-list 是初始化列表, 用于初始化数组;
 * 数组 new 表达式返回指向新分配的数组的第一个元素的指针.
 * 要释放动态数组, 可以使用数组 delete 表达式, 与数组 new 表达式不同, 数组 delete 表达式不需要指定长度.
 * delete[] my_int_ptr; 数组 delete 表达式也返回 void.
 * 
 */

struct Tracer
{
    Tracer(const char *name)
        : name{name}
    {
        printf("%s constructed.\n", name);
    }

    ~Tracer()
    {
        printf("%s destructed.\n", name);
    }

private:
    const char *const name;
};

static Tracer       t1{"Static variable"};
thread_local Tracer t2{"Thread-local variable"};

// ----------------------------------
int main(int argc, const char **argv)
{
    int *int_ptr = new int;
    printf("the value of int_ptr: %d\n", *int_ptr);

    float *float_ptr = new float{3.1415926f};
    printf("the value of float_ptr: %f\n", *float_ptr);

    if (nullptr != int_ptr)
    {
        delete int_ptr;
        int_ptr = nullptr;
    }

    if (nullptr != float_ptr)
    {
        delete float_ptr;
        float_ptr = nullptr;
    }

    // Dynamic arrays are arrays with dynamic storage duration
    const size_t length = 10;

    int *int_array_ptr = new int[length];
    printf("the value of array with NOT init-list: ");
    for (size_t i = 0; i < length; i++)
    {
        printf(" %d ", int_array_ptr[i]);
    }
    printf("\n");

    for (size_t i = 0; i < length; i++)
    {
        int_array_ptr[i] = i + 1;
    }

    printf("the value of array: ");
    for (size_t i = 0; i < length; i++)
    {
        printf(" %d ", int_array_ptr[i]);
    }
    printf("\n");

    int *int_array_ptr_init = new int[length]{42};
    printf("the value of array by init-list: ");
    for (size_t i = 0; i < length; i++)
    {
        printf(" %d ", int_array_ptr_init[i]);
    }
    printf("\n");

    if (nullptr != int_array_ptr)
    {
        delete[] int_array_ptr;
        int_array_ptr = nullptr;
    }

    if (nullptr != int_array_ptr_init)
    {
        delete[] int_array_ptr_init;
        int_array_ptr_init = nullptr;
    }

    /**
     * @brief 内存泄漏
     * 必须确保分配的动态对象也能被释放, 如果不能被释放, 就会导致内存泄漏, 即程序不再需要的内存没有被释放.
     * 当内存泄漏时, 会耗光环境中的资源, 而这些资源永远也无法回收, 这可能会导致性能问题或更糟糕的情况.
     * NOTE: 在实践中, 程序的操作环境可能会自动清理泄漏的资源;
     * 例如如果编写的是用户模式的代码, 现代操作系统会在程序退出时清理资源;
     * 但是如果编写的是内核代码, 操作系统将不会清理资源, 只有在计算机重启时才会回收资源.
     * 
     * -----Tracing the Object Life Cycle
     */
    printf("A\n");
    Tracer t3{"Automatic variable"};
    printf("B\n");
    const auto *t4 = new Tracer{"Dynamic variable"};
    printf("C\n");

    if (t4)
    {
        delete t4;
        t4 = nullptr;
    }

    return 0;
}
