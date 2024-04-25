/**
 * @file 01_pointerArray.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
  * @brief ===== 指针和数组
  * 指针与数组有几个共同的特点, 指针对对象的位置进行编码, 而数组则对连续对象的位置和长度进行编码.
  * 只要稍微对数组施加操作, 数组就会退化成指针, 
  * 退化后的数组会失去长度信息, 并转换为指向数组第一个元素的指针.
  * 
  */

struct College
{
    char name[256];
};

void print_name(College *college_ptr)
{
    /**
   * @brief print_name 函数接受一个指向 College 的指针参数,
   *  所以当调用 print_name 时 best_colleges 数组会退化为一个指针,
   *  因为数组会退化为指向第一个元素的指针, college_ptr 指向 best_colleges 中的第一个 College.
   * 它使用箭头运算符(->)访问 college_ptr 所指向的College 的 name 成员,
   * name 成员本身就是一个 char 数组,
   * printf 格式指定符 %s 期望C格式字符串: char 指针,
   * 因此 name 退化为满足 printf 要求的指针.
   * 
   */
    printf("%s College\n", college_ptr->name);
}

void print_names(College *colleges, size_t n_colleges)
{
    for (size_t i = 0; i < n_colleges; ++i)
    {
        printf("%s College\n", colleges[i].name);
    }
}

// ----------------------------------
int main(int argc, const char **argv)
{
    int  key_universe[]{3, 6, 9};
    int *key_ptr = key_universe;
    printf("the array first-address: %p\n", key_universe);
    printf("the pointer first-address: %p\n", key_ptr);
    printf("the array first-value: %d\n", *key_ptr);
    printf("the pointer to value: %d\n", *key_ptr);

    // 数组会退化为一个指针
    College best_colleges[] = {"Magdalen", "Nuffield", "Kellogg"};
    print_name(best_colleges);

    /**
     * @brief -------- 处理退化问题
     * step 1. 将数组作为两种参数传递:
     * 1. 指向第一个元素的指针;
     * 2. 数组的长度;
     * 实现这种模式的机制是方括号[], 方括号对指针的作用就像对数组的作用一样.
     * 
     * NOTE: 这种指针加长度的传递数组的方法在C风格的API中无处不在, 例如在Windows或Linux的系统编程中.
     *
     * step 2. 指针算术:
     * 要获得数组中第 n 个元素的地址, 第一种方法是直接用方括号([])获取第 n 个元素, 然后使用地址运算符(&)获得地址;
     * College* third college_ptr = &oxford[2];
     * 第二种方法, 指针算术， 也就是在指针上进行加减法的一套规则，
     * 当在指针上加减整数时， 编译器会使用指针指向的类型的大小计算出正确的字节偏移,
     * College* third college ptr = oxford + 2;
     * 
     */
    print_names(best_colleges, sizeof(best_colleges) / sizeof(College));

    College *third_college_ptr = &best_colleges[2];
    printf("the third element value: %s\n", third_college_ptr->name);

    College *third_college_ptr_ = best_colleges + 2;
    printf("the third element value: %s\n", third_college_ptr_->name);

    /**
     * @brief ====== 指针很危险
     * 无法将指针转换为数组, 编译器也不可能通过指针知道数组的大小.
     * 1. 缓冲区溢出:
     * 对于数组和指针可以使用括号运算符([])或指针算术来访问任意数组元素,
     * 通过写入越界内存产生一个重大错误, 访问位于越界索引的元素,
     * 没有边界检查, 这段代码在编译时也没有发出警告,
     * 在运行时, 会得到未定义行为, 未定义行为意味着C++语言规范没有规定会发生什么,
     * 所以程序可能会崩溃, 也可能会产生安全漏洞, 还可能会创造一个人工智能^_^.
     * 
     */
    char  lower[]   = "abc?e";
    char  upper[]   = "ABC?E";
    char *upper_ptr = upper;        // Equivalent: &upper[0]
    lower[3]        = 'd';          // lower now contains a b c d e \0
    upper_ptr[3]    = 'D';          // upper now contains A B C D E \0
    char letter_d   = lower[3];     // letter_d equals 'd'
    char letter_D   = upper_ptr[3]; // letter_D equals 'D'
    printf("lower: %c\n", letter_d);
    printf("upper: %c\n", letter_D);

    printf("lower: %s\nupper: %s\n", lower, upper);

    // lower[7] = 'g'; // !Super bad. You must never do this.
    // printf("lower[7]: %c\n", lower[7]);

    /**
    * @brief lower 数组的长度为6(字母a~e加上nul结束符),
    * 给 lower[7] 赋值很危险的原因便很清楚了,
    * 在这种情况下, 数据会写到一些不属于 lower 的内存中,
    * 这可能导致访问违规, 程序崩溃, 安全漏洞和数据损坏,
    * * 这类错误可能是非常隐蔽的, 因为错误写入发生的点可能与错误表现的点相距甚远.
    * 
    */
    *(lower + 7) = 'g';                         // !ERROR
    printf("*(lower + 7): %c\n", *(lower + 7)); // !ERROR

    /**
     * @brief ======== void指针和std::byte指针
     * 有时候, 指向的对象类型是不确定的, 这种情况下, 可以使用空(void)指针 void*,
     * void指针有严格的限制, 其中最主要的限制是不能对 void*进行解引用,
     * 因为指向的类型已经被擦除了所以解引用是没有意义的(void 对象的值的集合是空的), C++禁止使用void 指针算术.
     * 其他时候想在字节级别与内存进行交互,
     * 例如, 在文件和内存之间复制原始数据或者加密和压缩等底层操作,
     * 不能使用 void 指针来实现这样的目的, 因为位操作和指针算术是被禁止的.
     * 这种情况下, 使用 std::byte 指针.
     * 
     * ======== nullptr和布尔表达式
     * 指针可以有一个特殊的字面量 nullptr, 一般来说,等于 nullptr 的指针不指向任何东西.
     *  可以用 nullptr 表示没有更多的内存可以分配了或者发生了错误,
     * 指针具有隐式转换为布尔值的功能, 任何不是 nullptr 的值都会隐式转换为 true,
     * 而 nullptr 则会隐式转换为 false, 这对于返回函数成功运行的指针很有用.
     * 一个常见的用法是, 函数运行失败时返回 nullptr, 典型的例子是内存分配.
     * 
     */
    auto ptr = new int{42};
    if (!ptr)
        printf("new the memory is NOT successfully\n");

    if (ptr)
    {
        delete ptr;
        ptr = nullptr;
    }

    return 0;
}
