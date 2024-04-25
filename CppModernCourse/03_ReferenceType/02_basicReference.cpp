/**
 * @file 02_basicReference.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief ======== 引用 Reference
 * 引用(reference)是指针的更安全, 更方便版本,
 * 在类型名后附加&声明符即可声明引用,
 * 1. 引用不能被(轻易)设置为空,
 * 2. 不能被重新定位(或重新赋值), 这些特性消除了指针特有的一些问题.
 * 处理引用的语法比处理指针的语法要干净得多, 不使用成员指针运算符和解引用运算符,
 *  而是将引用完全当作目标类型来使用.
 * 
 * ====== 指针和引用的使用
 * 指针和引用在很大程度上是可以互换的, 但两者各有利弊.
 * 如果有时必须改变引用类型的值, 也就是说如果必须改变引用类型所指向的内容, 那么必须使用指针.
 * 许多数据结构(前向链表)都要求能够改变指针的值, 因为引用不能被重新定位, 
 * 而且它们一般不应该被赋值为nullptr, 所以有些场合并不适合使用引用.
 * 
 */

void add_num(int &value)
{
    ++value;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    float  num     = 0.3f;
    float &num_ref = num;
    printf("the num value is : %f\n", num);
    printf("the num_ref value is : %f\n", num_ref);
    num += 0.15f;
    printf("the num value is : %f\n", num);
    printf("the num_ref value is : %f\n", num_ref);

    int value = 41;
    printf("the original value: %d\n", value);
    add_num(value);
    printf("the original value: %d\n", value);

    /**
     * @brief 使用引用, 指针提供了很强的灵活性, 但这种灵活性是以安全为代价的.
     * 如果不需要灵活地重新定位和nullptr, 引用是最常用的引用类型.
     * 引用不能被重新定位的问题
     * 
     */
    int  original     = 100;
    int &original_ref = original;
    printf("Original: %d\n", original);
    printf("Reference: %d\n", original_ref);

    int new_value = 200;
    // 将另一个名为 new_value 并将它赋给 original_ref,
    // !这个赋值并没有将 original_ref 重新定位, 使其指向 new_value,
    // 而是将new_value 的值赋给它所指向的对象(original).
    // *结果就是所有这些变量 original,original_ref和 new_value 都变成了200.
    original_ref = new_value;
    printf("Original: %d\n", original);
    printf("New Value: %d\n", new_value);
    printf("Reference: %d\n", original_ref);

    return 0;
}
