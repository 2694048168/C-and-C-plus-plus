/**
 * @file 06_genericPointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdint>
#include <cstdio>

/**
  * @brief C++ 多级指针和泛型指针
  * 变量(对象): 类型 和 值;
  * *指针(引用类型): 解类型(解引用后的类型) 和 地址;
  * 指针也可以不指定解类型(解类型为空) ---> 泛型指针
  * 
  */

// ----------------------------------
int main(int argc, const char **argv)
{
    int   val = 4;
    void *ptr = &val; // 不指定解类型, 解类型为空, 泛型指针

    // 必须要先指定解类型才能进行指针的解引用操作
    printf("[INFO]the value: %d\n", *static_cast<int *>(ptr));
    *static_cast<int *>(ptr) = 42;
    printf("[INFO]the value: %d\n", *static_cast<int *>(ptr));

    // ======== 多级指针 ========
    /**
     * 对象(变量) | 类型       | 解类型
     * num      | uint16_t   | ----
     * ptr1     | uint16_t*  | uint16_t
     * ptr2     | uint16_t** | uint16_t*
     */
    uint16_t   num  = 24;
    uint16_t  *ptr1 = &num;
    uint16_t **ptr2 = &ptr1;

    printf("[====INFO num]the value: %d\n", num);
    printf("[====INFO ptr1]the value: %d\n", *ptr1);
    printf("[====INFO ptr2]the value: %d\n", **ptr2);

    return 0;
}
