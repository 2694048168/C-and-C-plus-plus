/**
 * @file 02_placementOperator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>
#include <new>

/**
* @brief 6. 布置运算符 Placement Operators
* 有时并不希望重写所有自由存储区分配, 此时可以使用布置(Placement)运算符,
* 它们对预分配内存执行适当的初始化.
* 1. void* operator new(size_t, void*);
* 2. void operator delete(size_t, void*);
* 3. void* operator new[](void*, void*);
* 4. void operator delete[](void*, void*);
*  
* 使用布置运算符可在任意内存中手动构造对象,
* 好处是可以手动管理对象的生命周期, 坏处是不能使用 delete 来释放生成的动态对象
* 必须直接调用对象的析构函数(且只能调用一次).
* 
*/
struct Point
{
    Point()
        : x{}
        , y{}
        , z{}
    {
        printf("Point at %p constructed.\n", this);
    }

    ~Point()
    {
        printf("Point at %p destructed.\n", this);
    }

    double x, y, z;
};

// ------------------------------------
int main(int argc, const char **argv)
{
    const auto point_size = sizeof(Point);
    std::byte  data[3 * point_size];
    printf("Data starts at %p.\n", data);

    auto point1 = new (&data[0 * point_size]) Point{};
    auto point2 = new (&data[1 * point_size]) Point{};
    auto point3 = new (&data[2 * point_size]) Point{};

    point1->~Point();
    point2->~Point();
    point3->~Point();

    // 观察到每个 placement-new 操作都使用 data 数组占用的内存来分配 Point,
    // 必须分别调用每个析构函数来销毁相应对象.

    return 0;
}
