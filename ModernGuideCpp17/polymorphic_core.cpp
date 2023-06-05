/**
 * @file polymorphic_core.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>
#include <string_view>

class Base
{
public:
    Base()  = default;
    ~Base() = default;

    virtual void print_info() const = 0;
};

class Derive : public Base
{
public:
    Derive()  = default;
    ~Derive() = default;

    virtual void print_info() const
    {
        std::cout << "the derive from the abstract class because of pure virtual function\n";
    }
};

// Shape --> Rectangle --> Square with the draw() function
class Shape
{
public:
    Shape() = default;

    // ~Shape() = default;
    virtual ~Shape()
    {
        std::cout << "Shape deconstruction\n";
    }

    Shape(std::string_view name)
        : m_name(name)
    {
    }

    // void draw() const
    virtual void draw() const
    {
        std::cout << "Shape Drawing " << m_name << "\n";
    }

    virtual void draw(std::string_view color) const
    {
        std::cout << "Shape Drawing " << m_name << " color " << color << "\n";
    }

protected:
    std::string m_name;
};

class Rectangle : public Shape
{
public:
    Rectangle() = default;

    // ~Rectangle() = default;
    virtual ~Rectangle()
    {
        std::cout << "Rectangle deconstruction\n";
    }

    Rectangle(double x, double y, std::string_view name)
        : m_x(x)
        , m_y(y)
        , Shape(name)
    {
    }

    // void draw() const
    // virtual void draw() const override final
    virtual void draw() const override
    {
        std::cout << "Rectangle Drawing " << get_x() << " " << get_y() << " " << m_name << "\n";
    }

    // protected:
public:
    double get_x() const
    {
        return m_x;
    }

    double get_y() const
    {
        return m_y;
    }

private:
    double m_x{0.0};
    double m_y{0.0};
};

class Square : public Rectangle
{
public:
    Square() = default;

    // ~Square() = default;
    virtual ~Square()
    {
        std::cout << "Square deconstruction\n";
    }

    Square(double x, std::string_view name)
        : Rectangle(x, x, name)
    {
    }

    // void draw() const
    virtual void draw() const override final
    {
        std::cout << "Square Drawing " << m_name << " with x: " << get_x() << "\n";
    }
};

void draw_shape(Shape *s)
{
    s->draw();
}

/**
 * @brief the core feature of polymorphism in Modern C++.
 * polymorphism 指为不同数据类型的实体提供统一的接口, 意味着调用成员函数时, 会根据调用函数的
 * 对象的类型来执行不同的函数. 与此对应, 静态绑定指将名称绑定到一个固定的函数定义, 然后在每次
 * 调用该名称的时执行该定义, 也是常态执行的范式.
 * 
 * 多态时面向对象的核心, 面向对象是使用多态性获得堆系统中每个源代码依赖项的绝对控制的能力
 * 1. 多态与静态绑定
 * 2. 虚函数与动态绑定
 * 3. 多态对象的应用场景与大小
 * 4. override and final
 * 5. overloading 与 多态
 * 6. 析构函数与多态
 * 7. dynamic_cast 类型转换
 * 8. typeid 操作符
 * 9. 纯虚函数与抽象类
 * 10. 接口式抽象类
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------- 多态与静态绑定 --------
    // 静态绑定
    Shape s1("Shape1");
    s1.draw();

    Rectangle r1(1.0, 2.0, "Rectangle1");
    r1.draw();

    Square sq1(3.0, "Square1");
    sq1.draw();

    // -------- 虚函数与动态绑定 --------
    std::cout << "---------------------------------" << std::endl;
    // base pointers with non-virtual
    Shape *shape_ptr = &s1;
    shape_ptr->draw();
    shape_ptr = &r1;
    shape_ptr->draw();
    shape_ptr = &sq1;
    shape_ptr->draw();

    // -------- 多态对象的应用场景与大小 --------
    std::cout << "---------------------------------" << std::endl;
    Shape *base_ptr = &s1;
    draw_shape(base_ptr);
    base_ptr = &r1;
    draw_shape(base_ptr);
    base_ptr = &sq1;
    draw_shape(base_ptr);

    // collections
    std::cout << "---------------------------------" << std::endl;
    Shape shapes[]{s1, r1, sq1};
    for (Shape &s : shapes)
    {
        s.draw();
    }

    // Shape &shapes_ref[]{s1, r1, sq1}; /* error */
    Shape *shapes_ptr[]{&s1, &r1, &sq1};
    for (Shape *s : shapes_ptr)
    {
        s->draw();
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "sizeof(Shape): " << sizeof(Shape) << "\n";         /* s=32, d=40 */
    std::cout << "sizeof(Rectangle): " << sizeof(Rectangle) << "\n"; /* s=48, d=56 */
    std::cout << "sizeof(Square): " << sizeof(Square) << "\n";       /* s=48, d=56 */

    // -------- override and final --------
    // -------- overloading 与 多态 --------
    std::cout << "---------------------------------" << std::endl;
    Shape *base_shape_ptr = &r1;
    base_shape_ptr->draw();
    base_shape_ptr->draw("red");

    // -------- 析构函数与多态 --------
    std::cout << "---------------------------------" << std::endl;
    Square sq1_(34.0, "Square1_");

    // 析构函数需要时虚函数, 才能调用子类的析构函数释放资源(heap memory)
    Shape *ptr_heap = new Square(6.0, "square2");
    delete ptr_heap;

    // -------- dynamic_cast 类型转换 --------
    std::cout << "---------------------------------" << std::endl;
    // 通过 dynamic_cast 将父类指针转换为子类指针,便于调用子类的其他非虚函数
    Shape *ptr_shape_heap = new Rectangle(1.0, 6.0, "rectangle2");
    ptr_shape_heap->draw();
    // std::cout << ptr_shape_heap->get_x() << std::endl; /* error */
    Rectangle *ptr_rectangle_heap = dynamic_cast<Rectangle *>(ptr_shape_heap);
    std::cout << ptr_rectangle_heap->get_x() << std::endl;

    // delete ptr_shape_heap; /* runtime error, move */
    delete ptr_rectangle_heap;

    // -------- typeid 操作符 --------
    std::cout << "---------------------------------" << std::endl;
    // seen in the 'type_inference.cpp' file
    Rectangle rectangle_obj = Rectangle(1.0, 6.0, "rectangle2");

    Shape *ptr_type = &rectangle_obj;
    Shape &ref_type = rectangle_obj;

    std::cout << "typeid(ptr): " << typeid(ptr_type).name() << std::endl;
    std::cout << "typeid(ref): " << typeid(ref_type).name() << std::endl;
    std::cout << "typeid(*ptr): " << typeid(*ptr_type).name() << std::endl;

    // -------- 纯虚函数与抽象类 --------
    /* 纯虚函数:
        virtual double func() const = 0;
    子类必须重写该纯虚函数, 或者继续作为抽象类来使用;
    类中存在纯虚函数的类称之为抽象类;
    抽象类不能实例化的;
    ------------------------------------- */
    std::cout << "---------------------------------" << std::endl;
    // Base base_obj{}; /* error */
    Derive derive_obj{};
    derive_obj.print_info();
    
    // -------- 接口式抽象类 --------
    /* C++ 如何设计接口 | interface | API:
    C++ 不存在声明接口(interface)的关键字的;
    一个只有纯虚函数和没有成员变量的抽象类可以认为是OOP中的接口;
    --------------------------------------------------- */
    std::cout << "---------------------------------" << std::endl;

    return 0;
}
