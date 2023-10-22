/**
 * @file 08_special_members.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// https://cplusplus.com/doc/tutorial/classes2/
// the special member functions of class that are implicitly defined
// as member of classes under certain circumstances:
// -------------------------------------------------------------
// 1. Default constructor	Class::Class();
// 2. Destructor	        Class::~Class();
// 3. Copy constructor	    Class::Class (const Class&);
// 4. Copy assignment	    Class& operator= (const Class&);
// 5. Move constructor	    Class::Class (Class&&);
// 6. Move assignment	    Class& operator= (Class&&);
// -------------------------------------------------------------
// explicitly members exist with their default definition
// or which are deleted by using the keywords default and delete, respectively
// function_declaration = default;
// function_declaration = delete;
// -------------------------------------------------

#include <iostream>
#include <string>

class ExampleClass
{
public:
    // special member function
    ExampleClass(int val)
        : total(val)
    {
    }

public:
    void accumulate(int x)
    {
        total += x;
        std::cout << "the value of data member: " << total << "\n";
    }

private:
    int total;
};

class ExampleClass2
{
public:
    // special member function
    ExampleClass2()
        : ptr_str(new std::string)
    {
    }

    ExampleClass2(const std::string &str)
        : ptr_str(new std::string(str))
    {
    }

    // using custom copy constructor that performs a deep copy,
    // not the default copy constructor that performs shallow copy
    ExampleClass2(const ExampleClass2 &obj)
        : ptr_str(new std::string(obj.content()))
    {
    }

    ExampleClass2 &operator=(const ExampleClass2 &obj)
    {
        // delete ptr_str;
        // ptr_str = new std::string(obj.content());

        *ptr_str = obj.content();

        return *this;
    }

    ExampleClass2(ExampleClass2 &&obj)
        : ptr_str(obj.ptr_str)
    {
        // obj ownership move transferred
        obj.ptr_str = nullptr;
    }

    ExampleClass2 &operator=(ExampleClass2 &&obj)
    {
        delete ptr_str;
        ptr_str     = obj.ptr_str;
        obj.ptr_str = nullptr;

        return *this;
    }

    ~ExampleClass2()
    {
        // RAII tech. and cleanup
        delete ptr_str;
    }

public:
    const std::string &content() const
    {
        return *ptr_str;
    }

    ExampleClass2 operator+(const ExampleClass2 &rhs)
    {
        return ExampleClass2(content() + rhs.content());
    }

private:
    std::string *ptr_str;
};

class Rectangle
{
public:
    // special member function
    Rectangle(int width, int height)
        : width(width)
        , height(height)
    {
    }

    Rectangle() = default;

    Rectangle(const Rectangle &other) = delete;

public:
    int area() const
    {
        return width * height;
    }

    void set_width(int width)
    {
        this->width = width;
    }

    int get_width() const
    {
        return this->width;
    }

    int get_height() const
    {
        return this->height;
    }

    void set_height(int height)
    {
        this->height = height;
    }

private:
    int width;
    int height;
};

// -----------------------------
int main(int argc, char **argv)
{
    // ExampleClass example_obj; /* Error, non-default constructor */
    ExampleClass example_obj{42};
    example_obj.accumulate(24);

    ExampleClass2 example_obj2{"hello Cpp world\n"};
    // ExampleClass2 example_obj2;
    std::cout << "the content: " << example_obj2.content();

    ExampleClass2 example_obj3(example_obj2);
    std::cout << "the content(copy constructor): " << example_obj3.content();

    ExampleClass2 example_obj4{"Wei Li\n"};
    example_obj3 = example_obj4;
    std::cout << "the content(copy assignment): " << example_obj3.content();

    // ------------------------------
    ExampleClass2 foo("Exam");
    ExampleClass2 bar = ExampleClass2("ple for move semantics"); /* move-construction */

    foo = foo + bar; /* move-assignment */

    std::cout << "foo's content: " << foo.content() << '\n';

    // ----------------------
    Rectangle foo_obj;
    Rectangle bar_obj(10, 20);

    std::cout << "bar's area: " << bar_obj.area() << '\n';

    return 0;
}
