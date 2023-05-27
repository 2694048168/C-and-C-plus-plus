/**
 * @file this_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

class Test
{
private:
    int random_seed{42};

public:
    void setX(int random_seed)
    {
        // The 'this' pointer is used to retrieve the object's random_seed
        // hidden by the local variable 'random_seed'
        this->random_seed = random_seed;
    }

    void print()
    {
        std::cout << "random_seed = " << random_seed << "\n";
    }
};

class Points2D
{
public:
    Points2D(int x = 0, int y = 0)
    {
        this->x = x;
        this->y = y;
    }

    /* All calls modify the same object as the same object 
    is returned by reference */
    Points2D &setX(int x)
    {
        this->x = x;
        return *this;
    }

    Points2D &setY(int y)
    {
        this->y = y;
        return *this;
    }

    void print() const
    {
        std::cout << "x = " << x << ", y= " << y << std::endl;
    }

private:
    int x{};
    int y{};
};

/**
 * @brief 'this' pointer in C++
 * 1. When local variable’s name is same as member’s name,
 *      such as for constructors, initializer list can also be used
 *      when parameter name is same as member’s name.
 * 
 * 2. To return reference to the calling object,
 *      When a reference to a local object is returned, the returned reference 
 *      can be used to chain function calls on a single object. 
 * 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* --------------- Step 1. --------------- */
    int random_seed = 20;

    Test obj;
    obj.setX(random_seed);
    obj.print();

    /* --------------- Step 2. --------------- */
    Points2D obj_point2d(6, 6);
    obj_point2d.print();
    // Chained function calls.
    obj_point2d.setX(42).setY(24);
    obj_point2d.print();

    return 0;
}