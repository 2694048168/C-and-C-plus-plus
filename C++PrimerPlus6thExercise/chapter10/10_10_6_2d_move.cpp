/**
 * @file 10_10_6_2d_move.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

class Move
{
private:
    double x = 0.0;
    double y = 0.0;

public:
    Move(double a = 0, double b = 0)
        : x(a)
        , y(b)
    {
    }

    void show_move() const
    {
        std::cout << "The coordinate(X, Y): (" << this->x;
        std::cout << ", " << this->y << ")\n";
        std::cout << "----------------------------------\n";
    }

    Move add(const Move &m) const
    {
        Move obj;

        obj.x = m.x + this->x;
        obj.y = m.y + this->y;

        return obj;
    }

    void reset(double a = 0, double b = 0)
    {
        this->x = a;
        this->y = b;
    }
};

/**
 * @brief 编写C++程序, 二维坐标的移动
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Move point2d;
    point2d.show_move();

    Move point2d_2(2.3, 4.5);
    point2d_2.show_move();

    Move point2d_3 = point2d.add(point2d_2);
    point2d_3.show_move();

    Move point2d_4 = point2d_2.add(point2d_3);
    point2d_4.show_move();

    point2d_3.reset();
    point2d_3.show_move();

    return 0;
}