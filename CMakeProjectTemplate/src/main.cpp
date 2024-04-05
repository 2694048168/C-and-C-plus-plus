#include "MathLib/Math.hpp"
#include "PrintModuleLib/Module.hpp"

// Eigen头文件，<Eigen/Dense>包含Eigen库里面所有的函数和类
#include <Eigen/Dense>
#include <iostream>

// --------------------------
int main(int argc, const char **argv)
{
    PrintModule *p_print = PrintModule::getInstance();
    p_print->printInfo("Welcome to the Modern C++ and CMake");

    MyMath math_obj;

    int num1 = 12;
    int num2 = 24;
    int res  = 0;
    math_obj.addNumer(num1, num2, res);
    p_print->printDebug("the sum is ", res);

    int num3 = 12;
    int num4 = 4;
    int res1 = 0;
    if (math_obj.mulNumer(num3, num4, res1))
        p_print->printDebug("the mul is ", res1);

    // 测试使用 Eigen3 三方库
    p_print->printInfo("Eigen3 simple example");
    // MatrixXd 表示的是动态数组，初始化的时候指定数组的行数和列数
    Eigen::MatrixXd mat_(2, 2);
    //mat(i,j) 表示第i行第j列的值，这里对数组进行初始化
    mat_(0, 0) = 3;
    mat_(1, 0) = 2.5;
    mat_(0, 1) = -1;
    mat_(1, 1) = mat_(1, 0) + mat_(0, 1);
    std::cout << mat_ << std::endl; // eigen重载了<<运算符，可以直接输出eigen矩阵的值

    return 0;
}
