/**
 * @file Math.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 简单自定义封装一个数学处理库
 * @version 0.1
 * @date 2024-04-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __MATH_HPP__

#    ifdef _WIN32
// Windows MSVC '__declspec' 特有的动态库装载方式
#        ifdef MYMATH_API_EXPORT
#            define MYMATHLIB_API __declspec(dllexport) /* generate dynamic library */
#        else
#            define MYMATHLIB_API __declspec(dllimport) /* using dynamic library */
#        endif                                          // MYMATH_API_EXPORT
#    elif __linux__
#        define MYMATHLIB_API
#    elif __APPLE__
#        define MYMATHLIB_API
#    endif // _WIN32

class MYMATHLIB_API MyMath
{
public:
    MyMath()  = default;
    ~MyMath() = default;

    bool addNumer(const int &val1, const int &val2, int &res);
    bool subNumer(const int &val1, const int &val2, int &res);
    bool mulNumer(const int &val1, const int &val2, int &res);
    bool divNumer(const int &val1, const int &val2, int &res);
};

#endif // !__MATH_HPP__