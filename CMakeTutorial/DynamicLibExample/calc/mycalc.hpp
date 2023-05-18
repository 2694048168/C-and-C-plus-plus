/**
 * @file mycalc.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-18
 * @version 0.1.1
 * @copyright Copyright (c) 2023

 * @brief the header file for dynamic library.
 * @attention Windows and Linux loadding dynamic library by different ways.
 */

#ifndef __MYCALC_HPP
#define __MYCALC_HPP 

#ifdef _WIN32
    // Windows MSVC '__declspec' 特有的动态库装载方式
    #ifdef MYMATH_API_EXPORT
        #define MYMATHLIB_API __declspec(dllexport) /* generate dynamic library */
    #else
        #define MYMATHLIB_API __declspec(dllimport) /* using dynamic library */
    #endif // MYMATH_API_EXPORT`
#elif __linux__
    #define MYMATHLIB_API
#elif __APPLE__
    #define MYMATHLIB_API
#endif // _WIN32

extern "C" MYMATHLIB_API int myadd(int integer_1, int integer_2);
extern "C" MYMATHLIB_API int mysub(int integer_1, int integer_2);
extern "C" MYMATHLIB_API int mydiv(int integer_1, int integer_2);
extern "C" MYMATHLIB_API int mymul(int integer_1, int integer_2);

#endif // !__MYCALC_HPP