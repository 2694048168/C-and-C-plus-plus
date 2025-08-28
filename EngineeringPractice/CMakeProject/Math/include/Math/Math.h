/**
 * @file Math.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-28
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#ifndef Module_EXPORT
#    if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(API_EXPORT)
#        define Module_EXPORT __declspec(dllexport)
#    elif (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#        define Module_EXPORT __declspec(dllimport)
#    elif defined __GNUC__ && __GNUC__ >= 4 && (defined(API_EXPORT) || defined(__APPLE__))
#        define Module_EXPORT __attribute__((visibility("default")))
#    endif
#endif

#define CALLING_CONVENTIONS __cdecl // __stdcall

namespace Math {

class Module_EXPORT MathExample
{
public:
    int Add(int a, int b);
    int Sub(int a, int b);
    int Div(int a, int b);
    int Mul(int a, int b);

public:
    MathExample()  = default;
    ~MathExample() = default;
};

} // namespace Math