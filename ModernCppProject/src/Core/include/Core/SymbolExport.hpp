/**
 * @file SymbolExport.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 动态库符号导出
 * @version 0.1
 * @date 2026-04-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

// 调用约定
#define CALLING_CONVENTIONS __cdecl // __stdcall

// 标准纯粹 C++ 代码, 处理动态库符号导出
#ifdef _WIN32
// Windows MSVC '__declspec' 特有的动态库装载方式
#    ifdef LIB_API_EXPORT
#        define LIB_API __declspec(dllexport) /* generate dynamic library */
#    else
#        define LIB_API __declspec(dllimport) /* using dynamic library */
#    endif
#elif __linux__
#    define LIB_API
#elif __APPLE__
#    define LIB_API
#endif // _WIN32

// 标准 C++ & QT 代码, 处理动态库符号导出
#ifndef BUILD_STATIC
#    ifdef LIB_API_WIDGET_EXPORT
#        define LIB_API_WIDGET Q_DECL_EXPORT
#    else
#        define LIB_API_WIDGET Q_DECL_IMPORT
#    endif
#endif

#if defined(QT_LIBRARY)
#  define QT_EXPORT Q_DECL_EXPORT
#else
#  define QT_EXPORT Q_DECL_IMPORT
#endif
