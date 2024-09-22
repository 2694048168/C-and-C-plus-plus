#ifndef __CUSTOM_CMATH_H__
#define __CUSTOM_CMATH_H__

#ifdef _WIN32

// Windows MSVC '__declspec' 特有的动态库装载方式
#    ifdef CUSTOM_MATH_API_EXPORT
#        define CUSTOM_MATH_API __declspec(dllexport) /* generate dynamic library */
#    else
#        define CUSTOM_MATH_API __declspec(dllimport) /* using dynamic library */
#    endif

#elif __linux__
#    define CUSTOM_MATH_API

#elif __APPLE__
#    define CUSTOM_MATH_API
#endif // _WIN32

#define __callback__ __stdcall
// #define __callback __cdecl

#ifdef __cplusplus
extern "C"
{
#endif

int CUSTOM_MATH_API __callback__ custom_add(int x, int y);
int CUSTOM_MATH_API __callback__ custom_sub(int x, int y);
int CUSTOM_MATH_API __callback__ custom_mul(int x, int y);
int CUSTOM_MATH_API __callback__ custom_div(int x, int y);

#ifdef __cplusplus
}
#endif

#endif /* __CUSTOM_CMATH_H__ */
