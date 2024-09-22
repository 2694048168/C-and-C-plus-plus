#ifndef __CUSTOM_CPP_MATH_H__
#define __CUSTOM_CPP_MATH_H__

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

// 使用 C 结构体来包装 C++ 类的实例对象
typedef struct CustomMathWrapper *pCustomMathWrapper;
typedef struct CustomMathWrapper  CustomMathWrapper;

// 构造函数
pCustomMathWrapper CUSTOM_MATH_API __callback__ CustomMath_create();

// 析构函数
void CUSTOM_MATH_API __callback__ CustomMath_destroy(CustomMathWrapper *wrapper);

// 公有函数
int CUSTOM_MATH_API __callback__ CustomMath_add(CustomMathWrapper *wrapper, int x, int y);
int CUSTOM_MATH_API __callback__ CustomMath_sub(CustomMathWrapper *wrapper, int x, int y);
int CUSTOM_MATH_API __callback__ CustomMath_mul(CustomMathWrapper *wrapper, int x, int y);
int CUSTOM_MATH_API __callback__ CustomMath_div(CustomMathWrapper *wrapper, int x, int y);

#ifdef __cplusplus
}
#endif

#endif /* __CUSTOM_CPP_MATH_H__ */
