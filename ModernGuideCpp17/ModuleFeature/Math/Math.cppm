/**
 * @file Math.cppm
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

// 预处理器宏，用于控制导入/导出
#ifdef MATH_EXPORTS
#    define MATH_API __declspec(dllexport)
#else
#    define MATH_API __declspec(dllimport)
#endif

export module MathModule;

namespace MathModule {
export MATH_API int add(int a, int b);
export MATH_API int sub(int a, int b);
export MATH_API int div(int a, int b);
export MATH_API int mul(int a, int b);

export class MATH_API MathDemo
{
public:
    MathDemo();
    ~MathDemo();

    int Add(int a, int b);
    int Sub(int a, int b);
    int Div(int a, int b);
    int Mul(int a, int b);
};
} // namespace MathModule
