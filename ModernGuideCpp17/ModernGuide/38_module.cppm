/**
 * @file 38_module.cppm
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

// module.cppm
export module math; // 声明正在定义一个名为 "math" 的模块

// 希望外部世界能调用这个函数，所以用 export 关键字导出它
export int add(int a, int b)
{
    return a + b;
}

// 只导出函数的声明
export int sub(int a, int b)
{
    return a - b;
}
