// 每个头文件都需要的，保证头文件只被包含一次
#ifndef __ADD_H__
#define __ADD_H__

// 如果使用 C++ 编译器，则将头文件中所有声明包含在 extern "C" 中
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// body of header
int add(int x, int y);

#ifdef __cplusplus
} // closing brace for extern "C"
#endif // __cplusplus

#endif // __ADD_H__