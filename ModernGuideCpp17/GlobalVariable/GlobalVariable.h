#ifndef __GLOBAL_VARIABLE_H__
#define __GLOBAL_VARIABLE_H__

/**
 * @file GlobalVariable.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 全局变量和全局对象的声明, 可以在整个C++程序中直接使用
 *        (linking的时候可以被找到符号symbol)
 * @version 0.1
 * @date 2023-12-03
 * 
 * https://github.com/2694048168
 * @copyright Copyright (c) 2023
 * 
 */

#include "DataStruct.h"

#include <map>
#include <vector>


/**
 * @brief 测试全局变量的声明
 * 
 */
extern int num;

/**
 * @brief 测试全局对象的声明
 * 
 */
extern std::map<const char *, std::vector<ImgData>> mapTable;

#endif /* __GLOBAL_VARIABLE_H__ */