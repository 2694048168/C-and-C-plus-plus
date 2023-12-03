#pragma once

#include <vector>
#include "DataStruct.h"

// 模拟程序中对全局变量的修改, 
// 多线程可以导致数据竞争问题, 需要注意对全局变量和全局对象做加锁处理
void addOnce(const int& value);


bool addMapTable(const char* key, std::vector<ImgData> value);