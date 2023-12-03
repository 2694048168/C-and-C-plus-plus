#include "DataStruct.h"

/**
 * @brief 自定义图像数据结构体
 * 构造函数进行初始化, 析构函数进行释放资源(特别是指针申请的内存), RAII技术思想
 */
ImgData::ImgData()
{
    pImgBuf    = nullptr;
    nImgWidth  = 0;
    nImgHeight = 0;
}

ImgData::~ImgData()
{
    // 全局变量或全局对象的析构时机, 需要仔细考虑
    if (pImgBuf)
    {
        delete pImgBuf;
        pImgBuf = nullptr;
    }
}
