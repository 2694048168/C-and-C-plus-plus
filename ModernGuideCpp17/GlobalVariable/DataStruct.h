#ifndef __DATA_STRUCT_H__
#define __DATA_STRUCT_H__

/**
 * @file DataStruct.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 程序的所有自定义的数据结构体
 * @version 0.1
 * @date 2023-12-03
 * 
 * https://github.com/2694048168
 * @copyright Copyright (c) 2023
 * 
 */


/**
 * @brief 自定义图像数据结构体
 * 
 */
struct ImgData
{
    unsigned char* pImgBuf; /* 图像源数据地址指针 */
    unsigned int nImgWidth; /* 图像帧的宽度 W */
    unsigned int nImgHeight; /* 图像帧的高度 H */

    // 构造函数
    ImgData();
    // 析构函数
    ~ImgData();
};

#endif /* __DATA_STRUCT_H__ */