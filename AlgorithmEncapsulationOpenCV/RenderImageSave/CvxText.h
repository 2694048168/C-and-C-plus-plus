/**
 * @file CvxText.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief OpenCV汉字输出接口, 实现了汉字输出功能.
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * OpenCV 4.X 使用CvxText在图片显示汉字
 * https://bigbookplus.github.io/blog/2022-06-15-cvxtext-opencv45.html
 * 
 * https://github.com/liuxiaodongzl/opencv/blob/master/CvxText.h
 * 
 * OpenCV渲染中文字符
 * https://blog.mangoeffect.net/opencv/opencv-puttext-chinese-characters
 * 
 * 
 * 
 */

#pragma once

#include <ft2build.h>
#include FT_FREETYPE_H
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"

/**
* \class CvxText
* \brief OpenCV中输出汉字
*
* OpenCV中输出汉字。字库提取采用了开源的FreeType库。由于FreeType是
* GPL版权发布的库，和OpenCV版权并不一致，因此目前还没有合并到OpenCV扩展库中.
*
* 显示汉字的时候需要一个汉字字库文件，字库文件系统一般都自带了。
* 这里采用的是一个开源的字库：“文泉驿正黑体”。
*
* 关于"OpenCV扩展库"的细节请访问
* http://code.google.com/p/opencv-extension-library/
*
* 关于FreeType的细节请访问
* http://www.freetype.org/
*
*/

class CvxText
{
    // 禁止copy
    CvxText &operator=(const CvxText &);

public:
    // 装载字库文件
    CvxText(const char *freeType);
    virtual ~CvxText();

    /**
    * 获取字体。目前有些参数尚不支持。
    *
    * \param font        字体类型, 目前不支持
    * \param size        字体大小/空白比例/间隔比例/旋转角度
    * \param underline   下画线
    * \param diaphaneity 透明度
    *
    * \sa setFont, restoreFont
    */
    void getFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);

    /**
    * 设置字体。目前有些参数尚不支持。
    *
    * \param font        字体类型, 目前不支持
    * \param size        字体大小/空白比例/间隔比例/旋转角度
    * \param underline   下画线
    * \param diaphaneity 透明度
    *
    * \sa getFont, restoreFont
    */
    void setFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
    void setFontSize(int size);

    /**
    * 恢复原始的字体设置。
    *
    * \sa getFont, setFont
    */
    void restoreFont();

    /**
    * 输出汉字(颜色默认为黑色); 遇到不能输出的字符将停止。
    *
    * \param img   输出的影象
    * \param text  文本内容
    * \param pos   文本位置
    * \param color 文本颜色
    *
    * \return 返回成功输出的字符长度，失败返回-1。
    */
    int putTextExt(IplImage* img, const char* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    //int putTextExt(cv::Mat& frame, const char* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    int putTextExt(cv::Mat frame, const char* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    int putTextExt(cv::Mat frame, const char* text, CvPoint pos, cv::Scalar color = { 255, 255, 255 });

    /**
    * 输出汉字(颜色默认为黑色); 遇到不能输出的字符将停止。
    *
    * \param img   输出的影象
    * \param text  文本内容
    * \param pos   文本位置
    * \param color 文本颜色
    *
    * \return 返回成功输出的字符长度，失败返回-1。
    */
    int putTextExt(IplImage* img, const wchar_t* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    //int putTextExt(cv::Mat& frame, const wchar_t* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    int putTextExt(cv::Mat frame, const wchar_t* text, CvPoint pos, CvScalar color = { 255, 255, 255 });
    int putTextExt(cv::Mat frame, const wchar_t* text, CvPoint pos, cv::Scalar color = { 255, 255, 255 });

private:
    // 输出当前字符, 更新m_pos位置
    void putWChar(IplImage* img, wchar_t wc, CvPoint& pos, CvScalar color, int fontSize=12);
    void putWChar(IplImage* img, wchar_t wc, CvPoint& pos, cv::Scalar color, int fontSize = 12);

private:
    FT_Library m_library; // 字库
    FT_Face    m_face;    // 字体

    // 默认的字体输出参数
    int      m_fontType;
    CvScalar m_fontSize;
    int      m_size;
    bool     m_fontUnderline;
    float    m_fontDiaphaneity;
};
