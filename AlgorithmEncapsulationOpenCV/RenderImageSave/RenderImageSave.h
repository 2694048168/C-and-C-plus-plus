/**
 * @file RenderImageSave.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "CvxText.h"
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>

#define RENDER_COLOR_RED    cv::Scalar{0, 0, 255}
#define RENDER_COLOR_GREEN  cv::Scalar{0, 255, 0}
#define RENDER_COLOR_BLUE   cv::Scalar{255, 0, 0}
#define RENDER_COLOR_YELLOW cv::Scalar{0, 255, 255}
#define RENDER_COLOR_BLACK  cv::Scalar{0, 0, 0}
#define RENDER_COLOR_WHITE  cv::Scalar{255, 255, 255}

enum class RenderColor : char
{
    RED    = 0,
    GREEN  = 1,
    BLUE   = 2,
    YELLOW = 3,
    BLACK  = 4,
    WHITE  = 5,

    NUM_COLOR
};

struct TextInfo
{
    int         x;
    int         y;
    std::string text;
    RenderColor color          = RenderColor::RED; // default color
    int         text_thickness = 100;              // unit: pixel
};

struct LineInfo
{
    int         start_x;
    int         start_y;
    int         end_x;
    int         end_y;
    RenderColor color          = RenderColor::RED; // default color
    int         line_thickness = 12;               // unit: pixel
};

struct RectInfo
{
    int         x;
    int         y;
    int         w;
    int         h;
    RenderColor color          = RenderColor::RED; // default color
    int         rect_thickness = 12;               // unit: pixel
};

struct CircleInfo
{
    int         center_x;
    int         center_y;
    int         radius;
    RenderColor color            = RenderColor::RED; // default color
    int         circle_thickness = 8;                // unit: pixel
};

class SaveRenderImage
{
public:
    SaveRenderImage();
    ~SaveRenderImage();

    /* 设置渲染图像储存路径
    * @param folderPath: 系统储存渲染结果图像的文件夹路径
    * @note: supported the Chinese path, but not supported Special characters such as spaces,
    */
    //static void SetSaveFolderPath(const std::string& folderPath);
    void SetSaveFolderPath(const std::string &folderPath);

    /* 添加待绘制渲染的文本信息
    * @position{ 500, 500 }: the Starting coordinate position in the upper left corner, unit is pixel.
    * @color: the render color of the text font. see the enum 'RenderColor'.
    * @text: the Actual rendered text string.
    * @thickness: the rendered font size. Set rendering through freetype, unit is pixel.
    */
    void AddText(std::vector<TextInfo> textInfo);

    /* 添加待绘制渲染的直线信息
    * @start_point(x, y): the start 2D point of the line, unit is pixel.
    * @end_point(x, y): the end 2D point of the line, unit is pixel.
    * @color: the render color of the line. see the enum 'RenderColor'
    * @thickness: the thickness of the line, unit is pixle.
    */
    void AddLine(std::vector<LineInfo> lineInfo);

    /* 添加待绘制渲染的矩形信息
    * @roi_rectangle{ 2500, 200, 400, 500 }: the (top_left_x, top_left_y, width, height) of the ROI rectangle, unit is pixel.
    * @color: the render color of the Wireframe for a rectangle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a rectangle, unit is pixle.
    */
    void AddTect(std::vector<RectInfo> rectInfo);

    /* 添加待绘制渲染的圆形信息
    * @center_point{x, y}: the center point coordinate of 2D circle, unit is pixel.
    * @radius: the radius of rendered 2D circle, unit is pixel.
    * @color: the render color of the Wireframe for a circle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a circle, unit is pixle.
    */
    void AddCircle(std::vector<CircleInfo> circlesInfo);

    /* 清除所以绘制渲染的信息
    * @note: the line, rectangle, text, circle...
    */
    void ClearAllGraphics();

    /* 添加当前绘制渲染的图像
    * @image: the rendering image to draw line, rectangle, text, circle...
    * @pData: the data buffer of image byte-stream data
    * @step: step – Number of bytes each matrix row occupies.
    * @note: the overloadding to cover different image format(such as cv::Mat)
    */
    void SetCurrentImage(cv::Mat &image);
    //void SetCurrentImage(cv::Mat image);
    void SetCurrentImage(void *pData, int height, int width, size_t step, int type = CV_8UC1);

    /* 保存绘制渲染的结果图
    * @param filepath: 保存渲染结果图像的文件路径
    * @note: the filepath is NOT filename
    */
    void SaveRenderResultImage(std::string &filepath);

    // NOTE: gray-image using colormap via applycolormap?
    // https://blog.csdn.net/qq_44799835/article/details/122535696

private:
    cv::Scalar getRenderColor(RenderColor color);

private:
    //static std::string m_folderPath;
    std::string m_folderPath;

    std::vector<TextInfo>   m_textsInfo;   //绘制文字
    std::vector<LineInfo>   m_linesInfo;   //绘制直线
    std::vector<RectInfo>   m_rectsInfo;   //绘制矩形线框
    std::vector<CircleInfo> m_circlesInfo; //绘制圆形线框

    cv::Mat          m_image;
    CvxText          m_textRender = CvxText(R"(C:\Windows\Fonts\simhei.ttf)");
    std::vector<int> m_compression_params;
};
