#pragma once

//定义符号(symbols)的输入和输出
#define SAVE_RENDER_IMAGE_EXPORT
#ifdef SAVE_RENDER_IMAGE_EXPORT
#    define SAVE_RENDER_IMAGE_API __declspec(dllexport)
#else
#    define SAVE_RENDER_IMAGE_API __declspec(dllimport)
#endif

#include "RenderGraphics.hpp"
#include "opencv2/opencv.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

class SAVE_RENDER_IMAGE_API SaveRenderImage
{
public:
    /* 设置渲染图像储存路径
    * @param folderPath: 系统储存渲染结果图像的文件夹路径
    * @note: supported the Chinese path, but not supported Special characters such as spaces,
    */
    //static void SetSaveFolderPath(const std::string& folderPath);
    virtual void SetSaveFolderPath(const std::string &folderPath) = 0;

    /* 添加待绘制渲染的文本信息
    * @position{ 500, 500 }: the Starting coordinate position in the upper left corner, unit is pixel.
    * @color: the render color of the text font. see the enum 'RenderColor'.
    * @text: the Actual rendered text string.
    * @thickness: the rendered font size. Set rendering through freetype, unit is pixel.
    */
    virtual void AddText(std::vector<RenderGraphicsSave::TextInfo> textInfo) = 0;
    /* UI界面渲染坐标位置(x, y)[屏幕显示坐标] 和图像保存渲染位置的不同(图像像素坐标)
    */
    virtual void SetTextCoordinateScale(int scale_coordinate) = 0;

    /* 添加待绘制渲染的直线信息
    * @start_point(x, y): the start 2D point of the line, unit is pixel.
    * @end_point(x, y): the end 2D point of the line, unit is pixel.
    * @color: the render color of the line. see the enum 'RenderColor'
    * @thickness: the thickness of the line, unit is pixel.
    */
    virtual void AddLine(std::vector<RenderGraphicsSave::LineInfo> lineInfo) = 0;

    /* 添加待绘制渲染的矩形信息
    * @roi_rectangle{ 2500, 200, 400, 500 }: the (top_left_x, top_left_y, width, height) of the ROI rectangle, unit is pixel.
    * @color: the render color of the Wireframe for a rectangle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a rectangle, unit is pixel.
    */
    virtual void AddRect(std::vector<RenderGraphicsSave::RectInfo> rectInfo) = 0;

    /* 添加待绘制渲染的圆形信息
    * @center_point{x, y}: the center point coordinate of 2D circle, unit is pixel.
    * @radius: the radius of rendered 2D circle, unit is pixel.
    * @color: the render color of the Wireframe for a circle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a circle, unit is pixel.
    */
    virtual void AddCircle(std::vector<RenderGraphicsSave::CircleInfo> circlesInfo) = 0;

    /* 清除所以绘制渲染的信息
    * @note: the line, rectangle, text, circle...
    */
    virtual void ClearAllGraphics() = 0;

    /* 添加当前绘制渲染的图像
    * @image: the rendering image to draw line, rectangle, text, circle...
    * @pData: the data buffer of image byte-stream data.
    * @step: step – Number of bytes each matrix row occupies.
    * @note: the overloading to cover different image format(such as cv::Mat).
    */
    virtual void SetCurrentImage(cv::Mat &image) = 0;
    //virtual void SetCurrentImage(cv::Mat image) = 0;
    virtual void SetCurrentImage(void *pData, int height, int width, size_t step, int type = CV_8UC1) = 0;

    /* 保存绘制渲染的结果图
    * @param filepath: 保存渲染结果图像的文件路径
    * @param isSaveBinary: 是否已二进制数据流进行保存图像数据, default=false
    * if the isSaveBinary==true, recommended mark the W_H via file suffixes.
    * @scale_factor: 0 < scale < 1, downsample image; scale > 1, upsample image; default=1.0, NO scale image.
    * @note: the filepath is NOT filename
    */
    //virtual void SaveRenderResultImage(std::string& filepath, bool isSaveBinary=false) = 0;
    virtual void SaveRenderResultImage(std::string filepath, double scale_factor = 1.0, bool isSaveBinary = false) = 0;

    /* 从保存绘制渲染的结果图的二进制数据流文件中加载并保存为图像格式
    * @param filepath: 保存渲染结果图像的二进制数据流文件路径
    * @param format: 保存的图像格式, default=JPEG, this option is Reserved.
    * @param savepath: 保存图像路径, default=filepath, just different for suffixes.
    * @note: if NOT the same 'Instance', must to be the image shape(C, H, W), default=3.
    * @note: the filepath is NOT filename
    */
    virtual void SaveBinaryToImage(std::string &filepath)                                          = 0;
    virtual void SaveBinaryToImage(std::string &filepath, unsigned int width, unsigned int height) = 0;
};

/** @brief Create SaveRenderImage Instance
The function createSaveRenderImage create instance to render graphics into image.
The supported graphics see in 'RenderGraphics.hpp'
*/
SAVE_RENDER_IMAGE_API std::shared_ptr<SaveRenderImage> createSaveRenderImage();
