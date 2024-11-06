#include "CvxText.h"
#include "RenderImageSave.h"

#include <filesystem>
#include <fstream>

class SAVE_RENDER_IMAGE_API SaveRenderImageImpl : public SaveRenderImage
{
public:
    SaveRenderImageImpl();
    ~SaveRenderImageImpl();

    /* 设置渲染图像储存路径
    * @param folderPath: 系统储存渲染结果图像的文件夹路径
    * @note: supported the Chinese path, but not supported Special characters such as spaces,
    */
    //static void SetSaveFolderPath(const std::string& folderPath);
    void SetSaveFolderPath(const std::string &folderPath) override;

    /* 添加待绘制渲染的文本信息
    * @position{ 500, 500 }: the Starting coordinate position in the upper left corner, unit is pixel.
    * @color: the render color of the text font. see the enum 'RenderColor'.
    * @text: the Actual rendered text string.
    * @thickness: the rendered font size. Set rendering through freetype, unit is pixel.
    */
    void AddText(std::vector<RenderGraphicsSave::TextInfo> textInfo) override;
    /* UI界面渲染坐标位置(x, y)[屏幕显示坐标] 和图像保存渲染位置的不同(图像像素坐标)
    */
    void SetTextCoordinateScale(int scale_coordinate) override;

    /* 添加待绘制渲染的直线信息
    * @start_point(x, y): the start 2D point of the line, unit is pixel.
    * @end_point(x, y): the end 2D point of the line, unit is pixel.
    * @color: the render color of the line. see the enum 'RenderColor'
    * @thickness: the thickness of the line, unit is pixel.
    */
    void AddLine(std::vector<RenderGraphicsSave::LineInfo> lineInfo) override;

    /* 添加待绘制渲染的矩形信息
    * @roi_rectangle{ 2500, 200, 400, 500 }: the (top_left_x, top_left_y, width, height) of the ROI rectangle, unit is pixel.
    * @color: the render color of the Wireframe for a rectangle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a rectangle, unit is pixel.
    */
    void AddRect(std::vector<RenderGraphicsSave::RectInfo> rectInfo) override;

    /* 添加待绘制渲染的圆形信息
    * @center_point{x, y}: the center point coordinate of 2D circle, unit is pixel.
    * @radius: the radius of rendered 2D circle, unit is pixel.
    * @color: the render color of the Wireframe for a circle. see the enum 'RenderColor'
    * @thickness: the thickness of the wireframe for a circle, unit is pixel.
    */
    void AddCircle(std::vector<RenderGraphicsSave::CircleInfo> circlesInfo) override;

    /* 清除所以绘制渲染的信息
    * @note: the line, rectangle, text, circle...
    */
    void ClearAllGraphics() override;

    /* 添加当前绘制渲染的图像
    * @image: the rendering image to draw line, rectangle, text, circle...
    * @pData: the data buffer of image byte-stream data.
    * @step: step – Number of bytes each matrix row occupies.
    * @note: the overloading to cover different image format(such as cv::Mat).
    */
    void SetCurrentImage(cv::Mat &image) override;
    //void SetCurrentImage(cv::Mat image);
    void SetCurrentImage(void *pData, int height, int width, size_t step, int type = CV_8UC1) override;

    /* 保存绘制渲染的结果图
    * @param filepath: 保存渲染结果图像的文件路径
    * @param isSaveBinary: 是否已二进制数据流进行保存图像数据, default=false
    * if the isSaveBinary==true, recommended mark the W_H via file suffixes.
    * @note: the filepath is NOT filename
    */
    //void SaveRenderResultImage(std::string& filepath, bool isSaveBinary = false) override;
    void SaveRenderResultImage(std::string filepath, double scale_factor = 1.0, bool isSaveBinary = false) override;

    /* 从保存绘制渲染的结果图的二进制数据流文件中加载并保存为图像格式
    * @param filepath: 保存渲染结果图像的二进制数据流文件路径
    * @param format: 保存的图像格式, default=JPEG, this option is Reserved.
    * @param savepath: 保存图像路径, default=filepath, just different for suffixes.
    * @note: if not the same 'Instance', must to be the image shape(C, H, W), default=3.
    * @note: the filepath is NOT filename
    */
    void SaveBinaryToImage(std::string &filepath) override;
    void SaveBinaryToImage(std::string &filepath, unsigned int width, unsigned int height) override;

    // NOTE: gray-image using colormap via applycolormap?
    // https://blog.csdn.net/qq_44799835/article/details/122535696

private:
    cv::Scalar getRenderColor(RenderGraphicsSave::RenderColor color);

private:
    //static std::string m_folderPath;
    std::string m_folderPath;

    std::vector<RenderGraphicsSave::TextInfo>   m_textsInfo;   //绘制文字
    std::vector<RenderGraphicsSave::LineInfo>   m_linesInfo;   //绘制直线
    std::vector<RenderGraphicsSave::RectInfo>   m_rectsInfo;   //绘制矩形线框
    std::vector<RenderGraphicsSave::CircleInfo> m_circlesInfo; //绘制圆形线框

    cv::Mat          m_image;
    cv::Mat          m_scale_image;
    CvxText          m_textRender = CvxText(R"(C:\Windows\Fonts\simhei.ttf)");
    std::vector<int> m_compression_params;
    int              m_text_scale_coordinate;
};

SaveRenderImageImpl::SaveRenderImageImpl()
    : m_folderPath{""}
    , m_text_scale_coordinate{10}
{
    m_textRender.setFontSize(100);

    //m_compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    //m_compression_params.push_back(0); // 0表示无损保存

    //https://docs.opencv.org/4.10.0/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
    m_compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    m_compression_params.push_back(50); // quality from 0 to 100 (the higher is the better). Default value is 95
}

SaveRenderImageImpl::~SaveRenderImageImpl()
{
    SaveRenderImageImpl::ClearAllGraphics();
}

void SaveRenderImageImpl::SetSaveFolderPath(const std::string &folderPath)
{
    m_folderPath = folderPath;
}

void SaveRenderImageImpl::AddText(std::vector<RenderGrphicsSave::TextInfo> textInfo)
{
    m_textsInfo = textInfo;
    textInfo.clear();
}

void SaveRenderImageImpl::SetTextCoordinateScale(int scale_coordinate)
{
    m_text_scale_coordinate = scale_coordinate;
}

void SaveRenderImageImpl::AddLine(std::vector<RenderGrphicsSave::LineInfo> lineInfo)
{
    m_linesInfo = lineInfo;
    lineInfo.clear();
}

void SaveRenderImageImpl::AddRect(std::vector<RenderGrphicsSave::RectInfo> rectInfo)
{
    m_rectsInfo = rectInfo;
    rectInfo.clear();
}

void SaveRenderImageImpl::AddCircle(std::vector<RenderGrphicsSave::CircleInfo> circlesInfo)
{
    m_circlesInfo = circlesInfo;
    circlesInfo.clear();
}

void SaveRenderImageImpl::ClearAllGraphics()
{
    m_textsInfo.clear();
    m_linesInfo.clear();
    m_rectsInfo.clear();
    m_circlesInfo.clear();
}

void SaveRenderImageImpl::SetCurrentImage(cv::Mat &image)
{
    m_image = image;
}

//void SaveRenderImageImpl::SetCurrentImage(cv::Mat image)
//{
//    m_image = image;
//}

void SaveRenderImageImpl::SetCurrentImage(void *pData, int height, int width, size_t step, int type)
{
    m_image = cv::Mat(height, width, type, pData, step);
}

//
//void SaveRenderImageImpl::SaveRenderResultImage(std::string& filepath, bool isSaveBinary)
//{
//    // ========= params checks =========
//    if (m_image.empty())
//        return;
//    else
//    {
//        // render image must be a color image(three channels)
//        if (CV_8UC1 == m_image.type())
//            cv::cvtColor(m_image, m_image, cv::COLOR_GRAY2RGB);
//    }
//
//    // ========= draw the line info =========
//    if (m_linesInfo.empty())
//    {
//        // Not to render the line
//    }
//    else
//    {
//        for (const auto& line_info : m_linesInfo)
//        {
//            cv::line(m_image, cv::Point{ line_info.start_x, line_info.start_y },
//                cv::Point{ line_info.end_x, line_info.end_y }, getRenderColor(line_info.color), line_info.line_thickness);
//        }
//    }
//
//    // ========= draw the rectangle info =========
//    if (m_rectsInfo.empty())
//    {
//        // Not to render the rectangle
//    }
//    else
//    {
//        for (const auto& rect_info : m_rectsInfo)
//        {
//            //cv::Rect roi_rectangle{ 2500, 200, 400, 500 }; // top_left_x, top_left_y, width, height
//            cv::rectangle(m_image, cv::Rect{ rect_info.x, rect_info.y, rect_info.w, rect_info.h },
//                getRenderColor(rect_info.color), rect_info.rect_thickness);
//        }
//    }
//
//    // ========= draw the text info =========
//    if (m_rectsInfo.empty())
//    {
//        // Not to render the text
//    }
//    else
//    {
//        for (const auto& text_info : m_textsInfo)
//        {
//            m_textRender.setFontSize(text_info.text_thickness);
//            m_textRender.putTextExt(m_image, text_info.text.c_str(),
//                CvPoint{ text_info.x, text_info.y }, getRenderColor(text_info.color));
//        }
//    }
//
//    // ========= draw the circle info =========
//    if (m_circlesInfo.empty())
//    {
//        // Not to render the circle
//    }
//    else
//    {
//        for (const auto& circle_info : m_circlesInfo)
//        {
//            cv::circle(m_image, cv::Point{ circle_info.center_x, circle_info.center_y },
//                circle_info.radius, getRenderColor(circle_info.color), circle_info.circle_thickness);
//        }
//    }
//
//    // ========= save the render image into disk =========
//    // ========= step 1. IO-speed? =========
//    // ========= step 2. compression? =========
//    // ========= step 3. binary file or image format? =========
//    /* cv::imwrite函数的进阶技巧
//    * 1 性能优化-优化图像写入速度, cv::imwrite函数的写入速度可以通过以下方法优化：
//    * 1.1 使用无损压缩格式：无损压缩格式（如PNG、TIFF）虽然文件体积更大，但写入速度更快? ---> NG
//    * 1.2 减少图像尺寸：较小的图像写入速度更快? ---> OK
//    * 1.3 使用并行写入：对于大型图像，可以使用并行写入来提高写入速度;
//    * 2 性能优化-使用并行写入, 并行写入可以通过以下方式实现：
//    * 2.1 使用多线程：创建多个线程并行写入图像的不同部分;
//    * 2.2 使用多进程：创建多个进程并行写入图像的不同部分;
//    */
//
//    if (!isSaveBinary)
//    {
//        cv::imwrite(filepath, m_image, m_compression_params);
//    }
//    else
//    {
//        // 大量高速写入 ---> 二进制形式
//        std::vector<uint8_t> frame;
//        cv::imencode(".jpg", m_image, frame, m_compression_params);
//        std::ofstream outimgfile;
//        outimgfile.open(filepath.c_str(), std::ios_base::out | std::ios::binary);
//        //outimgfile.open(filepath.c_str(), std::ios::binary);
//        outimgfile.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(uint8_t));
//        outimgfile.close();
//    }
//
//}

void SaveRenderImageImpl::SaveRenderResultImage(std::string filepath, double scale_factor, bool isSaveBinary)
{
    // ========= params checks =========
    if (m_image.empty())
        return;
    else
    {
        // render image must be a color image(three channels)
        if (CV_8UC1 == m_image.type())
            cv::cvtColor(m_image, m_image, cv::COLOR_GRAY2RGB);
    }

    // ========= draw the line info =========
    if (m_linesInfo.empty())
    {
        // Not to render the line
    }
    else
    {
        for (const auto &line_info : m_linesInfo)
        {
            cv::line(m_image, cv::Point{line_info.start_x, line_info.start_y},
                     cv::Point{line_info.end_x, line_info.end_y}, getRenderColor(line_info.color),
                     line_info.line_thickness);
        }
    }

    // ========= draw the rectangle info =========
    if (m_rectsInfo.empty())
    {
        // Not to render the rectangle
    }
    else
    {
        for (const auto &rect_info : m_rectsInfo)
        {
            //cv::Rect roi_rectangle{ 2500, 200, 400, 500 }; // top_left_x, top_left_y, width, height
            cv::rectangle(m_image, cv::Rect{rect_info.x, rect_info.y, rect_info.w, rect_info.h},
                          getRenderColor(rect_info.color), rect_info.rect_thickness);
        }
    }

    // ========= draw the text info =========
    if (m_textsInfo.empty())
    {
        // Not to render the text
    }
    else
    {
        for (const auto &text_info : m_textsInfo)
        {
            //m_textRender.setFontSize(text_info.text_thickness);
            m_textRender.putTextExt(
                m_image, text_info.text.c_str(),
                CvPoint{text_info.x * m_text_scale_coordinate, text_info.y * m_text_scale_coordinate},
                getRenderColor(text_info.color));
        }
    }

    // ========= draw the circle info =========
    if (m_circlesInfo.empty())
    {
        // Not to render the circle
    }
    else
    {
        for (const auto &circle_info : m_circlesInfo)
        {
            cv::circle(m_image, cv::Point{circle_info.center_x, circle_info.center_y}, circle_info.radius,
                       getRenderColor(circle_info.color), circle_info.circle_thickness);
        }
    }

    // ========= save the render image into disk =========
    // ========= step 1. IO-speed? =========
    // ========= step 2. compression? =========
    // ========= step 3. binary file or image format? =========
    /* cv::imwrite 函数性能优化-优化图像写入速度, cv::imwrite 函数的写入速度可以通过以下方法优化：
    * 1.1 使用无损压缩格式：无损压缩格式（如PNG、TIFF）虽然文件体积更大，但写入速度更快? ---> NG
    * 1.2 减少图像尺寸：较小的图像写入速度更快? ---> OK
    * 1.3 使用并行写入：对于大型图像，可以使用并行写入来提高写入速度;
    * @note: Use Multi-Threading for Parallel Processing.
    */

    cv::Size dsize = cv::Size(m_image.cols * scale_factor, m_image.rows * scale_factor);
    m_scale_image  = cv::Mat(dsize, m_image.type());
    /*
    * 速度比较：INTER_NEAREST（最近邻插值)>INTER_LINEAR(线性插值)>INTER_CUBIC(三次样条插值)>INTER_AREA  (区域插值);
    * ----对图像进行缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法;
    * ----OpenCV推荐：如果要缩小图像，通常推荐使用#INTER_AREA插值效果最好;
    * ----而要放大图像，通常使用INTER_CUBIC(速度较慢，但效果最好)，或者使用INTER_LINEAR(速度较快，效果还可以);
    * ----至于最近邻插值INTER_NEAREST，一般不推荐使用;
    */
    cv::resize(m_image, m_scale_image, dsize, scale_factor, scale_factor, cv::INTER_NEAREST);

    if (!isSaveBinary)
    {
        cv::imwrite(filepath, m_scale_image, m_compression_params);
    }
    else
    {
        // 大量高速写入 ---> 二进制形式
        std::vector<uint8_t> frame;
        cv::imencode(".jpg", m_scale_image, frame, m_compression_params);
        std::ofstream outimgfile;
        auto          __filepathRender__ = std::filesystem::path(filepath);
        //auto str1 = __filepathRender__.root_name(); // "D:"
        //auto str2 = __filepathRender__.root_directory(); // "\\"
        //auto str3 = __filepathRender__.root_path(); // "D:\\"
        //auto str4 = __filepathRender__.relative_path(); // "images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3_Render_0.bin"
        //auto str5 = __filepathRender__.parent_path(); // "D:\\images\图片文件整理\4621\相机2瑕疵前相机内侧-2"
        //auto str6 = __filepathRender__.filename(); // 091106-234_3_Render_0.bin
        //auto str7 = __filepathRender__.stem(); // 091106-234_3_Render_0
        //auto str8 = __filepathRender__.extension(); // .bin
        char          buf[256];
        sprintf_s(buf, "_W%d_H%d_", m_scale_image.cols, m_scale_image.rows);
        auto filename     = __filepathRender__.stem().concat(buf);
        auto __filepath__ = __filepathRender__.parent_path()
                                .append(filename.string())
                                .concat(__filepathRender__.extension().string());
        //outimgfile.open(filepath.c_str(), std::ios_base::out | std::ios::binary);
        outimgfile.open(__filepath__.string(), std::ios_base::out | std::ios::binary);
        outimgfile.write(reinterpret_cast<const char *>(frame.data()), frame.size() * sizeof(uint8_t));
        /*
        cv::parallel_for_(cv::Range(0, m_scale_image.rows), [&](const cv::Range& range) {
            for (int idx_row = range.start; idx_row < range.end; ++idx_row)
            {
                outimgfile.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(uint8_t));
            }
            });
        */

        outimgfile.close();
    }
}

void SaveRenderImageImpl::SaveBinaryToImage(std::string &filepath)
{
    // 大量读取 ---> 二进制形式
    std::ifstream inimgfile;
    size_t        size_vec           = m_scale_image.rows * m_scale_image.cols;
    auto          __filepathRender__ = std::filesystem::path(filepath);
    __filepathRender__.replace_extension("jpeg");

    inimgfile.open(filepath, std::ios_base::in | std::ios::binary);
    std::vector<uint8_t> p_data(size_vec);
    inimgfile.read(reinterpret_cast<char *>(p_data.data()), size_vec * sizeof(uint8_t));
    cv::Mat img_save = cv::imdecode(p_data, 1);
    cv::imwrite(__filepathRender__.string(), img_save);
    inimgfile.close();
}

void SaveRenderImageImpl::SaveBinaryToImage(std::string &filepath, unsigned int width, unsigned int height)
{
    // 大量读取 ---> 二进制形式
    std::ifstream inimgfile;
    size_t        size_vec           = width * height;
    auto          __filepathRender__ = std::filesystem::path(filepath);
    __filepathRender__.replace_extension("jpeg");

    inimgfile.open(filepath, std::ios_base::in | std::ios::binary);
    std::vector<uint8_t> p_data(size_vec);
    inimgfile.read(reinterpret_cast<char *>(p_data.data()), size_vec * sizeof(uint8_t));
    cv::Mat img_save = cv::imdecode(p_data, 1);
    cv::imwrite(__filepathRender__.string(), img_save);
    inimgfile.close();
}

cv::Scalar SaveRenderImageImpl::getRenderColor(RenderGrphicsSave::RenderColor color)
{
    switch (color)
    {
    case RenderGraphicsSave::RenderColor::RED:
        return cv::Scalar RENDER_COLOR_RED;
        break;
    case RenderGraphicsSave::RenderColor::GREEN:
        return cv::Scalar RENDER_COLOR_GREEN;
        break;
    case RenderGraphicsSave::RenderColor::BLUE:
        return cv::Scalar RENDER_COLOR_BLUE;
        break;
    case RenderGraphicsSave::RenderColor::YELLOW:
        return cv::Scalar RENDER_COLOR_YELLOW;
        break;
    case RenderGraphicsSave::RenderColor::BLACK:
        return cv::Scalar RENDER_COLOR_BLACK;
        break;
    case RenderGraphicsSave::RenderColor::WHITE:
        return cv::Scalar RENDER_COLOR_WHITE;
        break;
    default:
        return cv::Scalar RENDER_COLOR_RED;
        break;
    }
}

SAVE_RENDER_IMAGE_API std::shared_ptr<SaveRenderImage> createSaveRenderImage()
{
    return std::shared_ptr<SaveRenderImageImpl>{new SaveRenderImageImpl()};
}
