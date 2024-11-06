#include "RenderImageSave.h"

SaveRenderImage::SaveRenderImage()
    : m_folderPath{""}
{
    m_textRender.setFontSize(100);

    //m_compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    //m_compression_params.push_back(0); // 0表示无损保存

    //https://docs.opencv.org/4.10.0/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
    m_compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    m_compression_params.push_back(50); // quality from 0 to 100 (the higher is the better). Default value is 95
}

SaveRenderImage::~SaveRenderImage()
{
    ClearAllGraphics();
}

void SaveRenderImage::SetSaveFolderPath(const std::string &folderPath)
{
    m_folderPath = folderPath;
}

void SaveRenderImage::AddText(std::vector<TextInfo> textInfo)
{
    m_textsInfo = textInfo;
    textInfo.clear();
}

void SaveRenderImage::AddLine(std::vector<LineInfo> lineInfo)
{
    m_linesInfo = lineInfo;
    lineInfo.clear();
}

void SaveRenderImage::AddTect(std::vector<RectInfo> rectInfo)
{
    m_rectsInfo = rectInfo;
    rectInfo.clear();
}

void SaveRenderImage::AddCircle(std::vector<CircleInfo> circlesInfo)
{
    m_circlesInfo = circlesInfo;
    circlesInfo.clear();
}

void SaveRenderImage::ClearAllGraphics()
{
    m_textsInfo.clear();
    m_linesInfo.clear();
    m_rectsInfo.clear();
    m_circlesInfo.clear();
}

void SaveRenderImage::SetCurrentImage(cv::Mat &image)
{
    m_image = image;
}

//void SaveRenderImage::SetCurrentImage(cv::Mat image)
//{
//    m_image = image;
//}

void SaveRenderImage::SetCurrentImage(void *pData, int height, int width, size_t step, int type)
{
    m_image = cv::Mat(height, width, type, pData, step);
}

void SaveRenderImage::SaveRenderResultImage(std::string &filepath)
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
    if (m_rectsInfo.empty())
    {
        // Not to render the text
    }
    else
    {
        for (const auto &text_info : m_textsInfo)
        {
            m_textRender.setFontSize(text_info.text_thickness);
            m_textRender.putTextExt(m_image, text_info.text.c_str(), CvPoint{text_info.x, text_info.y},
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
    cv::imwrite(filepath, m_image, m_compression_params);
}

cv::Scalar SaveRenderImage::getRenderColor(RenderColor color)
{
    switch (color)
    {
    case RenderColor::RED:
        return RENDER_COLOR_RED;
        break;
    case RenderColor::GREEN:
        return RENDER_COLOR_GREEN;
        break;
    case RenderColor::BLUE:
        return RENDER_COLOR_BLUE;
        break;
    case RenderColor::YELLOW:
        return RENDER_COLOR_YELLOW;
        break;
    case RenderColor::BLACK:
        return RENDER_COLOR_BLACK;
        break;
    case RenderColor::WHITE:
        return RENDER_COLOR_WHITE;
        break;
    default:
        return RENDER_COLOR_RED;
        break;
    }
}
