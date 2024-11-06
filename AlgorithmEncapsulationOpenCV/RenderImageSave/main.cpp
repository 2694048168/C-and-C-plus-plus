#include "RenderImageSave.h"
#include "opencv2/opencv.hpp"

#include <fstream>
#include <numeric>
#include <string>

//------------------------------------
int main(int argc, const char **argv)
{
    /* ------------------------------------ */
    std::ofstream fout;
    const char   *log_filepath = "time_result.txt";
    fout.open(log_filepath);
    if (fout.fail())
    {
        std::cout << "open the " << log_filepath << " failed\n";
    }
    /* ------------------------------------ */
    SaveRenderImage save_render_image;
    std::string     filepath = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3.jpeg)";
    cv::Mat         image    = cv::imread(filepath, 0);

    // ============================================
    std::vector<TextInfo> textsInfo;
    TextInfo              textInfo;
    textInfo.x     = 500;
    textInfo.y     = 500;
    textInfo.text  = "极耳宽度: 35.15 mm";
    textInfo.color = RenderColor::GREEN;
    textsInfo.emplace_back(textInfo);
    textInfo.y     = 600;
    textInfo.text  = "极耳高度: 15.20 mm";
    textInfo.color = RenderColor::GREEN;
    textsInfo.emplace_back(textInfo);
    textInfo.y     = 700;
    textInfo.text  = "极耳间距: 45.75 mm";
    textInfo.color = RenderColor::GREEN;
    textsInfo.emplace_back(textInfo);
    textInfo.y     = 800;
    textInfo.text  = "极耳宽度: 35.15 mm";
    textInfo.color = RenderColor::GREEN;
    textsInfo.emplace_back(textInfo);
    textInfo.y     = 900;
    textInfo.text  = "极耳高度: 15.20 mm";
    textInfo.color = RenderColor::RED;
    textsInfo.emplace_back(textInfo);
    textInfo.y     = 1000;
    textInfo.text  = "当前帧结果判定: OK";
    textInfo.color = RenderColor::YELLOW;
    textsInfo.emplace_back(textInfo);

    // ============================================
    std::vector<LineInfo> linesInfo;
    LineInfo              lineInfo;
    lineInfo.start_x = 2870;
    lineInfo.start_y = 10;
    lineInfo.end_x   = 2870;
    lineInfo.end_y   = 1880;
    lineInfo.color   = RenderColor::GREEN;
    linesInfo.emplace_back(lineInfo);
    lineInfo.start_x = 13960;
    lineInfo.start_y = 10;
    lineInfo.end_x   = 13960;
    lineInfo.end_y   = 1880;
    lineInfo.color   = RenderColor::RED;
    linesInfo.emplace_back(lineInfo);

    // ============================================
    std::vector<RectInfo> rectsInfo;
    RectInfo              rectInfo;
    rectInfo.x     = 2500;
    rectInfo.y     = 200;
    rectInfo.w     = 400;
    rectInfo.h     = 500;
    rectInfo.color = RenderColor::GREEN;
    rectsInfo.emplace_back(rectInfo);
    rectInfo.x     = 13950;
    rectInfo.y     = 160;
    rectInfo.w     = 400;
    rectInfo.h     = 500;
    rectInfo.color = RenderColor::RED;
    rectsInfo.emplace_back(rectInfo);
    // ============================================
    std::vector<CircleInfo> circlesInfo;
    CircleInfo              circleInfo;
    circleInfo.center_x = 12100;
    circleInfo.center_y = 750;
    circleInfo.radius   = 100;
    circleInfo.color    = RenderColor::BLUE;
    circlesInfo.emplace_back(circleInfo);
    // ============================================
    std::vector<double> save_vec;
    const int           num_repeate = 100;
    for (int idx{0}; idx < num_repeate; ++idx)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        save_render_image.ClearAllGraphics();
        save_render_image.AddLine(linesInfo);
        save_render_image.AddTect(rectsInfo);
        save_render_image.AddText(textsInfo);
        save_render_image.AddCircle(circlesInfo);
        save_render_image.SetCurrentImage(image);
        std::string filepathRender = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3_Render.jpeg)";
        //char buf[256];
        //sprintf_s(buf, "091106-234_3_Render_%d.jpeg", idx);
        //std::string filepathRender = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\)" + std::string(buf);
        save_render_image.SaveRenderResultImage(filepathRender);

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        fout << "[== " << ++idx << " ======Draw and save the render image after]time comsumption: " << duration.count()
             << " ms\n";
        save_vec.emplace_back(duration.count());
    }

    auto   mean     = std::accumulate(save_vec.cbegin(), save_vec.cend(), 0.0) / save_vec.size();
    double variance = 0.0;
    for (const auto &elem : save_vec)
    {
        variance += std::pow(elem - mean, 2);
    }
    variance /= save_vec.size();
    auto standard_deviation = std::sqrt(variance);

    fout << "\n[========Save render image after]time comsumption AVERAGE: " << mean << " ms\n";
    fout << "\n[========Save render image after]time comsumption MAx: "
         << *std::max_element(save_vec.cbegin(), save_vec.cend()) << " ms\n";
    fout << "\n[========Save render image after]time comsumption Min: "
         << *std::min_element(save_vec.cbegin(), save_vec.cend()) << " ms\n";
    fout << "\n[========Save render image after]time comsumption VAE: " << variance << " ms\n";
    fout << "\n[========Save render image after]time comsumption STD: " << standard_deviation << " ms\n";
    fout.close();

    return 0;
}
