/**
 * @file main_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "CvxText.h"
#include "utility.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <numeric>
#include <string>

// =====================================
int main(int argc, const char **argv)
{
    std::string filepath = R"(D:\images\相机2瑕疵前相机内侧-2\091106-234_3.jpeg)";
    //cv::Mat image = cv::imread(filepath, 0);
    cv::Mat     image = cv::imread(filepath);

    /* ------------------------------------ */
    std::ofstream fout;
    const char   *log_filepath = "time_result.txt";
    fout.open(log_filepath);
    if (fout.fail())
    {
        std::cout << "open the " << log_filepath << " failed\n";
    }
    /* ------------------------------------ */

    const int           num_repeate = 100;
    std::vector<double> line_vec;
    std::vector<double> rect_vec;
    std::vector<double> text_vec;
    std::vector<double> save_vec;
    for (int idx{0}; idx < num_repeate; ++idx)
    {
        auto       startTime = std::chrono::high_resolution_clock::now();
        // ============ line ============
        cv::Point  start_point{2870, 10};
        cv::Point  end_point{2870, 1880};
        cv::Scalar line_color{0, 255, 0};
        int        line_thickness = 12; // unit: pixel
        /*
        * cv::LineTypes {
      cv::FILLED = -1 ,
      cv::LINE_4 = 4 ,
      cv::LINE_8 = 8 ,
      cv::LINE_AA = 16
        */
        int        line_type = 4; // default==8,
        cv::line(image, start_point, end_point, line_color, line_thickness, line_type);

        cv::Point start_point_{13960, 10};
        cv::Point end_point_{13960, 1880};
        cv::line(image, start_point_, end_point_, line_color, line_thickness);
        // ======================================================================
        auto endTime_line  = std::chrono::high_resolution_clock::now();
        auto duration_line = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_line - startTime);
        fout << "[========Draw lines after]time comsumption: " << duration_line.count() << " ms\n";
        line_vec.emplace_back(duration_line.count());
#ifdef _DEBUG
        std::cout << "[========Draw lines after]time comsumption: " << duration_line.count() << " ms\n";
#endif // _DEBUG

        // ============ segment ============
        //cv::polylines(image, );
        // ======================================================================

        // ============ rectangle ============
        cv::Rect roi_rectangle{2500, 200, 400, 500}; // top_left_x, top_left_y, width, height
        cv::rectangle(image, roi_rectangle, line_color, line_thickness);

        cv::Rect roi_rectangle_{13950, 160, 400, 500}; // top_left_x, top_left_y, width, height
        cv::rectangle(image, roi_rectangle_, line_color, line_thickness);
        // ======================================================================
        auto endTime_rectangle  = std::chrono::high_resolution_clock::now();
        auto duration_rectangle = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_rectangle - startTime);
        fout << "[========Draw rectangles after]time comsumption: " << duration_rectangle.count() << " ms\n";
        rect_vec.emplace_back(duration_rectangle.count());
#ifdef _DEBUG
        std::cout << "[========Draw rectangles after]time comsumption: " << duration_rectangle.count() << " ms\n";
#endif // _DEBUG

        // ============ rectangle ============
        //const std::string text_msg = "极耳宽度: 35.15 mm";
        //const std::string text_msg = "极耳宽度: 35.15 mm\n极耳高度: 15.20 mm\n极耳间距: 45.75 mm";
        //const std::string text_msg = u8"极耳宽度: 35.15 mm";
        cv::Point left_bottom{500, 500}; // cv::getTextSize() to compute the Position coordinates
        //int fontFace = cv::FONT_HERSHEY_COMPLEX;
        int       fontFace = cv::FONT_HERSHEY_SIMPLEX;
        //auto fontFace = cv::fontFace("SimHei");
        //double text_scale = 6.0;
        int       fontSize = 120; // unit: pixel

        // ------------ Step 1 ------------
        //cv::putText(image, text_msg, left_bottom, fontFace, text_scale, line_color);

        // ------------ Step 2 ------------
        //PutTextExt(image, text_msg.c_str(), left_bottom, line_color, fontSize);
        //PutTextExt(image, text_msg.c_str(), left_bottom, line_color, fontSize, "微软雅黑");
        //PutTextExt(image, text_msg.c_str(), left_bottom, line_color, fontSize, "楷体");

        // ------------ Step 3 ------------
        const std::string text_msg1 = "极耳宽度: 35.15 mm";
        const std::string text_msg2 = "极耳高度: 15.20 mm";
        const std::string text_msg3 = "极耳间距: 45.75 mm";
        const std::string text_msg4 = "极耳宽度: 35.15 mm";
        const std::string text_msg5 = "极耳高度: 15.20 mm";
        const std::string text_msg6 = "极耳间距: 45.75 mm";

        CvxText textRender(R"(C:\Windows\Fonts\simhei.ttf)");
        textRender.setFontSize(100);
        //CvxText textRender(R"(C:\Windows\Fonts\arialbd.ttf)");
        CvScalar color_{0, 255, 0};
        //CvPoint position_{ 500, 500 };
        CvPoint  position1{500, 500};
        CvPoint  position2{500, 600};
        CvPoint  position3{500, 700};
        CvPoint  position4{500, 800};
        CvPoint  position5{500, 900};
        CvPoint  position6{500, 1000};
        //auto img = cvIplImage(image);
        //textRender.putTextExt(&img, text_msg.c_str(), position_, color_);
        //textRender.putTextExt(image, text_msg.c_str(), position_, color_);
        textRender.putTextExt(image, text_msg1.c_str(), position1, color_);
        textRender.putTextExt(image, text_msg2.c_str(), position2, color_);
        textRender.putTextExt(image, text_msg3.c_str(), position3, color_);
        textRender.putTextExt(image, text_msg4.c_str(), position4, color_);
        textRender.putTextExt(image, text_msg5.c_str(), position5, color_);
        textRender.putTextExt(image, text_msg6.c_str(), position6, color_);
        // ======================================================================
        auto endTime_text  = std::chrono::high_resolution_clock::now();
        auto duration_text = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_text - startTime);
        fout << "[========Draw Texts after]time comsumption: " << duration_text.count() << " ms\n";
        text_vec.emplace_back(duration_text.count());
#ifdef _DEBUG
        std::cout << "[========Draw Texts after]time comsumption: " << duration_text.count() << " ms\n";
#endif // _DEBUG

        // ======================================================================
        // 大量写入 ---> 二进制形式
        std::vector<uint8_t> frame;
        std::vector<int>     compress_params;
        int                  quality = 50; // 0~100
        compress_params.emplace_back(cv::IMWRITE_JPEG_QUALITY);
        compress_params.emplace_back(quality);
        cv::imencode(".jpg", image, frame, compress_params);
        std::string filepathRenderBinary
            = R"(D:\images\相机2瑕疵前相机内侧-2\091106-234_3_lineRender.bin)";
        std::ofstream outimgfile;
        //outimgfile.open(filepathRenderBinary.c_str(), std::ios_base::out | std::ios::binary);
        outimgfile.open(filepathRenderBinary.c_str(), std::ios::binary);
        //for (int r = 0; r < image.rows; r++)
        //outimgfile.write(reinterpret_cast<const char*>(image.ptr(r)), image.cols * image.elemSize());
        outimgfile.write(reinterpret_cast<const char *>(frame.data()), frame.size() * sizeof(uint8_t));
        outimgfile.close();

        //std::vector<int> compression_params;
        //compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        //compression_params.push_back(0); // 0表示无损保存
        // https://blog.csdn.net/dageda1991/article/details/77926927
        // https://blog.csdn.net/Dontla/article/details/102701956#:~:text=1%E3%80%81%E4%BD%BF%E7%94%A8opencv
        //compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        //compression_params.push_back(0); // 0表示无损保存
        //std::string filepathRender = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3_lineRender.jpeg)";
        //std::string filepathRender = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3_lineRender.png)";

        // applyColorMap（） 伪彩色函数
        //cv::Mat image_;
        //cv::applyColorMap(image, image_, cv::ColormapTypes::COLORMAP_AUTUMN);
        //cv::imwrite(filepathRender, image_, compression_params);
        //cv::imwrite(filepathRender, image, compression_params);

        //cv::Mat image_merge;
        //std::vector<cv::Mat> src_colorImgs{ image.clone(), image.clone(), image.clone() };
        //cv::merge(src_colorImgs, image_merge);
        //cv::imwrite(filepathRender, image_merge, compression_params);

        //cv::imwrite(filepathRender, image, compression_params);
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        fout << "[========Save render image after]time comsumption: " << duration.count() << " ms\n";
        save_vec.emplace_back(duration.count());
#ifdef _DEBUG
        std::cout << "[========Save render image after]time comsumption: " << duration.count() << " ms\n";
#endif // _DEBUG
    }

    fout << "\n[========Draw lines after]AVERAGE time comsumption: "
         << std::accumulate(line_vec.begin(), line_vec.end(), 0.0) / line_vec.size() << " ms\n";
    fout << "[========Draw rectangles after]AVERAGE time comsumption: "
         << std::accumulate(rect_vec.begin(), rect_vec.end(), 0.0) / rect_vec.size() << " ms\n";
    fout << "[========Draw Texts after]AVERAGE time comsumption: "
         << std::accumulate(text_vec.begin(), text_vec.end(), 0.0) / text_vec.size() << " ms\n";
    fout << "[========Save render image after]AVERAGE time comsumption: "
         << std::accumulate(save_vec.begin(), save_vec.end(), 0.0) / save_vec.size() << " ms\n";
    fout.close();

    // 大量读取 ---> 二进制形式
    //std::ifstream inimgfile;
    //inimgfile.open(filepathRenderBinary, std::ios_base::in | std::ios::binary);
    //inimgfile.open(filepathRenderBinary, std::ios::binary);
    //cv::Mat img_save = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    //for (int r{0}; r < img_save.rows; ++r)
    //inimgfile.read(reinterpret_cast<char*>(img_save.ptr(r)), img_save.cols * img_save.elemSize());
    //std::vector<uint8_t> p_data(frame.size());
    //inimgfile.read(reinterpret_cast<char*>(p_data.data()), frame.size() * sizeof(uint8_t));
    //cv::Mat img_save = cv::imdecode(p_data, 1);
    //std::string filepathRender__ = R"(D:\images\图片文件整理\4621\相机2瑕疵前相机内侧-2\091106-234_3_lineRender.jpeg)";
    //cv::imwrite(filepathRender__, img_save);
    //inimgfile.close();

    return 0;
}
