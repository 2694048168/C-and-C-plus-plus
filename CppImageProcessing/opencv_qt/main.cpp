#include "utility.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

#include <QApplication>
#include <QString>
#include <QPainter>
#include <QFont>
#include <QPen>

bool ImageSaver(const QString& path, QImage& image)
{
    QPainter p;
    if (!p.begin(&image))
        return false;

    //p.setPen(QPen(Qt::red));
    p.setPen(QPen(Qt::green));
    p.setFont(QFont("Times", 28, QFont::Bold));
    p.drawText(image.rect(), Qt::AlignCenter, "Width: 35.21");
    p.end();

    return image.save(path);
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    // this should allow `QApplication` to complete its initialization
    QApplication::processEvents();

    std::cout << "======== Image Convert between OpenCV and QImage ========\n\n";
    const std::string single_filename = "images/period.jpg";
    const std::string color_filename = "images/baboon.jpg";
    const QString single_filename_qt = "images/period.jpg";
    const QString color_filename_qt = "images/baboon.jpg";

    std::cout << "======== Convert OpenCV into QImage ========\n";
    // -------- the single channel image(grayscale image)
    cv::Mat single_img_mat = cv::imread(single_filename);
    //QImage single_img_qt = WeiLi::Utility::mat_to_qimage_cpy(single_img_mat);
    QImage single_img_qt = WeiLi::Utility::mat_to_qimage_ref(single_img_mat);
    single_img_qt.save("images/period_mat2qt.jpg");

    single_img_qt.convertToFormat(QImage::Format_RGB888);
    if (!ImageSaver("images/period_text_qt.jpg", single_img_qt))
        std::cout << "save qt grayscale with color text image NOT successfully\n";

    // -------- the three channel image(RGB color image)
    cv::Mat color_img_mat = cv::imread(color_filename);
    //QImage color_img_qt = WeiLi::Utility::mat_to_qimage_cpy(color_img_mat);
    QImage color_img_qt = WeiLi::Utility::mat_to_qimage_ref(color_img_mat);
    color_img_qt.save("images/baboon_mat2qt.jpg");

    std::cout << "======== Convert QImage into OpenCV ========\n";
    // -------- the single channel image(grayscale image)
    QImage single_img_qt2(single_filename_qt);
    //single_img_qt2.load(single_filename_qt);
    //cv::Mat single_img = WeiLi::Utility::qimage_to_mat_cpy(single_img_qt2);
    cv::Mat single_img = WeiLi::Utility::qimage_to_mat_ref(single_img_qt2);

    cv::cvtColor(single_img, single_img, cv::COLOR_GRAY2BGR);
    cv::putText(single_img, "width: 35.20 mm", cv::Point(20, 100),
        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0));

    cv::imwrite("images/period_qt2mat.jpg", single_img);
    QImage img_qt_text = WeiLi::Utility::mat_to_qimage_cpy(single_img);
    img_qt_text.save("images/period_text.jpg");

    // -------- the three channel image(RGB color image)
    QImage color_img_qt2(color_filename_qt);
    //color_img_qt2.load(color_filename_qt);
    //cv::Mat color_img = WeiLi::Utility::qimage_to_mat_cpy(color_img_qt2);
    cv::Mat color_img = WeiLi::Utility::qimage_to_mat_ref(color_img_qt2);
    cv::imwrite("images/baboon_qt2mat.jpg", color_img);

    std::cout << "[=====] Save the generated image successfully\n";
}
