/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Use the OpenCV functions pyrUp() and pyrDown()
 * to downsample or upsample a given image.
 *
 * @file extract_pattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-02
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    std::cout << "\n Zoom In-Out Image \n"
                 "------------------  \n"
                 " * [i] -> Zoom in   \n"
                 " * [o] -> Zoom out  \n"
                 " * [ESC] -> Close program \n"
              << std::endl;

    const char *filename = argc >= 2 ? argv[1] : "chicky_512.png";
    cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n";
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;

        return EXIT_FAILURE;
    }

    const char *window_name = "Image Pyramids";
    while (true)
    {
        cv::imshow(window_name, src);
        // char ch = (char)cv::waitKey(0);
        char ch = static_cast<char>(cv::waitKey(0));

        if (ch == 27) /* Esc keyboard */
        {
            break;
        }
        else if (ch == 'i')
        {
            cv::pyrUp(src, src, cv::Size(src.cols * 2, src.rows * 2));
            std::cout << "[====] Image Zoom In by x2 \n";
        }
        else if (ch == 'o')
        {
            cv::pyrDown(src, src, cv::Size(src.cols / 2, src.rows / 2));
            std::cout << "[====] Image Zoom Out by /2 \n";
        }
    }

    return EXIT_SUCCESS;
}
