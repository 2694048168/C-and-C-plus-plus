#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

// -----------------------------
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "[Usage] DisplayImage.out <Image_Path>\n";
        return -1;
    }
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!image.data)
    {
        std::cout << "[Error] No image data, please check filepath.\n";
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);

    cv::waitKey(0);

    return 0;
}
