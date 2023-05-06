/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::remap to implement simple remapping routines.
 *
 * @file remapping.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

void update_map(int &ind, cv::Mat &map_x, cv::Mat &map_y);

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    cv::CommandLineParser parser(argc, argv,
                                 "{@image |chicky_512.png|input image name}");
    std::string filename = parser.get<std::string>(0);
    cv::Mat src = cv::imread(cv::samples::findFile(filename),
                             cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Cannot read image: " << filename << std::endl;
        return -1;
    }

    cv::Mat dst(src.size(), src.type());
    cv::Mat map_x(src.size(), CV_32FC1);
    cv::Mat map_y(src.size(), CV_32FC1);

    const char *remap_window = "Image Remapping";
    cv::namedWindow(remap_window, cv::WINDOW_AUTOSIZE);

    int ind = 0;
    for (;;)
    {
        update_map(ind, map_x, map_y);

        cv::remap(src, dst, map_x, map_y,
                  cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT,
                  cv::Scalar(0, 0, 0));

        cv::imshow(remap_window, dst);

        char c = (char)cv::waitKey(1000);
        if (c == 27)
        {
            break;
        }
    }

    return EXIT_SUCCESS;
}

/**
 * @brief Updating the mapping matrices: perform 4 different mappings. 
 *  map_x represents the first coordinate of h(i,j)
 *  and map_y the second coordinate.
*/
void update_map(int &ind, cv::Mat &map_x, cv::Mat &map_y)
{
    for (int i = 0; i < map_x.rows; i++)
    {
        for (int j = 0; j < map_x.cols; j++)
        {
            switch (ind)
            {
            case 0:
                if (j > map_x.cols * 0.25 && j < map_x.cols * 0.75 && i > map_x.rows * 0.25 && i < map_x.rows * 0.75)
                {
                    map_x.at<float>(i, j) = 2 * (j - map_x.cols * 0.25f) + 0.5f;
                    map_y.at<float>(i, j) = 2 * (i - map_x.rows * 0.25f) + 0.5f;
                }
                else
                {
                    map_x.at<float>(i, j) = 0;
                    map_y.at<float>(i, j) = 0;
                }
                break;
            case 1:
                map_x.at<float>(i, j) = (float)j;
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            case 2:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)i;
                break;
            case 3:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            default:
                break;
            } // end of switch
        }
    }
    ind = (ind + 1) % 4;
}