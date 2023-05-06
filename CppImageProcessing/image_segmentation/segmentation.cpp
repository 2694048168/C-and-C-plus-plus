/**
 * @brief Goal
 * 1. Use the OpenCV function cv::filter2D in order to
 *    perform some laplacian filtering for image sharpening;
 *
 * 2. Use the OpenCV function cv::distanceTransform in order to
 *    obtain the derived representation of a binary image, where the value of
 *    each pixel is replaced by its distance to the nearest background pixel;
 *
 * 3. Use the OpenCV function cv::watershed in order to
 *    isolate objects in the image from the background;
 *
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-06
 * @version Samples of OpenCV 4.7
 *
 */

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <vector>

/**
 * @brief function of main and the entry of program
 */
int main(int argc, const char **argv)
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* -------------------------------------------------- */
    const char *keys = "{@input | cards.png | input image}";
    cv::CommandLineParser parser(argc, argv, keys);
    cv::Mat src = cv::imread(cv::samples::findFile(
        parser.get<cv::String>("@input")));
    if (src.empty())
    {
        std::cout << "[Error] Could not open or find the image\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <input image>" << std::endl;
        return EXIT_FAILURE;
    }
    cv::imshow("Original Image", src);

    /* Change the background from white to black, since that will help later
    to extract better results during the use of Distance Transform.
    ---------------------------------------------------------------- */
    cv::Mat mask;
    cv::inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
    src.setTo(cv::Scalar(0, 0, 0), mask);
    cv::imshow("Black Background Image", src);

    // Create a kernel that we will use to sharpen our image;
    // an approximation of second derivative, a quite strong kernel.
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1,
                      1, -8, 1,
                      1, 1, 1);

    /* do the laplacian filtering as it is
    well, we need to convert everything in something more deeper then CV_8U
    because the kernel has some negative values,
    and we can expect in general to have a Laplacian image with negative values
    BUT a 8bits unsigned int (the one we are working with) can contain values
    from 0 to 255, so the possible negative number will be truncated.
    ----------------------------------------------------------------- */
    cv::Mat imgLaplacian;
    cv::filter2D(src, imgLaplacian, CV_32F, kernel);
    cv::Mat sharp;
    src.convertTo(sharp, CV_32F);
    cv::Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    cv::imshow("New Sharped Image", imgResult);

    // Create binary image from source image
    cv::Mat bw;
    cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
    cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imshow("Binary Image", bw);

    // Perform the distance transform algorithm
    cv::Mat dist;
    cv::distanceTransform(bw, dist, cv::DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
    cv::imshow("Distance Transform Image", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);

    // Dilate a bit the dist image
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(dist, dist, kernel1);
    cv::imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
    }

    // Draw the background marker
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
    cv::Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    cv::imshow("Markers", markers8u);

    // Perform the watershed algorithm
    cv::watershed(imgResult, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);

    cv::imshow("Markers_v2", mark);
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);

        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    // Visualize the final image
    cv::imshow("Final Result", dst);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}