/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::warpAffine
 *          to implement simple remapping routines.
 *
 * ---- Use the OpenCV function cv::getRotationMatrix2D
 *          to obtain a 2×3 rotation matrix.
 *
 * Affine Transformation to express(2D for Image and 3D for Graph):
 * 1. Rotations (linear transformation)
 * 2. Translations (vector addition)
 * 3. Scale operations (linear transformation)
 *
 * @file affine.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

/**
 * @brief main function
 */
int main(int argc, char const **argv)
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    cv::CommandLineParser parser(argc, argv,
                                 "{@image | lena.jpg | input image name}");
    std::string filename = parser.get<std::string>(0);
    cv::Mat src = cv::imread(cv::samples::findFile(filename),
                             cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Cannot read image: " << filename << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    // affine transform matrix
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f(0.0f, 0.0f);
    srcTri[1] = cv::Point2f(src.cols - 1.0f, 0.0f);
    srcTri[2] = cv::Point2f(0.0f, src.rows - 1.0f);

    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f(0.0f, src.rows * 0.33f);
    dstTri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.7f);

    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);

    cv::Mat warp_dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());

    cv::Point center = cv::Point(warp_dst.cols / 2, warp_dst.rows / 2);
    double angle = -50.0;
    double scale = 0.6;

    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);

    cv::Mat warp_rotate_dst;
    cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());

    const int offset = 200;
    cv::imshow("Source image", src);
    cv::moveWindow("Source image", offset, offset);

    cv::imshow("Warp", warp_dst);
    cv::moveWindow("Warp", src.cols + offset, offset);

    cv::imshow("Warp + Rotate", warp_rotate_dst);
    cv::moveWindow("Warp + Rotate", 2 * src.cols + offset, offset);

    cv::waitKey();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
