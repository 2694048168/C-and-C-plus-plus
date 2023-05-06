/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Use the OpenCV function cv::morphologyEx to apply
 * Morphological Transformation based Erosion and Dilation:
 * Opening | Closing | Morphological Gradient | Top Hat | Black Hat
 *
 * @file morphology.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-02
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// global variables
cv::Mat src_img;
cv::Mat dst_img;

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
const int max_operator = 4;
const int max_elem = 2;
const int max_kernel_size = 21;

const char *window_name = "Morphology Transformations";

void morphologyOperations(int, void *);

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    cv::CommandLineParser parser(argc, argv,
                                 "{@input | baboon.jpg | input image}");
    src_img = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                         cv::IMREAD_UNCHANGED);
    if (src_img.empty())
    {
        std::cout << "[Error] Could not open or find the image file\n\n";
        std::cout << "Usage: " << argv[0] << " <input image path>" << std::endl;
        return -1;
    }

    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    /* Create a Trackbars for morphology operation:
    ------------------------------------------------ */
    cv::createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat",
                       window_name,
                       &morph_operator,
                       max_operator,
                       morphologyOperations);

    cv::createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse",
                       window_name,
                       &morph_elem,
                       max_elem,
                       morphologyOperations);

    cv::createTrackbar("Kernel size:\n 2n +1",
                       window_name,
                       &morph_size,
                       max_kernel_size,
                       morphologyOperations);

    morphologyOperations(0, 0); /* init */

    cv::waitKey(0);

    return 0;
}

/**
 * @brief morphlogy operators callback function with the
 * global variables: morph_elem | morph_size | morph_operator,
 * it will callback when they update or onchange.
 * 
 * Every time we move any slider, the user's function morphologyOperations
 * will be called to effectuate a new morphology operation 
 * and it will update the output image based on the current trackbar values.
 */
void morphologyOperations(int, void *)
{
    // Since MORPH_X : 2,3,4,5 and 6
    int operation = morph_operator + 2;
    cv::Mat element = cv::getStructuringElement(morph_elem,
                        cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                        cv::Point(morph_size, morph_size));

    /* operation: The kind of morphology transformation to be performed.
    Note that we have 5 alternatives:
        Opening: MORPH_OPEN : 2
        Closing: MORPH_CLOSE: 3
        Gradient: MORPH_GRADIENT: 4
        Top Hat: MORPH_TOPHAT: 5
        Black Hat: MORPH_BLACKHAT: 6
    As you can see the values range from <2-6>, that is why we add (+2) to
    the values entered by the Trackbar: 
    ----------------------------------------------------------*/
    cv::morphologyEx(src_img, dst_img, operation, element);

    cv::imshow(window_name, dst_img);
}