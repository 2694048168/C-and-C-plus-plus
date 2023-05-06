/**
 * @brief Goal
 * In this tutorial will learn how to:
 *  1. Use the OpenCV function matchTemplate() to search for
 *     matches between an image patch and an input image
 *  2. Use the OpenCV function minMaxLoc() to find the maximum and minimum
 *     values (as well as their positions) in a given array.
 *
 * OpenCV implements Template matching in the function matchTemplate():
 * 1. method=TM_SQDIFF
 * 2. method=TM_SQDIFF_NORMED
 * 3. method=TM_CCORR
 * 4. method=TM_CCORR_NORMED
 * 5. method=TM_CCORR_NORMED
 * 6. method=TM_CCOEFF_NORMED
 *
 * @file template_matching.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

/**
 * @brief main function
 *
 * 1. Loads an input image, an image patch (template), and optionally a mask
 *
 * 2. Perform a template matching procedure by using the OpenCV function
 *    matchTemplate() with any of the 6 matching methods described before.
 *    The user can choose the method by entering its selection in the Trackbar.
 *    If a mask is supplied, it will only be used for the methods
 *    that support masking.
 *
 * 3. Normalize the output of the matching procedure
 * 4. Localize the location with higher matching probability
 * 5. Draw a rectangle around the area corresponding to the highest match
 *
 */

// Global Variables
bool use_mask;
cv::Mat img;
cv::Mat templ;
cv::Mat mask;
cv::Mat result;
const char *image_window = "Source Image";
const char *result_window = "Result window";

int match_method;
int max_Trackbar = 5;

// Function Headers
void MatchingMethod(int, void *);

const char *keys =
    "{ help  h| | Print help message. }"
    "{ @input1 | Template_Matching_Original_Image.jpg | image_name }"
    "{ @input2 | Template_Matching_Template_Image.jpg | template_name }"
    "{ @input3 |  | mask_name }";

int main(int argc, const char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    img = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input1")));
    templ = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input2")),
                       cv::IMREAD_COLOR);

    if (argc > 3)
    {
        use_mask = true;
        mask = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input3")), cv::IMREAD_COLOR);
    }

    if (img.empty() || templ.empty() || (use_mask && mask.empty()))
    {
        std::cout << "Can't read one of the images" << std::endl;
        return EXIT_FAILURE;
    }

    cv::namedWindow(image_window, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(result_window, cv::WINDOW_AUTOSIZE);

    // Create Trackbar
    const char *trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";

    cv::createTrackbar(trackbar_label, image_window,
                       &match_method, max_Trackbar,
                       MatchingMethod);

    MatchingMethod(0, 0);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}

/**
 * @function MatchingMethod
 * @brief Trackbar callback
 */
void MatchingMethod(int, void *)
{
    cv::Mat img_display;
    img.copyTo(img_display);

    // Create the result matrix
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create(result_rows, result_cols, CV_32FC1);

    //! [match_template]
    // Do the Matching and Normalize
    bool method_accepts_mask = (cv::TM_SQDIFF == match_method ||
                                match_method == cv::TM_CCORR_NORMED);
    if (use_mask && method_accepts_mask)
    {
        matchTemplate(img, templ, result, match_method, mask);
    }
    else
    {
        matchTemplate(img, templ, result, match_method);
    }

    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    //! [best_match]
    // Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;

    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    //! [match_loc]
    /* For SQDIFF and SQDIFF_NORMED, the best matches are lower values.
    For all the other methods, the higher the better */
    if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
    }
    else
    {
        matchLoc = maxLoc;
    }

    // Show me what you got
    cv::rectangle(img_display, matchLoc,
                  cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
                  cv::Scalar::all(0), 2, 8, 0);

    cv::rectangle(result, matchLoc,
                  cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
                  cv::Scalar::all(0), 2, 8, 0);

    cv::imshow(image_window, img_display);
    cv::imshow(result_window, result);
}
