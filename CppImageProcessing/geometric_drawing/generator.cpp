/**
 * Goals
 * In this tutorial will learn how to:
 *  1. Use the Random Number generator class (cv::RNG )
 *    and how to get a random number from a uniform distribution.
 *  2. Display text on an OpenCV window by using the function cv::putText
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Global Variables
const int NUMBER = 100;
const int DELAY = 6;

const int window_width = 900;
const int window_height = 600;
int x_1 = -window_width / 2;
int x_2 = window_width * 3 / 2;
int y_1 = -window_width / 2;
int y_2 = window_width * 3 / 2;

// Function headers
static cv::Scalar randomColor(cv::RNG &rng);
int Drawing_Random_Lines(cv::Mat &image, const char *window_name, cv::RNG rng);
int Drawing_Random_Rectangles(cv::Mat &image, const char *window_name, cv::RNG rng);
int Drawing_Random_Ellipses(cv::Mat &image, const char *window_name, cv::RNG rng);
int Drawing_Random_Polylines(cv::Mat &image, const char *window_name, cv::RNG rng);
int Drawing_Random_Filled_Polygons(cv::Mat &image, const char *window_name, cv::RNG rng);
int Drawing_Random_Circles(cv::Mat &image, const char *window_name, cv::RNG rng);
int Displaying_Random_Text(cv::Mat &image, const char *window_name, cv::RNG rng);
int Displaying_Big_End(cv::Mat &image, const char *window_name, cv::RNG rng);

// ------------------------------------
int main(int argc, char const *argv[])
{
    // the random number generator.
    cv::RNG rng(0xFFFFFFFF); /* 64-bit value used to initialize the RNG. */

    // create a image with black(zeros)
    const char *window_name = "random crazy image";
    cv::Mat image = cv::Mat::zeros(window_height, window_width, CV_8UC3);
    cv::imshow(window_name, image);

    int c = 0;
    c = Drawing_Random_Lines(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Drawing_Random_Rectangles(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Drawing_Random_Ellipses(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Drawing_Random_Polylines(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Drawing_Random_Filled_Polygons(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Drawing_Random_Circles(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }
    c = Displaying_Random_Text(image, window_name, rng);
    if (c != 0)
    {
        return 0;
    }

    c = Displaying_Big_End(image, window_name, rng);

    return 0;
}

static cv::Scalar randomColor(cv::RNG &rng)
{
    int icolor = (unsigned)rng;
    /* B-G-R-A(0): such as rng is #0xF9E6D8FF(64 bits),
    B value --> 0xF9E6D8 & 0x0000FF(255) --> 0x0000D8 --> 216
    G value --> 0xF9E6D8>>8 -->0x00F9E6 & 0x0000FF(255) --> 0x0000E6 --> 230
    R value --> 0xF9E6D8>>16 -->0x0000F9 & 0x0000FF(255) --> 0x0000F9 --> 249
    -----------------------------------------------------------------
    we can verify it in this link: https://godbolt.org/z/n1ePjTcjG
    ----------------------------------------------------------------- */
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

int Drawing_Random_Lines(cv::Mat &image, const char *title, cv::RNG rng)
{
    cv::Point pt1, pt2;
    for (int i = 0; i < NUMBER; ++i)
    {
        /* This generates a randomly uniformed distribution between
        the values a and b (inclusive in a, exclusive in b). */
        pt1.x = rng.uniform(x_1, x_2);
        pt1.y = rng.uniform(y_1, y_2);

        pt2.x = rng.uniform(x_1, x_2);
        pt2.y = rng.uniform(y_1, y_2);

        cv::line(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), 8);
        cv::imshow(title, image);

        /* waitKey(25) will display a frame and wait approximately 25 ms
        for a key press (suitable for displaying a video frame-by-frame). */
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }
    return 0;
}

int Drawing_Random_Rectangles(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    cv::Point pt1, pt2;
    int lineType = 8;
    int thickness = rng.uniform(-3, 10);

    for (int i = 0; i < NUMBER; i++)
    {
        pt1.x = rng.uniform(x_1, x_2);
        pt1.y = rng.uniform(y_1, y_2);
        pt2.x = rng.uniform(x_1, x_2);
        pt2.y = rng.uniform(y_1, y_2);

        cv::rectangle(image, pt1, pt2, randomColor(rng),
                      MAX(thickness, -1), lineType);

        cv::imshow(window_name, image);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Drawing_Random_Ellipses(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    int lineType = 8;

    for (int i = 0; i < NUMBER; i++)
    {
        cv::Point center;
        center.x = rng.uniform(x_1, x_2);
        center.y = rng.uniform(y_1, y_2);

        cv::Size axes;
        axes.width = rng.uniform(0, 200);
        axes.height = rng.uniform(0, 200);

        double angle = rng.uniform(0, 180);

        cv::ellipse(image, center, axes, angle, angle - 100, angle + 200,
                    randomColor(rng), rng.uniform(-1, 9), lineType);

        cv::imshow(window_name, image);

        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Drawing_Random_Polylines(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    for (int i = 0; i < NUMBER; i++)
    {
        cv::Point pt[2][3];
        pt[0][0].x = rng.uniform(x_1, x_2);
        pt[0][0].y = rng.uniform(y_1, y_2);
        pt[0][1].x = rng.uniform(x_1, x_2);
        pt[0][1].y = rng.uniform(y_1, y_2);
        pt[0][2].x = rng.uniform(x_1, x_2);
        pt[0][2].y = rng.uniform(y_1, y_2);
        pt[1][0].x = rng.uniform(x_1, x_2);
        pt[1][0].y = rng.uniform(y_1, y_2);
        pt[1][1].x = rng.uniform(x_1, x_2);
        pt[1][1].y = rng.uniform(y_1, y_2);
        pt[1][2].x = rng.uniform(x_1, x_2);
        pt[1][2].y = rng.uniform(y_1, y_2);

        const cv::Point *ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};

        cv::polylines(image, ppt, npt, 2, true, randomColor(rng),
                      rng.uniform(1, 10), 8);

        cv::imshow(window_name, image);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Drawing_Random_Filled_Polygons(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    int lineType = 8;

    for (int i = 0; i < NUMBER; i++)
    {
        cv::Point pt[2][3];
        pt[0][0].x = rng.uniform(x_1, x_2);
        pt[0][0].y = rng.uniform(y_1, y_2);
        pt[0][1].x = rng.uniform(x_1, x_2);
        pt[0][1].y = rng.uniform(y_1, y_2);
        pt[0][2].x = rng.uniform(x_1, x_2);
        pt[0][2].y = rng.uniform(y_1, y_2);
        pt[1][0].x = rng.uniform(x_1, x_2);
        pt[1][0].y = rng.uniform(y_1, y_2);
        pt[1][1].x = rng.uniform(x_1, x_2);
        pt[1][1].y = rng.uniform(y_1, y_2);
        pt[1][2].x = rng.uniform(x_1, x_2);
        pt[1][2].y = rng.uniform(y_1, y_2);

        const cv::Point *ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};

        cv::fillPoly(image, ppt, npt, 2, randomColor(rng), lineType);

        cv::imshow(window_name, image);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Drawing_Random_Circles(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    int lineType = 8;

    for (int i = 0; i < NUMBER; i++)
    {
        cv::Point center;
        center.x = rng.uniform(x_1, x_2);
        center.y = rng.uniform(y_1, y_2);

        cv::circle(image, center, rng.uniform(0, 300), randomColor(rng),
                   rng.uniform(-1, 9), lineType);

        cv::imshow(window_name, image);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Displaying_Random_Text(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    int lineType = 8;
    const std::string text = "Testing text rendering";

    for (int i = 0; i < NUMBER; ++i)
    {
        /* The bottom-left corner of the text will be located in
         the cv::Point org */
        cv::Point org;
        org.x = rng.uniform(x_1, x_2);
        org.y = rng.uniform(y_1, y_2);

        cv::putText(image, text, org, rng.uniform(0, 8),
                    rng.uniform(0, 100) * 0.05 + 0.1, randomColor(rng),
                    rng.uniform(1, 10), lineType);
        cv::imshow(window_name, image);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}

int Displaying_Big_End(cv::Mat &image, const char *window_name, cv::RNG rng)
{
    const std::string text = "OpenCV Wei Li";
    cv::Size textsize = cv::getTextSize(text, cv::FONT_HERSHEY_COMPLEX,
                                        3.0, 5, 0);
    cv::Point org((window_width - textsize.width) / 2,
                  (window_height - textsize.height) / 2);

    cv::Mat image2;
    for (int i = 0; i < 255; ++i)
    {
        /* image2 is the subtraction of image and Scalar::all(i).
        In fact, what happens here is that every pixel of image2 will be
        the result of subtracting every pixel of image minus the value of i
        (remember that for each pixel we are considering three values
        such as R, G and B, so each of them will be affected)

        Also remember that the subtraction operation always performs internally
        a saturate operation, which means that the result obtained will always
        be inside the allowed range (no negative and between 0 and 255). */
        image2 = image - cv::Scalar::all(i);

        cv::putText(image2, text, org, cv::FONT_HERSHEY_COMPLEX,
                    3, cv::Scalar(i, i, 255), 5, 8);
        cv::imshow(window_name, image2);
        if (cv::waitKey(DELAY) >= 0)
        {
            return -1;
        }
    }

    return 0;
}
