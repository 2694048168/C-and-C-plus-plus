/**
 * Goals
 * In this tutorial will learn how to:
 *   Draw a line by using the OpenCV function line()
 *   Draw an ellipse by using the OpenCV function ellipse()
 *   Draw a rectangle by using the OpenCV function rectangle()
 *   Draw a circle by using the OpenCV function circle()
 *   Draw a filled polygon by using the OpenCV function fillPoly()
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// #define WIDTH 400
const int WIDTH = 400;

void MyEllipse(cv::Mat &img, const double angle);
void MyFilledCircle(cv::Mat &img, const cv::Point center);
void MyPolygon(cv::Mat &img);
void MyLine(cv::Mat &img, const cv::Point start, const cv::Point end);

// ------------------------------------
int main(int argc, char const *argv[])
{
    /* represents a 2D point, sepecified by its image coordinates x and y */
    cv::Point position;
    position.x = 24;
    position.y = 42;
    cv::Point position_xy = cv::Point(42, 24);
    std::cout << "the image coordinates(x,y): " << position << std::endl;
    std::cout << "the image coordinates(x,y): " << position_xy << std::endl;

    /* Represents a 4-element vertor, widely used for passing pixel values.
    use it to extensively to represent BGR or BGRA color values. */
    double blue_value = 0.2;
    double green_value = 0.6;
    double red_value = 0.8;
    double alpha_value = 1.0;
    auto pixel_bgr = cv::Scalar(blue_value, green_value, red_value);
    auto pixel_bgra = cv::Scalar(blue_value, green_value, red_value, alpha_value);
    std::cout << "the pixel values(BGR): " << pixel_bgr << std::endl;
    std::cout << "the pixel values(BGRA): " << pixel_bgra << std::endl;

    /* simple demos(atom and rook) for geometric shape drawing.
    ------------------------------------------------------------*/
    // const cv::String atom_window = "Drawing 1: Atom";
    // const cv::String rook_window = "Drawing 2: Rook";
    // const std::string atom_window = "Drawing 1: Atom";
    // const std::string rook_window = "Drawing 2: Rook";
    // const char atom_window[] = "Drawing 1: Atom";
    // const char rook_window[] = "Drawing 2: Rook";
    const char *atom_window = "Drawing 1: Atom";
    const char *rook_window = "Drawing 2: Rook";

    cv::Mat atom_image = cv::Mat::zeros(WIDTH, WIDTH, CV_8UC3);
    cv::Mat rook_image = cv::Mat::zeros(WIDTH, WIDTH, CV_8UC3);

    MyEllipse(atom_image, 90);
    MyEllipse(atom_image, 0);
    MyEllipse(atom_image, 45);
    MyEllipse(atom_image, -45);
    MyFilledCircle(atom_image, cv::Point(WIDTH / 2, WIDTH / 2));

    MyPolygon(rook_image);
    cv::rectangle(rook_image,
                  cv::Point(0, 7 * WIDTH / 8),
                  cv::Point(WIDTH, WIDTH),
                  cv::Scalar(0, 255, 255),
                  cv::FILLED, /* the 'thickness' = -1 means filled the shape. */
                  cv::LINE_8);
    MyLine(rook_image,
           cv::Point(0, 15 * WIDTH / 16),
           cv::Point(WIDTH, 15 * WIDTH / 16));
    MyLine(rook_image,
           cv::Point(WIDTH / 4, 7 * WIDTH / 8),
           cv::Point(WIDTH / 4, WIDTH));
    MyLine(rook_image,
           cv::Point(WIDTH / 2, 7 * WIDTH / 8),
           cv::Point(WIDTH / 2, WIDTH));
    MyLine(rook_image,
           cv::Point(3 * WIDTH / 4, 7 * WIDTH / 8),
           cv::Point(3 * WIDTH / 4, WIDTH));

    cv::imshow(atom_window, atom_image);
    cv::moveWindow(atom_window, 0, 200);

    cv::imshow(rook_window, rook_image);
    cv::moveWindow(rook_window, WIDTH, 200);

    cv::waitKey(0);

    return 0;
}

void MyEllipse(cv::Mat &img, const double angle)
{
    int thickness = 2;
    int lineType = 8;

    cv::ellipse(img,
                cv::Point(WIDTH / 2, WIDTH / 2),
                cv::Size(WIDTH / 4, WIDTH / 16),
                angle,
                0,
                360,
                cv::Scalar(255, 0, 0, 0),
                thickness,
                lineType);
}

void MyFilledCircle(cv::Mat &img, const cv::Point center)
{
    cv::circle(img,
               center,
               WIDTH / 32,
               cv::Scalar(0, 0, 255),
               cv::FILLED, /* the 'thickness' = -1 means filled the shape. */
               cv::LINE_8);
}

void MyPolygon(cv::Mat &img)
{
    int lineType = cv::LINE_8;
    cv::Point rook_points[1][20]; /* vertices of the polygon */
    rook_points[0][0] = cv::Point(WIDTH / 4, 7 * WIDTH / 8);
    rook_points[0][1] = cv::Point(3 * WIDTH / 4, 7 * WIDTH / 8);
    rook_points[0][2] = cv::Point(3 * WIDTH / 4, 13 * WIDTH / 16);
    rook_points[0][3] = cv::Point(11 * WIDTH / 16, 13 * WIDTH / 16);
    rook_points[0][4] = cv::Point(19 * WIDTH / 32, 3 * WIDTH / 8);
    rook_points[0][5] = cv::Point(3 * WIDTH / 4, 3 * WIDTH / 8);
    rook_points[0][6] = cv::Point(3 * WIDTH / 4, WIDTH / 8);
    rook_points[0][7] = cv::Point(26 * WIDTH / 40, WIDTH / 8);
    rook_points[0][8] = cv::Point(26 * WIDTH / 40, WIDTH / 4);
    rook_points[0][9] = cv::Point(22 * WIDTH / 40, WIDTH / 4);
    rook_points[0][10] = cv::Point(22 * WIDTH / 40, WIDTH / 8);
    rook_points[0][11] = cv::Point(18 * WIDTH / 40, WIDTH / 8);
    rook_points[0][12] = cv::Point(18 * WIDTH / 40, WIDTH / 4);
    rook_points[0][13] = cv::Point(14 * WIDTH / 40, WIDTH / 4);
    rook_points[0][14] = cv::Point(14 * WIDTH / 40, WIDTH / 8);
    rook_points[0][15] = cv::Point(WIDTH / 4, WIDTH / 8);
    rook_points[0][16] = cv::Point(WIDTH / 4, 3 * WIDTH / 8);
    rook_points[0][17] = cv::Point(13 * WIDTH / 32, 3 * WIDTH / 8);
    rook_points[0][18] = cv::Point(5 * WIDTH / 16, 13 * WIDTH / 16);
    rook_points[0][19] = cv::Point(WIDTH / 4, 13 * WIDTH / 16);
    const cv::Point *ppt[1] = {rook_points[0]};
    int npt[] = {20};
    cv::fillPoly(img,
                 ppt,
                 npt,
                 1,
                 cv::Scalar(255, 255, 255),
                 lineType);
}

void MyLine(cv::Mat &img, const cv::Point start, const cv::Point end)
{
    int thickness = 2;
    int lineType = cv::LINE_8;
    cv::line(img,
             start,
             end,
             cv::Scalar(0, 0, 0),
             thickness,
             lineType);
}
