/**
 * @File    : morphology.cpp
 * @Brief   : 图像形态学操作: 腐蚀和膨胀运算; 开操作和闭操作
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

/** 腐蚀和膨胀运算
 * 和所有形态学滤波器一样，这两个滤波器的作用范围是由结构元素定义的像素集。
 * 在某个像素上应用结构元素时，结构元素的锚点与该像素对齐，所有与结构元素相交的像素就包含在当前集合中。 
 * 腐蚀就是把当前像素替换成所定义像素集合中的最小像素值； 
 * 膨胀是腐蚀的反运算，它把当前像素替换成所定义像素集合中的最大像素值。
 * 
 * 由于输入的二值图像只包含黑色（值为 0）和白色（值为 255）像素，因此每个像素都会被替换成白色或黑色像素。
 * 要形象地理解这两种运算的作用，可考虑背景（黑色）和前景（白色）的物体。腐蚀时，
 * 如果结构元素放到某个像素位置时碰到了背景（即交集中有一个像素是黑色的），那么这个像素就变为背景；
 * 膨胀时，如果结构元素放到某个背景像素位置时碰到了前景物体，那么这个像素就被标为白色。
 * 正因如此，图像腐蚀后物体尺寸会缩小（形状被腐蚀），而图像膨胀后物体会扩大。
 * 
 * 在腐蚀图像中，有些面积较小的物体（可看作背景中的“噪声”像素）会彻底消失。
 * 与之类似，膨胀后的物体会变大，而物体中一些“空隙”会被填满。 
 * 
 * OpenCV 默认使用 3×3 正方形结构元素。
 * 在调用函数时，将第三个参数指定为空矩阵（即 cv::Mat()），就能得到默认的结构元素。
 * 也可以通过提供一个矩阵来指定结构元素的大小（以及形状），矩阵中的非零元素将构成结构元素。
 * 
 * 实际上，对图像腐蚀两次相当于对结构元素自身膨胀后的图像进行腐蚀。这个规则也适用于膨胀。
 * 最后，鉴于前景/背景概念有很大的随意性，可得到以下的实验结论（这是腐蚀/膨胀运算的基本性质）
 * 用结构元素腐蚀前景物体可看作对图像背景部分的膨胀，也就是说：
 *   腐蚀图像相当于对其反色图像膨胀后再取反色；
 *   膨胀图像相当于对其反色图像腐蚀后再取反色。
 */

/**开操作和闭操作
 * 开启和闭合滤波器的定义只与基本的腐蚀和膨胀运算有关： 
 *    闭合的定义是对图像先膨胀后腐蚀
 *    开启的定义是对图像先腐蚀后膨胀
 * 
 * 注意，对一幅图像进行多次同样的开启运算是没有作用的（闭合运算也一样）
 * 事实上，因为第一次使用开启滤波器时已经填充了空隙，再使用同一个滤波器将不会使图像产生变化。
 * 用数学术语讲，这些运算是幂等（ idempotent）的
 */

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/binary.bmp");
    if (!image.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }
    cv::namedWindow("OrigialImage");
    cv::imshow("OrigialImage", image);

    // 腐蚀图像，默认使用 3X3 的结构元素 —— efault 3x3 structuring element (SE)
    cv::Mat eroded_image;
    cv::erode(image, eroded_image, cv::Mat());
    cv::namedWindow("Eroded Image");
    cv::imshow("Eroded Image", eroded_image);

    // 膨胀图像，默认使用 3X3 的结构元素 —— efault 3x3 structuring element (SE)
    cv::Mat dilated_image;
    cv::dilate(image, dilated_image, cv::Mat());
    cv::namedWindow("Dilated Image");
    cv::imshow("Dilated Image", dilated_image);

    // Erode the image with a larger SE
    // create a 7x7 mat with containing all 1s
    cv::Mat element(7, 7, CV_8U, cv::Scalar(1));
    // erode the image with that SE
    cv::erode(image, eroded_image, element);

    // Display the eroded image
    cv::namedWindow("Eroded Image (7x7)");
    cv::imshow("Eroded Image (7x7)", eroded_image);

    // Erode the image 3 times.
    cv::erode(image, eroded_image, cv::Mat(), cv::Point(-1, -1), 3);

    // Display the eroded image
    cv::namedWindow("Eroded Image (3 times)");
    cv::imshow("Eroded Image (3 times)", eroded_image);

    // 开操作和闭操作
    // Close the image
    cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));
    cv::Mat closed_image;
    cv::morphologyEx(image, closed_image, cv::MORPH_CLOSE, element5); //运算符 & 结构元素
    cv::namedWindow("Closed Image");
    cv::imshow("Closed Image", closed_image);

    // Open the image
    cv::Mat opened_image;
    cv::morphologyEx(image, opened_image, cv::MORPH_OPEN, element5);
    cv::namedWindow("Opened Image");
    cv::imshow("Opened Image", opened_image);

    // 按照定义依次进行明确的膨胀和腐蚀运算完成开闭操作
    // explicit closing
    // 1. dilate original image
    cv::Mat result;
    cv::dilate(image, result, element5);
    // 2. in-place erosion of the dilated image
    cv::erode(result, result, element5);
    cv::namedWindow("Closed Image (2)");
    cv::imshow("Closed Image (2)", result);

    // Close and Open the image
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, element5);
    cv::morphologyEx(image, image, cv::MORPH_OPEN, element5);

    // Display the close/opened image
    cv::namedWindow("Closed|Opened Image");
    cv::imshow("Closed|Opened Image", image);
    cv::imwrite("binaryGroup.bmp", image);

    // Read input image
    image = cv::imread("./../images/binary.bmp");
    if (!image.data)
    {
        std::cerr << "--Error reading binary image file." << std::endl;
        return 1;
    }

    // Open and Close the image
    cv::morphologyEx(image, image, cv::MORPH_OPEN, element5);
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, element5);

    // Display the close/opened image
    cv::namedWindow("Opened|Closed Image");
    cv::imshow("Opened|Closed Image", image);

    // 在灰度图像中应用形态学运算
    // 应用于灰度图像上可以检测图像的特征
    // Read input image (gray-level)
    image = cv::imread("./../images/boldt.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading boldt image file." << std::endl;
        return 1;
    }

    // Get the gradient image using a 3x3 structuring element
    // 用 3×3 结构元素得到梯度图像
    cv::morphologyEx(image, result, cv::MORPH_GRADIENT, cv::Mat());

    // Display the morphological edge image
    cv::namedWindow("Edge Image");
    // 为方便观察，对图像做了反色处理
    cv::imshow("Edge Image", 255 - result);

    // Apply threshold to obtain a binary image
    int threshold(80);
    cv::threshold(result, result,
                  threshold, 255, cv::THRESH_BINARY);

    // Display the close/opened image
    cv::namedWindow("Thresholded Edge Image");
    cv::imshow("Thresholded Edge Image", result);

    // Get the gradient image using a 3x3 structuring element
    cv::morphologyEx(image, result, cv::MORPH_GRADIENT, cv::Mat());

    // 很实用的形态学运算是顶帽（ hat-top）变换，
    // 它可以从图像中提取出局部的小型前景物体。
    /**hat-top 变换
     * 理解形态学运算在灰度图像上的效果有一个好办法，
     * 就是把图像看作是一个拓扑地貌，不同的灰度级别代表不同的高度（或海拔）。
     * 基于这种观点，明亮的区域代表高山，黑暗的区域代表深谷；
     * 边缘相当于黑暗和明亮像素之间的快速过渡，因此可以比作陡峭的悬崖。
     * 腐蚀这种地形的最终结果是：每个像素被替换成特定邻域内的最小值，从而降低它的高度。结果是悬崖“缩小”，山谷“扩大”。
     * 膨胀的效果刚好相反，即悬崖“扩大”，山谷“缩小”。但不管哪种情况，平地（即强度值固定的区域）都会相对保持不变。
     * 
     * 根据这个结论，可以得到一种检测图像边缘（或悬崖）的简单方法，
     * 即通过计算膨胀后的图像与腐蚀后的图像之间的的差距得到边缘。
     * 因为这两种转换后图像的差别主要在边缘地带，所以相减后会突出边缘。
     * 显然，结构元素越大，检测到的边缘就越宽。这种边缘检测运算称为 Beucher 梯度
     * 
     * 还有两种简单的方法能得到类似结果，即用膨胀后的图像减去原始图像，或者用原始图像减去腐蚀后的图像，那样得到的边缘会更窄。
     * 
     * 顶帽运算也基于图像比对，它使用了开启和闭合运算。
     * 因为灰度图像进行形态学开启运算时会先对图像进行腐蚀，局部的尖锐部分会被消除，其他部分则将保留下来。
     * 因此，原始图像和经过开启运算的图像的比对结果就是局部的尖锐部分。
     * 这些尖锐部分就是我们需要提取的前景物体。
     */
    // Read input image (gray-level)
    image = cv::imread("./../images/book.jpg", 0);
    if (!image.data)
        return 0;
    // rotate the image for easier display
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    // Apply the black top-hat transform using a 7x7 structuring element
    cv::Mat element7(7, 7, CV_8U, cv::Scalar(1));
    cv::morphologyEx(image, result, cv::MORPH_BLACKHAT, element7);

    // Display the top-hat image
    cv::namedWindow("7x7 Black Top-hat Image");
    cv::imshow("7x7 Black Top-hat Image", 255 - result);

    // Apply threshold to obtain a binary image
    threshold = 25;
    cv::threshold(result, result,
                  threshold, 255, cv::THRESH_BINARY);

    // Display the morphological edge image
    cv::namedWindow("Thresholded Black Top-hat");
    cv::imshow("Thresholded Black Top-hat", 255 - result);

    // Apply the black top-hat transform using a 7x7 structuring element
    cv::morphologyEx(image, result, cv::MORPH_CLOSE, element7);

    // Display the top-hat image
    cv::namedWindow("7x7 Closed Image");
    cv::imshow("7x7 Closed Image", 255 - result);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
