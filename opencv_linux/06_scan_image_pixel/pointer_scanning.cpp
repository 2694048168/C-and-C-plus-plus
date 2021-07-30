/**
 * @File    : pointer_scanning.cpp
 * @Brief   : 利用指针对图像像素进行扫描遍历
 * @Author  : Wei Li
 * @Date    : 2021-07-26
*/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Version 1
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        uchar *data = image.ptr<uchar>(row_index);
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            data[col_index] = data[col_index] / div * div + div / 2;
        }
    }
}

// Version 2
// 对输入图像进行深拷贝 对图像进行深复制最简单的方法是使用 clone()方法
void colorReduceIO(const cv::Mat &image, cv::Mat &result, int div = 64)
{
    int num_row = image.rows;
    int num_col = image.cols;
    int num_channel = image.channels();

    // allocate output image
    result.create(image.rows, image.cols, image.type());

    for (int row_index = 0; row_index < num_row; ++row_index)
    {
        const uchar *data_in = image.ptr<uchar>(row_index);
        uchar *data_out = result.ptr<uchar>(row_index);

        for (int col_index = 0; col_index < num_row * num_channel; ++col_index)
        {
            // pross each pixel
            data_out[col_index] = data_in[col_index] / div * div + div / 2;
        }
    }
}

// Version 3
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce1(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();
    uchar div2 = div >> 1; // div2 = div / 2;

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            // data[col_index] = data[col_index] / div * div + div / 2;
            *data++ = *data / div * div + div2;
        }
    }
}

// Version 4
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce2(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();
    uchar div2 = div >> 1;

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            // 利用模运算操作
            int pixel_value = *data;
            *data++ = pixel_value - pixel_value % div + div2;
        }
    }
}

// Version 5
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce3(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();
    // div 必须是 2 的幂次，才能使用 掩码 进行计算
    int numbers = static_cast<int>(std::log(static_cast<double>(div)) / std::log(2.0) + 0.5);
    // binary mask used to round the pixle value. 二进制掩码
    uchar mask = 0xFF << numbers;    // e.g. for div=16, mask=0xF0
    uchar div2 = 1 << (numbers - 1); // div2 = div / 2;

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            // 利用二进制掩码操作
            *data &= mask;
            *data++ |= div2; // add div/2
        }
    }
}

// Version 6
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce4(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();
    int numbers = static_cast<int>(std::log(static_cast<double>(div)) / std::log(2.0) + 0.5);

    // 用 step 数据属性可得到单位是字节的有效宽度
    // 性能上的考虑，会用几个额外的像素来填补行的长度, OpenCV 把经过填充的行的长度指定为有效宽度
    int step = image.step; // effective width
    // mask used to round the pixel value
    uchar mask = 0xFF << numbers;
    uchar div2 = div >> 1;

    // get the pointer to the image buffer.
    uchar *data = image.data;

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            *(data + col_index) &= mask;
            *(data + col_index) += div2;
        }
        data += step; // next line.
    }
}

// Version 7
// 矩阵行列进行扫描，利用行地址的指针
void colorReduce5(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int numbers = static_cast<int>(std::log(static_cast<double>(div)) / std::log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << numbers;

    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        // 每次重新计算行的大小
        for (int col_index = 0; col_index < image.cols * image.channels(); ++col_index)
        {
            // precess each pixel.
            *data &= mask;
            *data++ += div / 2;
        }
    }
}

// Version 8
// optimizes the case of continuous image
void colorReduce6(cv::Mat image, int div = 64)
{
    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();

    if (image.isContinuous())
    {
        // then no padded pixels.
        num_pixels_row = num_pixels_row * num_rows;
        num_rows = 1;
    }

    int numbers = static_cast<int>(std::log(static_cast<double>(div)) / std::log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << numbers;
    uchar div2 = div >> 1;

    // this for loop only once.
    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        // 每次重新计算行的大小
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            *data &= mask;
            *data++ += div2;
        }
    }
}

// Version 9
// applies reshape on continuous image
void colorReduce7(cv::Mat image, int div = 64)
{
    if (image.isContinuous())
    {
        // then no padded pixels.
        // (new number of channels, new number of rows)
        image.reshape(1, 1);
    }

    int num_rows = image.rows;
    int num_pixels_row = image.cols * image.channels();
    int numbers = static_cast<int>(std::log(static_cast<double>(div)) / std::log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << numbers;
    uchar div2 = div >> 1;

    // this for loop only once.
    for (int row_index = 0; row_index < num_rows; ++row_index)
    {
        // 行地址指针
        uchar *data = image.ptr<uchar>(row_index);
        // 每次重新计算行的大小
        for (int col_index = 0; col_index < num_pixels_row; ++col_index)
        {
            // precess each pixel.
            *data &= mask;
            *data++ += div2;
        }
    }
}

// Version 10
// processes the 3 channels inside the loop with Mat_ iterators
void colorReduce8(cv::Mat image, int div = 64)
{
    uchar div2 = div >> 1;

    // 利用 迭代器 进行访问图像像素
    cv::Mat_<cv::Vec3b>::iterator iterator_begin = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator iterator_end = image.end<cv::Vec3b>();
    for (; iterator_begin != iterator_end; ++iterator_begin)
    {
        // process each pixel.
        (*iterator_begin)[0] = (*iterator_begin)[0] / div * div + div2;
        (*iterator_begin)[1] = (*iterator_begin)[1] / div * div + div2;
        (*iterator_begin)[2] = (*iterator_begin)[2] / div * div + div2;
    }
}

// Version 11
// processes the 3 channels inside the loop with Mat_ iterators
// uses iterators on Vec3b
void colorReduce9(cv::Mat image, int div = 64)
{
    // 利用 迭代器 进行访问图像像素
    cv::Mat_<cv::Vec3b>::iterator iterator_begin = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator iterator_end = image.end<cv::Vec3b>();

    const cv::Vec3b offset(div / 2, div / 2, div / 2);

    for (; iterator_begin != iterator_end; ++iterator_begin)
    {
        // process each pixel.
        (*iterator_begin) = (*iterator_begin) / div * div + offset;
    }
}

// Version 12
// uses iterators with a binary mask
void colorReduce10(cv::Mat image, int div = 64)
{
    // div must be a power of 2
    int number = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << number; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1;  // div2 = div/2

    // 利用 迭代器 进行访问图像像素
    cv::Mat_<cv::Vec3b>::iterator iterator_begin = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator iterator_end = image.end<cv::Vec3b>();

    // 扫描所有像素点
    for (; iterator_begin != iterator_end; ++iterator_begin)
    {
        // process each pixel.
        (*iterator_begin)[0] &= mask;
        (*iterator_begin)[0] += div2;
        (*iterator_begin)[1] &= mask;
        (*iterator_begin)[1] += div2;
        (*iterator_begin)[2] &= mask;
        (*iterator_begin)[2] += div2;
    }
}

// Version 13
// uses ierators from Mat_
void colorReduce11(cv::Mat image, int div = 64)
{
    // 利用 迭代器 进行访问图像像素
    cv::Mat_<cv::Vec3b> cimage = image;
    cv::Mat_<cv::Vec3b>::iterator iterator_begin = cimage.begin();
    cv::Mat_<cv::Vec3b>::iterator iterator_end = cimage.end();
    uchar div2 = div >> 1;

    // 扫描所有像素点
    for (; iterator_begin != iterator_end; ++iterator_begin)
    {
        // process each pixel.
        (*iterator_begin)[0] = (*iterator_begin)[0] / div *div + div2;
        (*iterator_begin)[1] = (*iterator_begin)[1] / div *div + div2;
        (*iterator_begin)[2] = (*iterator_begin)[2] / div *div + div2;
    }
}

// Version 14
// uses the at method
void colorReduce12(cv::Mat image, int div = 64)
{
    int num_row = image.rows;
    int num_col = image.cols;
    uchar div2 = div >> 1;

    for (int row_index = 0; row_index < num_row; ++row_index)
    {
        for (int col_index = 0; col_index < num_col; ++col_index)
        {
            // pross each pixel
            image.at<cv::Vec3b>(row_index, col_index)[0] =
                image.at<cv::Vec3b>(row_index, col_index)[0] / div * div + div2;
            image.at<cv::Vec3b>(row_index, col_index)[1] =
                image.at<cv::Vec3b>(row_index, col_index)[1] / div * div + div2;
            image.at<cv::Vec3b>(row_index, col_index)[2] =
                image.at<cv::Vec3b>(row_index, col_index)[2] / div * div + div2;
        }
    }
}

// Version 15
// uses a look up table
void colorReduce13(cv::Mat image, int div = 64)
{
    cv::Mat lookup(1, 256, CV_8U);

    for (int i = 0; i < 256; i++)
    {

        lookup.at<uchar>(i) = i / div * div + div / 2;
    }

    cv::LUT(image, lookup, image);
}

// Version 16
// uses Mat overloaded operators
void colorReduce14(cv::Mat image, int div = 64)
{
    int number = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << number; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1;

    // perform color reduction
    image = (image & cv::Scalar(mask, mask, mask)) + cv::Scalar(div2, div2, div2);
}

// ----------------------------
const unsigned int NTESTS = 15;
const unsigned int NITERATIONS = 10;

// ----------------------------
int main(int argc, char **argv)
{
    // cv::Mat image = cv::imread("./../images/boldt.jpg")
    auto image = cv::imread("./../images/boldt.jpg");
    if (image.empty())
    {
        std::cout << "Error reading image file." << std::endl;
        return 1;
    }
    

    // Time and process the image.
    const int64 start = cv::getTickCount();
    colorReduce(image, 64);
    // Elapsed time in seconds.
    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();

    std::cout << "Duration= " << duration << "secs" << std::endl;
    cv::namedWindow("Image");
    cv::imshow("Image", image);
    cv::waitKey(0);

    // TEST Version
    int64 t[NTESTS], tinit;
    // timer values set to 0
    for (int i = 0; i < NTESTS; i++)
        t[i] = 0;

    cv::Mat images[NTESTS];
    cv::Mat result;

    // the versions to be tested
    typedef void (*FunctionPointer)(cv::Mat, int);
    FunctionPointer functions[NTESTS] = {colorReduce, colorReduce1, colorReduce2, colorReduce3, colorReduce4,
                                         colorReduce5, colorReduce6, colorReduce7, colorReduce8, colorReduce9,
                                         colorReduce10, colorReduce11, colorReduce12, colorReduce13, colorReduce14};
    // repeat the tests several times
    int n = NITERATIONS;
    for (int k = 0; k < n; k++)
    {

        std::cout << k << " of " << n << std::endl;

        // test each version
        for (int c = 0; c < NTESTS; c++)
        {

            images[c] = cv::imread("./../images/boldt.jpg");

            // set timer and call function
            tinit = cv::getTickCount();
            functions[c](images[c], 64);
            t[c] += cv::getTickCount() - tinit;

            std::cout << ".";
        }

        std::cout << std::endl;
    }

    // short description of each function
    std::string descriptions[NTESTS] = {
        "original version:",
        "with dereference operator:",
        "using modulo operator:",
        "using a binary mask:",
        "direct ptr arithmetic:",
        "row size recomputation:",
        "continuous image:",
        "reshape continuous image:",
        "with iterators:",
        "Vec3b iterators:",
        "iterators and mask:",
        "iterators from Mat_:",
        "at method:",
        "overloaded operators:",
        "look-up table:",
    };

    for (int i = 0; i < NTESTS; i++)
    {

        cv::namedWindow(descriptions[i]);
        cv::imshow(descriptions[i], images[i]);
    }

    // print average execution time
    std::cout << std::endl
              << "-------------------------------------------" << std::endl
              << std::endl;
    for (int i = 0; i < NTESTS; i++)
    {

        std::cout << i << ". " << descriptions[i] << 1000. * t[i] / cv::getTickFrequency() / n << "ms" << std::endl;
    }

    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
