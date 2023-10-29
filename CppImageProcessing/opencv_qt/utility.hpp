/**
 * @file utility.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-10-29
 *
 * @copyright Copyright (c) 2023
 *
 */

// reference from:
// https://qtandopencv.blogspot.com/2013/08/how-to-convert-between-cvmat-and-qimage.html
/**
 * @brief Convert between cv::mat and QImage correctly
 * 
 * This post will talk about how to convert between cv::Mat and QImage.
 * The solution is far from perfect and need to refine, 
 * please give me some helps if you know how to make the codes become better. 
 * Conversion between QImage and cv::Mat have some traps to deal with, 
 * I will write the codes step by step and explain why do I develop it like that, 
 * what kind of traps I want to deal with.
 * 
 */

#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

// #include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/core/mat.hpp>

#include <QImage>

namespace WeiLi::Utility {

/* Step 1. Convert cv::Mat to QImage without copy(share the same buffer)

The trap at here is mat.step, the buffer of the cv::Mat 
may do some padding(factor of 4) on each row for the sake of speed. 
If you just wrote
----------------------------------------------------------------- 
QImage mat_to_qimage_ref(cv::Mat &mat, QImage::Format format)
{
  return QImage(mat.data, mat.cols, mat.rows, format);
}
----------------------------------------------------------------- 
The results may work or may not work because the bytes of 
each row(we call it stride) may vary, vary or not is depends on the image.
To prevent this kind of tragedy, 
we need to tell the QImage how many bytes per row by mat.step.
----------------------------------------------------------------- 
QImage mat_to_qimage_ref(cv::Mat &mat, QImage::Format format)
{
    return QImage(mat.data, mat.cols, mat.rows, mat.step, format);
}
--------------------------------------------------------------- */

/* Step 2. Convert cv::Mat to QImage by copy

 !Warning--this is dangerous
-------------------------------------------------
QImage mat_to_qimage_cpy(mat.data, mat.cols, mat.rows, 
                         QImage::Format_ARGB32)
{
   for (int i = 0;i != image.height(); ++i) {
     memcpy(image.scanline(i), mat.ptr(i), mat.step);
   }
}
-------------------------------------------------
This codes looks unharm at first, 
you could found this kind of suggestion from many forums, 
but this answer is partially work only.
It create an empty mat, and copy the data of the mat to the QImage by memcpy,
fast and low calorie?
Wrong, because we didn't specify how many bytes per row when we construct the QImage.
The correct solution should be,
Let the library tackle the problem for us, we don't need to copy the buffer by ourselves.
-------------------------------------------------
QImage mat_to_qimage_cpy(const cv::Mat &mat, QImage::Format format)
{
    return QImage(mat.data, mat.cols, mat.rows, mat.step, format).copy();
}
-------------------------------------------------------------------- */

/* Step 3. Convert QImage to cv::Mat without copy(share the same buffer)

After you know how to convert cv::Mat to QImage, 
convert from QImage to cv::Mat is trivial, 
same concepts with different api.
----------------------------------------------------
cv::Mat qimage_to_mat_ref(QImage &img, int format)
{
    return cv::Mat(img.height(), img.width(), format, img.bits(), img.bytesPerLine());
}
---------------------------------------------------- */

/* Step 4. Convert QImage to cv::Mat by copy

I believe many c++ programmers would feel uncomfortable 
when you find out I used const_cast to cast the constness,
believe it or not, I have the same feel as yours, 
but I can't find out a better way to copy the buffer.
--------------------------------------------------------
cv::Mat qimage_to_mat_cpy(const QImage &img, int format)
{
    return cv::Mat(img.height(), img.width(), format, const_cast<uchar *>(img.bits()), img.bytesPerLine()).clone();
}
-------------------------------------------------------- */

/* ======== Other traps ========
Do we deal with all of the hassles already?
Not yet, the default channels of the openCV is BGR but the QImage is RGB, 
that means you need to convert the channels from RGB to BGR 
when convert between three channels image and vice versa.
In QImage, you could call "rgbSwapped()"; 
In cv::Mat, you could call cv::cvtColor to convert the channels.

======== Final results ========
It is a little bit annoying to write the conversion codes every-time,
so I do some simple encapsulation on it,

*These functions do not handles all of the image types between QImage and cv::Mat, 
but it should work on most of the images.
Take it if you like and do whatever you want.
----------------------------------------------- */

QImage mat_to_qimage_cpy(const cv::Mat &mat, bool swap = true);

QImage mat_to_qimage_ref(cv::Mat &mat, bool swap = true);

cv::Mat qimage_to_mat_cpy(const QImage &img, bool swap = true);

cv::Mat qimage_to_mat_ref(QImage &img, bool swap = true);

} // namespace WeiLi::Utility

#endif /* __UTILITY_HPP__ */