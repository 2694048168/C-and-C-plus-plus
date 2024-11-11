/**
 * @file 12_parallel_image_processing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief Python 和 C++ 中使用并行计算增强图像处理能力
 * *Python 使用 Joblib 库
 * *C++ 使用 OpenMP
 * 
 * 如果使用 Python, 可能听说过全局解释器锁(GIL), 限制了线程的真正并行性.
 * 可能想知道为什么 Python 在这方面会遇到困难(CPython解释器实现的历史原因).
 * Python 的 GIL 确保单个进程中一次只有一个线程运行, 这对于保证安全非常有用,
 * 但严重限制了图像处理等 CPU 密集型任务的性能.
 * 
import os
import cv2
from glob import glob
from joblib import Parallel, delayed

input_folder = 'images/'
output_folder = 'output/'

def convert_to_grayscale(image_path, output_folder):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    output_path = os.path.join(output_folder, os.path.basename(image_path))    
    cv2.imwrite(output_path, gray_img)

if __name__ == "__main__":   
    # get a list of abs path for all .jpg files within input_folder    
    image_files = glob(f"{input_folder}/*.jpg")
    # Parallel processing using Joblib    
    # Joblib将图像处理任务拆分到多个 CPU 核心上。只需设置n_jobs=-1，所有可用核心都会被利用
    Parallel(n_jobs=-1, backend="threading")(delayed(convert_to_grayscale)(image, output_folder) for image in image_files)

 * C++ 可以充分利用多线程的强大功能, 使用 OpenMP 轻松地将任务分配到不同的 CPU 内核上,
 * 以最小的努力实现真正的并行性
 * ? 安装 OpenCV 并设置 OpenMP
 * ? 确保拥有OpenMP(大多数现代编译器[GCC/Clang/MSVC]都具有开箱即用的 OpenMP 支持)
 * ? 要使用 OpenMP 编译 C++ 代码 g++ -fopenmp(fopenmp标志是启用OpenMP的标志)
 * 
 * 通过利用并行计算, 可以显著减少处理大量图像所需的时间.
 * 无论使用 Python 还是 C++, 都可以使用工具来加快工作流程.
 * 
 */

#include <omp.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

void convert_to_grayscale(const std::string &input_path, const std::string &output_folder)
{
    cv::Mat img = cv::imread(input_path);
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    std::string output_path = output_folder + "/" + std::filesystem::path(input_path).filename().string();
    cv::imwrite(output_path, gray_img);
}

int main(int argc, const char **argv)
{
    std::string input_folder  = "images/";
    std::string output_folder = "output/";

    std::vector<std::string> image_files;
    for (const auto &entry : std::filesystem::directory_iterator(input_folder))
    {
        image_files.emplace_back(entry.path().string());
    }

// Parallel processing using OpenMP
#pragma omp parallel for
    for (size_t i = 0; i < image_files.size(); ++i)
    {
        convert_to_grayscale(image_files[i], output_folder);
    }

    return 0;
}
