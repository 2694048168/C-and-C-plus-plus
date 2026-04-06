/**
 * @file 17_loc_problem_performance.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 擒贼擒王: 问题定位与核心逻辑优化
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 17_loc_problem_performance.exe 17_loc_problem_performance.c
 * clang -o 17_loc_problem_performance.exe 17_loc_problem_performance.c
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// 高精度计时（微秒）
double get_time_us(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
#endif
}

// 图像结构：灰度图，每个像素 0-255
typedef struct
{
    int width, height;
    unsigned char *data; // 行优先存储
} Image;

// 创建图像
Image *create_image(int w, int h)
{
    Image *img = (Image *)malloc(sizeof(Image));
    img->width = w;
    img->height = h;
    img->data = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    return img;
}

// 释放图像
void free_image(Image *img)
{
    if (img)
    {
        free(img->data);
        free(img);
    }
}

// 初始化测试图像（渐变+噪声）
void init_test_image(Image *img)
{
    for (int y = 0; y < img->height; ++y)
    {
        for (int x = 0; x < img->width; ++x)
        {
            img->data[y * img->width + x] = (x + y) % 256;
        }
    }
}

// ========== 朴素实现（未优化，可能有瓶颈） ==========
void blur_naive(const Image *src, Image *dst)
{
    int w = src->width, h = src->height;
    for (int y = 1; y < h - 1; ++y)
    {
        for (int x = 1; x < w - 1; ++x)
        {
            int sum = 0;
            // 3x3 邻域求和（热点循环的核心）
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    sum += src->data[(y + dy) * w + (x + dx)];
                }
            }
            dst->data[y * w + x] = sum / 9;
        }
    }
}

// ========== 优化版本：擒贼擒王，针对核心循环优化 ==========
// 优化点1：消除内部循环，手工展开3x3邻域
// 优化点2：减少重复计算行指针
// 优化点3：使用局部变量
void blur_optimized(const Image *src, Image *dst)
{
    int w = src->width, h = src->height;
    const unsigned char *src_data = src->data;
    unsigned char *dst_data = dst->data;

    for (int y = 1; y < h - 1; ++y)
    {
        // 提前计算上下三行的起始指针
        const unsigned char *row_up = src_data + (y - 1) * w;
        const unsigned char *row_mid = src_data + y * w;
        const unsigned char *row_down = src_data + (y + 1) * w;
        unsigned char *dst_row = dst_data + y * w;

        for (int x = 1; x < w - 1; ++x)
        {
            // 手工展开3x3求和（编译器可能自动向量化）
            int sum = row_up[x - 1] + row_up[x] + row_up[x + 1] + row_mid[x - 1] + row_mid[x] + row_mid[x + 1] +
                      row_down[x - 1] + row_down[x] + row_down[x + 1];
            dst_row[x] = sum / 9;
        }
    }
}

// 验证两个结果是否一致（误差容忍1）
int compare_images(const Image *a, const Image *b)
{
    if (a->width != b->width || a->height != b->height)
        return 0;
    for (int i = 0; i < a->width * a->height; ++i)
    {
        if (abs(a->data[i] - b->data[i]) > 1)
            return 0;
    }
    return 1;
}

int main(void)
{
    SetConsoleOutputCP(65001);

    const int WIDTH = 1920;
    const int HEIGHT = 1080;

    Image *src = create_image(WIDTH, HEIGHT);
    Image *dst1 = create_image(WIDTH, HEIGHT);
    Image *dst2 = create_image(WIDTH, HEIGHT);
    init_test_image(src);

    printf("擒贼擒王：图像模糊算法性能优化\n");
    printf("图像尺寸: %dx%d\n\n", WIDTH, HEIGHT);

    // 1. 测量朴素版本
    double start = get_time_us();
    blur_naive(src, dst1);
    double elapsed_naive = (get_time_us() - start) / 1000.0;
    printf("朴素版本耗时: %.2f ms\n", elapsed_naive);

    // 2. 测量优化版本
    start = get_time_us();
    blur_optimized(src, dst2);
    double elapsed_opt = (get_time_us() - start) / 1000.0;
    printf("优化版本耗时: %.2f ms\n", elapsed_opt);

    // 3. 验证正确性
    if (compare_images(dst1, dst2))
    {
        printf("结果一致，优化有效\n");
    }
    else
    {
        printf("结果不一致，请检查\n");
    }

    printf("\n擒贼擒王策略：\n");
    printf("- 使用 profiler 定位热点（本例中的双层内循环）\n");
    printf("- 核心优化：循环展开 + 减少重复寻址 + 手工预取行指针\n");
    printf("- 性能提升: %.1fx\n", elapsed_naive / elapsed_opt);
    printf("- 现代C特性：restrict、静态数组、编译器优化选项 -O2 -march=native\n");

    free_image(src);
    free_image(dst1);
    free_image(dst2);
    return 0;
}
