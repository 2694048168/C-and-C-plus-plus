/**
 * @brief OpenCV图像轮廓检测与绘制
 * 计算机视觉和图像处理中, 轮廓是沿着物体边界连接具有相同强度或颜色的连续点的曲线或边界.
 * 轮廓表示图像中物体的轮廓, 图像轮廓绘制是检测和提取图像中物体的边界或轮廓的过程.
 * 本质上, 图像轮廓绘制涉及识别形成连续曲线的相似强度或颜色的点,从而勾勒出物体的形状.
 * 轮廓在许多图像处理和计算机视觉任务中至关重要,因为它们可以有效地分割、
 * 分析和提取图像中各种对象的特征.
 * 图像轮廓通常用于简化图像,只关注重要的结构元素（即对象边界）,
 * 同时忽略不相关的细节,如纹理或对象内的细微变化.
 * 
 * * 图像轮廓绘制通常涉及步骤：1.预处理; 2.二进制转换; 3.轮廓检测; 4.轮廓绘制;
 * ?应用阈值或将灰度图像转换为二进制图像, 在二进制图像中,
 * ?对象由白色像素（前景）表示, 背景由黑色像素表示,
 * ?阈值处理有助于根据像素强度突出显示感兴趣的区域,
 * ?而边缘检测则可以强调发生急剧强度变化的边界.
 * 
 * *轮廓由点列表表示，每个点代表沿对象边界的像素位置.
 * cv::findContours(...);
 * ?1. image-图像(输入图像), 从中找到轮廓的源图像,必须是二进制图像,
 * ?这意味着它应该只包含两个像素值,通常为 0(黑色)表示背景, 255(白色)表示对象,
 * ?通常,这是通过使用阈值或边缘检测来创建合适的二进制图像来实现的,
 * ?请注意 findContours() 会修改输入图像, 因此如果需要保留原始图像,通常建议传递副本.
 * 
 * ?2. mode-模式(轮廓检索模式),此参数指定如何检索轮廓,
 * ?它控制轮廓的层次结构,对于嵌套或重叠对象特别有用,
 * ? cv::RETR_EXTERNAL、cv::RETR_LIST、cv::RETR_CCOMP、cv::RETR_TREE
 * 
 * ?3. method-方法(轮廓近似法),此参数控制轮廓点的近似方法,
 * ?本质上影响精度和用于表示轮廓的点数,
 * ? cv::CHAIN_APPROX_NONE、cv::CHAIN_APPROX_SIMPLE、
 * ? cv::CHAIN_APPROX_TC89_L1、cv::CHAIN_APPROX_TC89_KCOS
 * 
 * ?4. contours-轮廓(输出参数), 这是所有检测到的轮廓的输出列表,
 * ?每个单独的轮廓都表示为定义轮廓边界的点(x，y 坐标)的数组,
 * ?轮廓通常存储为包含一个或多个轮廓的列表.
 * !std::vector<std::vector<cv::Point>> contours;
 * 
 * ?5. hierarchy-层次结构(输出参数),这是一个可选的输出参数,
 * ?存储了轮廓之间的层级关系信息,
 * ?每条轮廓有四条信息[Next, Previous, First_Child, Parent];
 * ?下一个：同一层次结构的下一个轮廓的索引;
 * ?上一个：同一层次结构中前一个轮廓的索引;
 * ?First_Child：第一个子轮廓的索引;
 * ?父级：父轮廓的索引;
 * ?如果值为 -1, 则表示没有相应的轮廓(没有父轮廓或没有子轮廓).
 * !std::vector<cv::Vec4i>              hierarchy;
 * 
 * ?6. offset-偏移量(可选),此参数用于将轮廓点移动一定偏移量(x 和 y 值的元组),
 * ?如果处理子图像并需要将检测到的轮廓移回以匹配其在原始图像中的位置,这将非常有用,
 * 
 * *--------------------------------------------------------
 * OpenCV 中有四种主要的轮廓检索模式:
 * *cv2.RETR_EXTERNAL
 * 此模式用于仅检索最外层轮廓,它会忽略包含在另一个轮廓(轮廓的层次结构级别)内的任何轮廓,
 * 它实际上只考虑外部边界, 而忽略内部嵌套轮廓.
 * 如果只对物体的外部边界感兴趣,例如检测桌子上的硬币或将物体与背景隔离,此模式非常有用.
 * 当嵌套轮廓与分析无关时, 此模式非常有效.
 * 
 * *cv2.RETR_LIST
 * 此模式用于检索图像中的所有轮廓,但不建立它们之间的任何父子关系,
 * 所有轮廓都存储在一个简单的列表中, 此模式本质上使层次结构扁平化,
 * 它不区分外部和内部轮廓,并将每个轮廓视为独立的,没有层次分组.
 * 当不需要有关不同轮廓之间关系的信息而只需要整个轮廓集时,此模式很有用,
 * 例如分析图像中每个对象的形状或边界.
 * 
 * *cv2.RETR_CCOMP
 * 此模式用于检索所有轮廓, 并将它们组织成两级层次结构, 轮廓按层次排列,
 * 层次结构中只有两个级别,第一级表示对象的外部边界,第二级表示对象内部的轮廓.
 * 当需要一个简单的层次结构,并且外部边界和各自的内部对象边界之间有明显区分时,此模式很有用.
 * 
 * *cv2.RETR_TREE
 * 此模式用于检索所有轮廓并将它们组织成完整的层次结构(即树结构),
 * 此模式提供轮廓之间层次关系的最全面表示,包括父轮廓、子轮廓、兄弟轮廓和嵌套轮廓,
 * 所有轮廓都以完整的父子关系组织,这意味着不仅跟踪外部和内部轮廓,
 * 还跟踪这些内部轮廓中存在的任何嵌套,这形成了一个完整的树状层次结构.
 * 此模式适用于需要了解嵌套对象完整结构的复杂图像,例如分析文档中的轮廓或了解重叠形状,
 * 它提供有关轮廓如何相互嵌套的详细信息.
 * 
 * *--------------------------------------------------------
 * OpenCV 的 cv::findContours() 函数中的轮廓近似方法决定了
 * 如何通过近似轮廓点来表示轮廓,这直接影响轮廓表示的存储和精度.
 * 通过使用不同的近似方法,可以控制用于表示轮廓的点数,在内存效率和轮廓精度之间取得平衡.
 * 
 * *cv2.CHAIN_APPROX_NONE
 * 此方法存储轮廓边界上的所有点,通过保留轮廓上的每个点,它提供了轮廓的最详细表示.
 * 由于存储了每个边界点,因此此方法占用的内存最多,并且计算成本很高
 * 适用于需要高度精确的轮廓表示(例如需要精确测量时),
 * 它还可用于涉及轮廓边界上所有点的进一步分析,例如跟踪点或执行变换.
 * 
 * *cv2.CHAIN_APPROX_SIMPLE
 * 该方法通过删除直线上的所有冗余点来压缩轮廓的表示,
 * 它仅存储每个段的端点,从而有效地用更少的点近似轮廓,
 * 这显著减少了内存使用量,同时保留了轮廓的一般形状.
 * 最常用于典型的轮廓检测任务,在这些任务中,需要物体的整体形状,但不需要边界上的每个细节.
 * 适用于检测简单形状(例如矩形、圆形), 这些形状只需要关键点来表示轮廓的结构.
 * 
 * *cv2.CHAIN_APPROX_TC89_L1
 * 该方法使用 Teh-Chin 链近似算法, 与 CHAIN_APPROX_NONE 相比,
 * 该算法使用更少的点来近似轮廓, 这是一种中间方法,试图在轮廓精度和点数之间取得平衡.
 * CHAIN_APPROX_TC89_L1 提供了一种优化的表示,它使用的点比 CHAIN_APPROX_NONE 少,
 * 但可能比 CHAIN_APPROX_SIMPLE 保留更多细节.
 * 当需要优化内存使用但与 CHAIN_APPROX_SIMPLE 相比仍需要更高的精度时很有用.
 * 
 * *cv2.CHAIN_APPROX_TC89_KCOS
 * 与 CHAIN_APPROX_TC89_L1 类似,此方法也使用 Teh-Chin 链近似,
 * 但与 CHAIN_APPROX_TC89_L1 相比,近似程度略有不同,
 * 此方法通常会稍微更积极地减少点数,同时仍能捕捉轮廓的关键特征,
 * 它减少了点数,但与 CHAIN_APPROX_TC89_L1 相比,确切的行为可能略有不同,
 * 当想要比 CHAIN_APPROX_TC89_L1 更积极的轮廓点压缩并且只需要保留最关键的点时,使用此方法.
 * 例如当内存和速度至关重要,并且轮廓不是很复杂时,
 * CHAIN_APPROX_TC89_KCOS 可以在效率和轮廓细节之间提供良好的平衡.
 * 
 * ?轮廓近似方法很重要,因为它有助于控制用于表示轮廓形状的细节级别.
 * 通过选择适当的近似方法,可以在计算效率和给定应用所需的精度之间取得平衡.
 * 当轮廓细节级别会影响性能和准确性时,这种灵活性尤其有用.
 * 
 */

#include "ImageOperator/ContoursDetector.h"

#include <vector>

namespace Ithaca {

bool ContoursDetector::Run(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // 0. 参数检测
    if (srcImg.empty())
        return false;

    // 图像轮廓绘制通常步骤：1.预处理; 2.二进制转换; 3.轮廓检测; 4.轮廓绘制
    // 1. 预处理
    cv::Mat grayImg;
    if (CV_8UC3 == srcImg.type())
    {
        cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
        // gaussion blur to reduce noise
        cv::GaussianBlur(grayImg, grayImg, cv::Size(mControlParameter.blurKernelX, mControlParameter.blurKernelY), 0);
    }
    else
    {
        grayImg = srcImg;
    }

    // 2. 二进制转换
    cv::Mat binaryImg;
    cv::threshold(grayImg, binaryImg, mControlParameter.minThreshold, mControlParameter.maxThreshold,
                  cv::THRESH_BINARY);

    // 3. 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;
    cv::findContours(binaryImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 4. 轮廓绘制
    dstImg = srcImg.clone();
    if (CV_8UC1 == dstImg.type())
    {
        cv::cvtColor(dstImg, dstImg, cv::COLOR_GRAY2BGR);
    }
    cv::drawContours(dstImg, contours, -1, cv::Scalar(0, 255, 0), 1);

    return true;
}

} // namespace Ithaca
