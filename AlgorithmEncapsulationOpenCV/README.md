## Algorithm Encapsulation OpenCV via Modern C++

### Features
- 常见的数据滤波算法: 
    - 限幅滤波、限幅平均滤波、限幅消抖滤波、
    - 中位值滤波、中位值平均滤波、
    - 递推平均滤波、加权递推拼接滤波、算数平均滤波
    - 消抖滤波、一阶滞后滤波、低通滤波
- 渲染直线、矩形、圆形、中英文字等图元, 图像保存
- 机器视觉中 Blob 常用检测算法
- OpenCV实现纺织物缺陷检测: 脏污、油渍、线条破损
- 计算机图形学: 线段的交点计算及实现
- 计算机图形学: 计算点与线段的距离、投影和位置关系
- 计算机图形学: 判断点是否在多边形内部
- 三维空间中的向量与坐标系变换


### sub-pixel 
> 亚像素并不是实际存在的物理单元, 而是通过数学方法和图像处理算法在像素之间进行插值和估算得到的更精细的位置或度量. 从物理角度来看, 图像是对真实世界连续场景的采样和量化; 像素的大小和位置是由图像传感器的物理特性和采样频率决定的; 但真实世界中的物体特征和位置往往是连续变化的, 像素的离散性可能导致信息的丢失或精度的降低.

亚像素技术的实现通常基于以下几种常见的方法:
- 基于梯度的方法: 计算图像在水平和垂直方向上的梯度, 通过梯度的变化来更精确地确定边缘或特征的位置;
    - 例如在边缘处, 梯度值会发生显著变化, 通过分析梯度的分布, 可以估计出边缘在像素内部的更准确位置.
- 灰度重心法: 对于具有一定灰度分布的区域, 通过计算其灰度的重心来确定更精确的位置;
    - 这种方法适用于具有均匀灰度分布的特征.
- 曲线拟合方法: 对像素附近的灰度值进行曲线拟合, 如使用多项式曲线或其他函数来拟合灰度的变化,然后求解曲线的极值点或特定位置来获得亚像素精度的结果.

### 工业机器视觉中图像质量评估: 均匀性、对比度、分辨率与清晰度
- 均匀性 Uniformity
    - 均匀性指图像中亮度分布的平衡程度, 要求光线分布均匀且无明显的高光、阴影或暗区
    - 光照不均是导致图像质量下降的常见问题, 会严重干扰目标的边缘检测和区域分析
    - Good Uniformity Image
        - 光照分布均匀, 无突出的亮斑或阴影
        - 灰度值平滑分布, 目标与背景之间形成良好的分离
        - 环境光线对图像影响较小, 细节和纹理清晰可见
    - Poor Uniformity Image
        - 光照强度变化剧烈, 存在高光区域或深色阴影
        - 背景或目标区域中部分细节由于光照不均而模糊或丢失
        - 灰度值在局部集中, 导致对比度不足或细节丢失
    - Optimization Methods
        - 使用漫反射光源或环形光源, 减轻阴影与反光
        - 调整光源角度, 使目标表面光照均匀
        - 增加扩散板或使用多光源系统, 降低局部过亮或过暗的问题
- 对比度 Contrast
    - 对比度是图像目标与背景之间亮度差异的体现, 反映灰度分布的动态范围
    - 对比度的高低直接决定目标区域是否能够与背景区分开来
    - Good Uniformity Image
        - 目标与背景有明显的亮度差, 易于区分
        - 灰度分布均匀而广泛, 层次丰富
        - 目标边缘和细节清晰, 利于后续特征提取
    - Poor Uniformity Image
        - 目标与背景亮度接近, 难以检测到目标区域
        - 灰度值范围狭窄, 表现为图像整体灰蒙蒙一片, 缺乏层次感
        - 对比度过高导致某些区域过曝或过暗, 丢失细节
    - Optimization Methods
        - 通过调整光源的波长(如红外光或蓝光)增强目标与背景的亮度差异
        - 调整相机的曝光时间和增益, 适应现场光照条件
        - 应用图像增强技术, 如直方图均衡化、对比度拉伸或伽马校正
- 分辨率 Resolution
    - 分辨率是图像中可分辨最小细节的能力, 由成像设备的传感器分辨率和镜头的光学质量决定
    - 分辨率不足或过高都会对图像质量和处理效率造成影响
    - Good Uniformity Image
        - 图像分辨率适配任务需求, 所有关键特征清晰可见
        - 小目标或微小特征(如孔洞、线纹)能够被准确捕捉
        - 图像细节真实无失真, 无过度像素化现象
    - Poor Uniformity Image
        - 分辨率过低导致细节模糊, 特征不可辨识
        - 像素化严重, 边缘呈锯齿状, 影响测量精度
        - 分辨率过高造成计算资源浪费而无实际收益
    - Optimization Methods
        - 根据目标尺寸和检测精度需求选择适合的相机分辨率
        - 确保镜头解析力与相机传感器匹配, 避免过采样或欠采样
        - 在无法更换硬件时, 适当调整拍摄距离以提高有效分辨率
- 清晰度 Sharpness
    - 清晰度是衡量图像中目标边缘和细节锐利程度的指标, 
    - 受焦距准确性、成像运动模糊和镜头光学性能等因素影响
    - Good Uniformity Image
        - 图像边缘锐利, 目标与背景的边界清晰
        - 细小特征(如纹理、文字)清晰可辨, 且没有模糊
        - 无明显的运动模糊或离焦现象
    - Poor Uniformity Image
        - 图像模糊, 目标细节消失, 边缘呈现过渡渐变
        - 对焦不准或成像运动模糊导致无法准确提取特征
        - 过度锐化产生伪影, 干扰后续处理
    - Optimization Methods
        - 调整镜头的对焦位置, 确保焦点精确落在目标平面上
        - 缩短曝光时间以减少运动模糊, 或使用稳定装置固定拍摄设备
        - 采用去模糊和边缘增强算法进行后期处理
- 为确保获得优质图像(High-Quality Images), 可以从以下几点入手:
    - 合理布置光源, 优化光照条件
    - 调整相机参数, 确保对比度适中且分辨率匹配需求
    - 定期校准和优化镜头及相机焦距
    - 使用后期图像增强算法对图像进行适度优化
    - 通过严格控制这四个关键要素, 机器视觉系统能够在目标检测、
    特征提取和分析中实现高精度和高效率, 最终提升工业生产和检测的自动化水平
