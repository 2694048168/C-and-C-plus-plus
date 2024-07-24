#include "QtOpenGLWidgetsImages.h"

QtOpenGLWidgetsImages::QtOpenGLWidgetsImages(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QtOpenGLWidgetsImagesClass())
{
    ui->setupUi(this);

    // 将widget控件作为绘制窗口
    glImage = new GL_Image(ui->widget);
    glImage->setFixedSize(ui->widget->size());

    connect(&timer, SIGNAL(timeout()), this, SLOT(slotTimeOut()));
    timer.setTimerType(Qt::PreciseTimer);
    timer.start(100);
}

QtOpenGLWidgetsImages::~QtOpenGLWidgetsImages()
{
    delete ui;
}


// 主界面开启定时器，在界面循环显示4个方向的图片
void QtOpenGLWidgetsImages::slotTimeOut()
{
    // 需要修改为自己的图片路径
    auto imageName = R"(D:\Development\GitRepository\ComputerVisionDeepLearning\DigitalImageProcessing\image/lena.jpg)";
    QImage image(imageName);
    QImage rgba = image.rgbSwapped(); //qimage加载的颜色通道顺序和opengl显示的颜色通道顺序不一致,调换R通道和B通道
    glImage->setImageData(rgba.bits(), rgba.width(), rgba.height());
    //glImage->setImageData(image.bits(), image.width(), image.height());
    glImage->repaint(); //窗口重绘，repaint会调用paintEvent函数，paintEvent会调用paintGL函数实现重绘
}
