#include "QtGLImage.h"
#include <filesystem>
#include "ThreadShowImage.h"

QtGLImage::QtGLImage(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QtGLImageClass())
{
    ui->setupUi(this);

    sl_setImgPath();

    // 将widget控件作为绘制窗口
    glImage = new GL_Image(ui->widget);
    glImage->setFixedSize(ui->widget->size());

    //connect(&timer, SIGNAL(timeout()), this, SLOT(slotTimeOut()));
    //timer.setTimerType(Qt::PreciseTimer);
    //timer.start(100);
    m_pThreadShowImg = new ThreadShowImage;
    m_pThreadShowImg->start();

    //connect(ui->btn_start, &QPushButton::clicked, this, &QtGLImage::sl_setImgPath);
    connect(m_pThreadShowImg, &ThreadShowImage::currentImg, this, &QtGLImage::sl_UpdateShowImage);

}

QtGLImage::~QtGLImage()
{
    delete ui;
}

void QtGLImage::sl_setImgPath()
{
    m_imgPath = R"(D:\Development\GitRepository\C-and-C-plus-plus\QT_Learning\QtGLImage\images/)";

    // 遍历目录下的所有文件
    for (const auto& entry : std::filesystem::directory_iterator(m_imgPath))
    {
        // 如果是文件，则输出文件名
        /*if (entry.is_regular_file())
        {
            m_imgFilelist.push_back(entry.path().filename().string());
        }*/

        // add file path
        if (entry.path().extension() == ".png")
        {
            m_imgFilelist.push_back(m_imgPath + entry.path().filename().string());
        }
    }
}

void QtGLImage::sl_UpdateShowImage(QImage image)
{
    //qimage加载的颜色通道顺序和opengl显示的颜色通道顺序不一致,调换R通道和B通道
    QImage rgba = image.rgbSwapped();
    glImage->setImageData(rgba.bits(), rgba.width(), rgba.height());
    //窗口重绘，repaint会调用paintEvent函数，paintEvent会调用paintGL函数实现重绘
    glImage->repaint();
    glImage->update();
}


// 主界面开启定时器，在界面循环显示4个方向的图片
void QtGLImage::slotTimeOut()
{
    for (const auto& filename : m_imgFilelist)
    {
        QImage image(QString(filename.c_str()));
        //qimage加载的颜色通道顺序和opengl显示的颜色通道顺序不一致,调换R通道和B通道
        QImage rgba = image.rgbSwapped();
        glImage->setImageData(rgba.bits(), rgba.width(), rgba.height());
        //窗口重绘，repaint会调用paintEvent函数，paintEvent会调用paintGL函数实现重绘
        glImage->repaint();
    }

    //char imageName[100];
    // 需要修改为自己的图片路径
    //sprintf(imageName, "C:/Users/26940/Pictures/Screenshots/lena%d.jpg", i++ % 4);
}
