#include "ThreadShowImage.h"
#include <filesystem>
//#include <Windows.h>

ThreadShowImage::ThreadShowImage(QObject* parent) :QThread(parent)
{
    m_imgPath = R"(D:\Development\GitRepository\C-and-C-plus-plus\QT_Learning\QtGLImage\images/)";
}

void ThreadShowImage::run()
{
    //qDebug() << "当前线程对象的地址: " << QThread::currentThread();


    // 遍历目录下的所有文件
    for (const auto& entry : std::filesystem::directory_iterator(m_imgPath))
    {
        if (entry.path().extension() == ".png")
        {
            QImage img((m_imgPath + entry.path().filename().string()).c_str());
            emit currentImg(img);
            //msleep(500);
        }
        //msleep(500);
    }
}

