#ifndef __IMAGE_RENDER_H__
#define __IMAGE_RENDER_H__

#include <qtmetamacros.h>

#include <QTimer>
#include <QWidget> // Qt标准窗口类头文件

QT_BEGIN_NAMESPACE

// mainwindow.ui 文件中也有一个类叫 MainWindow, 将这个类放到命名空间 Ui 中
namespace Ui {
class FImageRender;
}

QT_END_NAMESPACE

class ImageRender : public QWidget
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : ImageRender(QWidget *parent = nullptr);
    ~ImageRender();

private slots:
    void sl_UpdateImage();

private:
    Ui::FImageRender *ui; // 定义指针指向窗口的 UI 对象
    QTimer           *m_pTimer;
};

#endif /* __IMAGE_RENDER_H__ */