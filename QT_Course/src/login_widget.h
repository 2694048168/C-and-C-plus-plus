#ifndef __LOGIN_WIDGET_H__
#define __LOGIN_WIDGET_H__

#include <QEvent.h>

#include <QObject>
#include <QWidget> // Qt标准窗口类头文件

QT_BEGIN_NAMESPACE

// mainwindow.ui 文件中也有一个类叫 MainWindow, 将这个类放到命名空间 Ui 中
namespace Ui {
class FLoginWidget;
}

QT_END_NAMESPACE

class LoginWidget : public QWidget
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : LoginWidget(QWidget *parent = nullptr);
    ~LoginWidget();

protected:
    void keyPressEvent(QKeyEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    Ui::FLoginWidget *ui; // 定义指针指向窗口的 UI 对象
};

#endif /* __LOGIN_WIDGET_H__ */