#ifndef __MAIN_WINDOW_H__
#define __MAIN_WINDOW_H__

#include <QMainWindow> // Qt标准窗口类头文件

QT_BEGIN_NAMESPACE

// mainwindow.ui 文件中也有一个类叫 MainWindow, 将这个类放到命名空间 Ui 中
namespace Ui {
class CMainWindow;
}

QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    // 重写事件处理器函数
    void closeEvent(QCloseEvent *ev) override;
    void resizeEvent(QResizeEvent *ev) override;

private:
    Ui::CMainWindow *ui; // 定义指针指向窗口的 UI 对象
};
#endif // __MAIN_WINDOW_H__
