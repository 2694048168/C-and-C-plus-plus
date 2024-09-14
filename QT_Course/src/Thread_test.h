#ifndef __THREAD_TEST_H__
#define __THREAD_TEST_H__

#include <QWidget> // Qt标准窗口类头文件

QT_BEGIN_NAMESPACE

// mainwindow.ui 文件中也有一个类叫 MainWindow, 将这个类放到命名空间 Ui 中
namespace Ui {
class CThreadTest;
}

QT_END_NAMESPACE

class ThreadTest : public QWidget
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : ThreadTest(QWidget *parent = nullptr);
    ~ThreadTest();

private:
    Ui::CThreadTest *ui; // 定义指针指向窗口的 UI 对象
};
#endif // __THREAD_TEST_H__
