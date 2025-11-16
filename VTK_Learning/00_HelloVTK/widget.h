#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

namespace Ui {
class Window;
}

QT_END_NAMESPACE

class WindowWidget : public QMainWindow
{
    Q_OBJECT

public:
    WindowWidget(QWidget *parent = nullptr);
    ~WindowWidget();

private:
    Ui::Window *ui;
};
#endif // MAINWINDOW_H
