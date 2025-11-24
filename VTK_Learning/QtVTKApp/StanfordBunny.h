#pragma once

#include "ui_StanfordBunny.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkObject.h>
#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>

#include <QWidget>
#include <string>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class StanfordBunnyForm;
}

QT_END_NAMESPACE

namespace Ithaca {

// 键盘事件回调函数
void KeyPressCallback(vtkObject *caller, unsigned long eventId, void *clientData, void *callData);

// 鼠标点击事件回调函数
void PickCallback(vtkObject *caller, unsigned long eventId, void *clientData, void *callData);

class StanfordBunny : public QWidget
{
    Q_OBJECT

public:
    void Run();

    void SetMessageCallback(std::function<void(bool, const std::string &)> callback);

public:
    // Constructor/Destructor
    explicit StanfordBunny(QWidget *parent = nullptr);
    virtual ~StanfordBunny();

    static vtkSmartPointer<vtkTextActor> mpTextActor;

    //更原生的Qt集成，可以考虑使用Qt的信号槽机制来处理交互，而不是VTK的回调
    //protected:
    //virtual void keyPressEvent(QKeyEvent *event) override;
    //virtual void mousePressEvent(QMouseEvent *event) override;

private:
    // Designer form
    Ui::StanfordBunnyForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;

    vtkSmartPointer<vtkPLYReader> mpReader = nullptr;

    std::function<void(bool, const std::string &)> mCallbackFunc = nullptr;

    std::string mFilepath = "";
};

} // namespace Ithaca