#pragma once

#include "ui_VisualPointCould.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkAppendPolyData.h>
#include <vtkCommand.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPlaneSource.h>
#include <vtkSmartPointer.h>

#include <QWidget>
#include <string>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class VisualPointCouldForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class VisualPointCould : public QWidget
{
    Q_OBJECT

public:
    void Run();

    void SetMessageCallback(std::function<void(bool, const std::string &)> callback);

public:
    // Constructor/Destructor
    explicit VisualPointCould(QWidget *parent = nullptr);
    virtual ~VisualPointCould();

public slots:
    void sl_ZeroPlaneSwitch();

private:
    void InternalRun(bool enableZeroPlane = false, int numPoints = 5000);
    // 生成点云数据, 即扫描物体的真实数据
    void GenerateSamplePointCloud(int numPoints = 10000, bool enableZeroPlane = true);
    // 创建网格平面作为参考平面
    void CreateReferencePlane(double xSize = 100.0, double ySize = 80.0, int xResolution = 20, int yResolution = 20);

private:
    // Designer form
    Ui::VisualPointCouldForm *ui = nullptr;

    QVTKOpenGLNativeWidget                       *mpVtkRenderWidget = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow    = nullptr;
    vtkSmartPointer<vtkOrientationMarkerWidget>   mpAxesWidget      = nullptr;

    std::function<void(bool, const std::string &)> mCallbackFunc = nullptr;

    vtkSmartPointer<vtkPolyData>       mpPolyData       = nullptr;
    vtkSmartPointer<vtkPlaneSource>    mpReferencePlane = nullptr;
    vtkSmartPointer<vtkAppendPolyData> mpAppendFilter   = nullptr;
};

// 创建自定义回调来保持坐标轴可见
class AxesWidgetCallback : public vtkCommand
{
public:
    static AxesWidgetCallback *New()
    {
        return new AxesWidgetCallback;
    }

    void SetWidget(vtkOrientationMarkerWidget *widget)
    {
        m_Widget = widget;
    }

    virtual void Execute(vtkObject *caller, unsigned long eventId, void *callData)
    {
        if (m_Widget && !m_Widget->GetEnabled())
        {
            m_Widget->SetEnabled(1);
            m_Widget->InteractiveOn();
        }
    }

private:
    vtkOrientationMarkerWidget *m_Widget = nullptr;
};

} // namespace Ithaca