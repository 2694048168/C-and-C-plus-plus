#include <QVTKOpenGLNativeWidget.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QWidget>

// ---------------------------------
int main(int argc, char *argv[])
{
    // 1. 首先创建 QApplication
    QApplication app(argc, argv);

    // 2. 创建 Qt 主窗口
    QMainWindow mainWindow;
    mainWindow.setWindowTitle("VTK with Qt");
    mainWindow.resize(800, 600);

    // 3. 创建中央部件和布局
    QWidget     *centralWidget = new QWidget();
    QVBoxLayout *layout        = new QVBoxLayout(centralWidget);

    // 4. 创建 VTK widget
    QVTKOpenGLNativeWidget *vtkWidget = new QVTKOpenGLNativeWidget();
    layout->addWidget(vtkWidget);

    mainWindow.setCentralWidget(centralWidget);

    // 5. 设置 VTK 可视化管线
    vtkNew<vtkSphereSource> sphere;
    sphere->SetRadius(1.0);
    sphere->SetThetaResolution(50);
    sphere->SetPhiResolution(50);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(sphere->GetOutputPort());

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 0.0, 0.0); // 红色

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.2, 0.4); // 蓝色背景

    vtkWidget->renderWindow()->AddRenderer(renderer);

    // 6. 显示窗口并启动事件循环
    mainWindow.show();

    return app.exec();
}
