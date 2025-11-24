#include "StanfordBunny.h"

#include <QVTKInteractor.h>
#include <vtkAutoInit.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObject.h>
#include <vtkPointPicker.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTextProperty.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#include <QFileDialog>
#include <QString>

namespace Ithaca {

// 键盘事件回调函数
void KeyPressCallback(vtkObject *caller, unsigned long eventId, void *clientData, void *callData)
{
    QVTKInteractor *interactor = static_cast<QVTKInteractor *>(caller);
    std::string     key        = interactor->GetKeySym();

    vtkRenderer *renderer = static_cast<vtkRenderer *>(clientData);
    vtkActor    *actor    = static_cast<vtkActor *>(renderer->GetActors()->GetLastActor());

    if (key == "r" || key == "R")
    {
        // 重置相机视角
        renderer->GetActiveCamera()->SetPosition(0, 0, 10);
        renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
        renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
        renderer->ResetCamera();
        if (StanfordBunny::mpTextActor)
        {
            StanfordBunny::mpTextActor->SetInput("相机视角已重置");
            //StanfordBunny::mpTextActor->SetInput("Camera ViewPoint Reset");
        }
    }
    else if (key == "w" || key == "W")
    {
        // 切换线框/表面显示模式
        if (actor->GetProperty()->GetRepresentation() == VTK_SURFACE)
        {
            actor->GetProperty()->SetRepresentationToWireframe();
            if (StanfordBunny::mpTextActor)
            {
                StanfordBunny::mpTextActor->SetInput("线框模式");
                //StanfordBunny::mpTextActor->SetInput("Wireframe Model");
            }
        }
        else
        {
            actor->GetProperty()->SetRepresentationToSurface();
            if (StanfordBunny::mpTextActor)
            {
                StanfordBunny::mpTextActor->SetInput("表面模式");
                //StanfordBunny::mpTextActor->SetInput("Surface Mode");
            }
        }
    }
    else if (key == "1")
    {
        // 设置模型颜色为红色
        actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
        if (StanfordBunny::mpTextActor)
        {
            StanfordBunny::mpTextActor->SetInput("颜色: 红色");
            //StanfordBunny::mpTextActor->SetInput("Color: Red");
        }
    }
    else if (key == "2")
    {
        // 设置模型颜色为原始颜色
        actor->GetProperty()->SetColor(0.9, 0.7, 0.6); // 类似兔子颜色的肤色
        if (StanfordBunny::mpTextActor)
        {
            StanfordBunny::mpTextActor->SetInput("颜色: 原始");
            //StanfordBunny::mpTextActor->SetInput("Color: Original");
        }
    }
    else if (key == "3")
    {
        // 设置模型颜色为蓝色
        actor->GetProperty()->SetColor(0.0, 0.0, 1.0);
        if (StanfordBunny::mpTextActor)
        {
            StanfordBunny::mpTextActor->SetInput("颜色: 蓝色");
            //StanfordBunny::mpTextActor->SetInput("Color: Blue");
        }
    }

    interactor->GetRenderWindow()->Render();
}

// 鼠标点击事件回调函数
void PickCallback(vtkObject *caller, unsigned long eventId, void *clientData, void *callData)
{
    //vtkRenderWindowInteractor *interactor = static_cast<vtkRenderWindowInteractor *>(caller);
    QVTKInteractor *interactor = static_cast<QVTKInteractor *>(caller);

    if (interactor->GetKeyCode() == 'p' || interactor->GetKeyCode() == 'P')
    {
        int *clickPos = interactor->GetEventPosition();

        vtkPointPicker *picker = vtkPointPicker::New();
        picker->SetTolerance(0.001);

        vtkRenderer *renderer = static_cast<vtkRenderer *>(clientData);
        int          picked   = picker->Pick(clickPos[0], clickPos[1], 0, renderer);

        if (picked)
        {
            double           *pos = picker->GetPickPosition();
            std::stringstream ss;
            ss << "拾取点坐标: (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")";
            //ss << "Slect Point Coord: (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")";
            if (StanfordBunny::mpTextActor)
            {
                StanfordBunny::mpTextActor->SetInput(ss.str().c_str());
            }
        }

        picker->Delete();
        interactor->GetRenderWindow()->Render();
    }
}

// ----------------------------------------------------
vtkSmartPointer<vtkTextActor> StanfordBunny::mpTextActor = vtkSmartPointer<vtkTextActor>::New();

StanfordBunny::StanfordBunny(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::StanfordBunnyForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpReader       = vtkSmartPointer<vtkPLYReader>::New();
    // 请确保 bunny.ply 文件在正确路径下，或使用绝对路径
    mFilepath
        = R"(D:/Development/GitRepository/C-and-C-plus-plus/VTK_Learning/models/bunny/reconstruction/bun_zipper.ply)";

    // 设置字体族和字体文件, 支持中文显示 & 使用 UTF-8 编码字符串
    // 在 VTK 中可能需要设置字体大小不小于 18 才能正常显示中文
    mpTextActor->GetTextProperty()->SetFontFamily(VTK_FONT_FILE);               // 使用字体文件
    mpTextActor->GetTextProperty()->SetFontFile("C:/Windows/Fonts/simhei.ttf"); // 指定中文字体路径
    mpTextActor->GetTextProperty()->SetFontSize(20);                            // 尝试设置较大字号

    ui->lineEdit_loadFilepath->setText(mFilepath.c_str());
    QObject::connect(ui->btn_loadFilepath, &QPushButton::clicked, this, &StanfordBunny::sl_LoadModelFilepath);
}

StanfordBunny::~StanfordBunny()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void StanfordBunny::Run()
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);
    mpVtkRenderWidget->setRenderWindow(mpRenderWindow);

    mpReader->SetFileName(mFilepath.c_str());
    // 检查文件是否成功加载
    mpReader->Update();
    if (mpReader->GetErrorCode() != 0)
    {
        if (mCallbackFunc)
            mCallbackFunc(false, "错误: 无法加载PLY文件 " + mFilepath + " 请确保文件存在且路径正确");
        return;
    }
    if (mCallbackFunc)
    {
        auto message_ = "成功加载模型，顶点数: " + std::to_string(mpReader->GetOutput()->GetNumberOfPoints());
        mCallbackFunc(true, message_);
    }

    // 2. 创建映射器（Mapper）
    auto pMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pMapper->SetInputConnection(mpReader->GetOutputPort());
    pMapper->ScalarVisibilityOff(); // 不使用标量数据着色

    // 3. 创建演员（Actor）
    auto pActor = vtkSmartPointer<vtkActor>::New();
    pActor->SetMapper(pMapper);
    // 设置演员属性
    pActor->GetProperty()->SetColor(0.9, 0.7, 0.6); // 设置颜色（类似兔子颜色）
    pActor->GetProperty()->SetSpecular(0.3);        // 设置高光强度
    pActor->GetProperty()->SetSpecularPower(20);    // 设置高光指数

    // 4. 创建渲染器（Renderer）并添加演员
    auto pRenderer = vtkSmartPointer<vtkRenderer>::New();
    pRenderer->AddActor(pActor);
    //pRenderer->SetBackground(0.1, 0.2, 0.4); // 设置背景颜色（深蓝色）
    pRenderer->SetBackground(0.8, 0.8, 0.8); // 设置背景颜色（浅灰色）
    pRenderer->ResetCamera();                // 重置相机以确保所有演员可见

    // 5. 创建渲染窗口（RenderWindow）
    mpRenderWindow->AddRenderer(pRenderer);

    // 6. 获取Qt集成的交互器并设置交互样式
    auto pInteractor = mpVtkRenderWidget->interactor();
    auto pStyle      = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    pInteractor->SetInteractorStyle(pStyle);

    // 7. 创建信息文本显示
    mpTextActor->SetInput("按 R:重置视角, W:切换线框, 1/2/3:改变颜色, P+点击:拾取点");
    //mpTextActor->SetInput("R:Reset ViewPoint, W:Switch Wireframe Model, 1/2/3:Color Select, P+Click:Selct Point");
    mpTextActor->GetTextProperty()->SetFontSize(16);
    mpTextActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0); // 白色文字
    mpTextActor->SetPosition(10, 10);
    pRenderer->AddActor2D(mpTextActor);

    // 8. 设置回调函数
    auto pKeyPressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    pKeyPressCallback->SetCallback(KeyPressCallback);
    pKeyPressCallback->SetClientData(pRenderer);
    pInteractor->AddObserver(vtkCommand::KeyPressEvent, pKeyPressCallback);

    auto pPickCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    pPickCallback->SetCallback(PickCallback);
    pPickCallback->SetClientData(pRenderer);
    pInteractor->AddObserver(vtkCommand::LeftButtonPressEvent, pPickCallback);

    // 9. 触发渲染（Qt会自动管理渲染循环）
    mpRenderWindow->Render();
}

void StanfordBunny::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mCallbackFunc = callback;
}

void StanfordBunny::sl_LoadModelFilepath()
{
    // 设置文件过滤器
    QString filter = "3D模型文件 (";
    filter += "*.ply *.obj *.stl *.vtk *.vtp";
    filter += ");;";
    filter += "PLY文件 (*.ply);;";
    filter += "OBJ文件 (*.obj);;";
    filter += "STL文件 (*.stl);;";
    filter += "VTK文件 (*.vtk *.vtp)";

    // 打开文件选择对话框
    QString filePath = QFileDialog::getOpenFileName(this, "选择3D模型文件", QDir::homePath(), filter);

    if (filePath.isEmpty())
    {
        if (mCallbackFunc)
            mCallbackFunc(false, "选择3D模型文件 路径为空");
        return;
    }

    mFilepath = filePath.toStdString();
    Run();
}

} // namespace Ithaca
