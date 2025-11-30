#include "VisualPointCould.h"

#include <QVTKInteractor.h>
#include <vtkAutoInit.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkVertexGlyphFilter.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#include <random>

namespace Ithaca {

VisualPointCould::VisualPointCould(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::VisualPointCouldForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;
    mpRenderWindow    = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpAxesWidget      = vtkSmartPointer<vtkOrientationMarkerWidget>::New();

    mpPolyData       = vtkSmartPointer<vtkPolyData>::New();
    mpReferencePlane = vtkSmartPointer<vtkPlaneSource>::New();
    mpAppendFilter   = vtkSmartPointer<vtkAppendPolyData>::New();

    QObject::connect(ui->btn_planeSwitch, &QPushButton::clicked, this, &VisualPointCould::sl_ZeroPlaneSwitch);
}

VisualPointCould::~VisualPointCould()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

// 生成点云数据, 即扫描物体的真实数据
void VisualPointCould::GenerateSamplePointCloud(int numPoints, bool enableZeroPlane)
{
    auto pPoints = vtkSmartPointer<vtkPoints>::New();
    auto pColors = vtkSmartPointer<vtkFloatArray>::New();
    pColors->SetName("Depth");
    pColors->SetNumberOfComponents(1);

    double xRange = 50.0;
    double yRange = 30.0;

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> x_dist(-xRange, xRange);
    std::uniform_real_distribution<> y_dist(-yRange, yRange);

    // 1. 构造一个有深度的形状
    for (int i = 0; i < numPoints; ++i)
    {
        double x = x_dist(gen);
        double y = y_dist(gen);
        double z = 10 * sin(x * 0.1) * cos(y * 0.1) + 5 * cos(x * 0.05) * sin(y * 0.05);

        pPoints->InsertNextPoint(x, y, z);
        pColors->InsertNextValue(z);
    }

    // 2. 添加深度为0的参考平面
    if (enableZeroPlane)
    {
        int    planePoints = numPoints / 5; // 平面点数约为原始点云的1/5
        double xPlaneRange = xRange + 10.0;
        double yPlaneRange = yRange + 10.0;

        std::uniform_real_distribution<> plane_x_dist(-xPlaneRange, xPlaneRange); // 平面范围稍大
        std::uniform_real_distribution<> plane_y_dist(-yPlaneRange, yPlaneRange);

        for (int i = 0; i < planePoints; ++i)
        {
            double x = plane_x_dist(gen);
            double y = plane_y_dist(gen);
            double z = 0.0; // 深度为0

            pPoints->InsertNextPoint(x, y, z);
            pColors->InsertNextValue(z);
        }
    }

    mpPolyData->SetPoints(pPoints);
    mpPolyData->GetPointData()->SetScalars(pColors);
}

void VisualPointCould::CreateReferencePlane(double xSize, double ySize, int xResolution, int yResolution)
{
    mpReferencePlane->SetOrigin(-xSize / 2, -ySize / 2, 0.0);
    mpReferencePlane->SetPoint1(xSize / 2, -ySize / 2, 0.0);
    mpReferencePlane->SetPoint2(-xSize / 2, ySize / 2, 0.0);
    mpReferencePlane->SetXResolution(xResolution);
    mpReferencePlane->SetYResolution(yResolution);
    mpReferencePlane->Update();

    // 为平面创建颜色数组（深度为0）
    auto pColors = vtkSmartPointer<vtkFloatArray>::New();
    pColors->SetName("Depth");
    pColors->SetNumberOfComponents(1);

    int numPoints = mpReferencePlane->GetOutput()->GetNumberOfPoints();
    for (int i = 0; i < numPoints; ++i)
    {
        pColors->InsertNextValue(0.0f);
    }
    mpReferencePlane->GetOutput()->GetPointData()->SetScalars(pColors);
}

void VisualPointCould::Run()
{
    InternalRun();
}

void VisualPointCould::InternalRun(bool enableZeroPlane, int numPoints)
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);
    mpVtkRenderWidget->setRenderWindow(mpRenderWindow);

    // Step1. 生成点云数据
    GenerateSamplePointCloud(numPoints, enableZeroPlane);

    // Step2. 通过顶点过滤器,将点(x,y,z)显示为像素点(x,y)
    auto pGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    if (false == enableZeroPlane)
    {
        CreateReferencePlane();
        mpAppendFilter->AddInputData(mpPolyData);
        mpAppendFilter->AddInputData(mpReferencePlane->GetOutput());
        mpAppendFilter->Update();
        pGlyphFilter->SetInputData(mpAppendFilter->GetOutput());
    }
    else
    {
        pGlyphFilter->SetInputData(mpPolyData);
    }
    pGlyphFilter->Update();

    // Step3. 深度信息颜色表
    auto pLutTable = vtkSmartPointer<vtkLookupTable>::New();
    pLutTable->SetHueRange(0.667, 0.0); // blue into red
    pLutTable->SetTableRange(mpPolyData->GetPointData()->GetScalars()->GetRange());
    pLutTable->Build();

    // Step4. Pipeline for Mapper and Actor
    auto pMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pMapper->SetInputConnection(pGlyphFilter->GetOutputPort());
    pMapper->SetScalarRange(mpPolyData->GetPointData()->GetScalars()->GetRange());
    pMapper->SetLookupTable(pLutTable);
    pMapper->SetScalarModeToUsePointData();

    auto pActor = vtkSmartPointer<vtkActor>::New();
    pActor->SetMapper(pMapper);
    pActor->GetProperty()->SetPointSize(2);

    // Step5. 增加坐标轴XYZ
    auto pAxes = vtkSmartPointer<vtkAxesActor>::New();
    pAxes->SetTotalLength(20, 20, 20);
    pAxes->SetShaftTypeToCylinder();
    pAxes->SetCylinderRadius(0.02);
    pAxes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);
    pAxes->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);
    pAxes->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);

    // Step6. 增加 Color legend for Depth info
    auto pScalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    pScalarBar->SetLookupTable(pLutTable);
    pScalarBar->SetTitle("Depth");
    pScalarBar->SetNumberOfLabels(15);
    pScalarBar->SetWidth(0.1);
    pScalarBar->SetHeight(0.7);
    pScalarBar->SetPosition(0.9, 0.25);
    pScalarBar->GetTitleTextProperty()->SetFontSize(4);
    pScalarBar->GetLabelTextProperty()->SetFontSize(8);

    // Step7. Pipeline for renderer and window and Interactor
    auto pRenderer = vtkSmartPointer<vtkRenderer>::New();
    mpRenderWindow->AddRenderer(pRenderer);
    //mpRenderWindow->SetSize(860, 640);
    //mpRenderWindow->SetWindowName("3D Camera Scan Data Visual");

    // 6. 获取Qt集成的交互器并设置交互样式
    auto pInteractor = mpVtkRenderWidget->interactor();
    pInteractor->SetRenderWindow(mpRenderWindow);
    auto pStyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    pInteractor->SetInteractorStyle(pStyle);

    // Setp8. 添加坐标轴小部件
    //auto pAxesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    mpAxesWidget->SetOrientationMarker(pAxes);
    mpAxesWidget->SetInteractor(pInteractor);
    //mpAxesWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
    mpAxesWidget->SetViewport(0.8, 0.0, 1.0, 0.2);
    mpAxesWidget->SetEnabled(1);
    mpAxesWidget->InteractiveOn();

    // 使用回调,强制坐标轴小部件在每次渲染时更新
    auto pAxesCallback = vtkSmartPointer<AxesWidgetCallback>::New();
    pAxesCallback->SetWidget(mpAxesWidget);
    mpRenderWindow->AddObserver(vtkCommand::RenderEvent, pAxesCallback);

    // Setp9. 将所有组件添加到渲染器
    pRenderer->AddActor(pActor);
    pRenderer->AddActor2D(pScalarBar);
    pRenderer->SetBackground(0.1, 0.1, 0.2); // 设置背景色

    // Step10. 设置相机位置
    pRenderer->GetActiveCamera()->SetPosition(100, 100, 100);
    pRenderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    pRenderer->GetActiveCamera()->SetViewUp(0, 0, 1);
    pRenderer->ResetCamera();

    // 8. 创建信息文本显示
    auto pTextActor = vtkSmartPointer<vtkTextActor>::New();
    // 核心设置：指定中文字体文件
    pTextActor->GetTextProperty()->SetFontFamily(VTK_FONT_FILE);
    pTextActor->GetTextProperty()->SetFontFile("C:/Windows/Fonts/simhei.ttf"); // 请确保路径正确
    pTextActor->SetInput(
        "3D扫描数据可视化已启动\n鼠标操作指南:\n  - 左键拖拽: 旋转视图\n  - 右键拖拽: 缩放视图\n  - 中键拖拽: "
        "平移视图");
    pTextActor->GetTextProperty()->SetFontSize(12);
    pTextActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0); // 白色文字
    pTextActor->SetPosition(10, 10);
    pRenderer->AddActor2D(pTextActor);

    // Step11. 开始渲染
    mpRenderWindow->Render();
}

void VisualPointCould::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mCallbackFunc = callback;
}

void VisualPointCould::sl_ZeroPlaneSwitch()
{
    InternalRun(true);
}

} // namespace Ithaca
