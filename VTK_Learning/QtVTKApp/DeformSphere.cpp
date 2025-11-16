#include "DeformSphere.h"

#include <vtkActor.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>

namespace Ithaca {

DeformSphere::DeformSphere(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::DeformSphereForm())
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    // connect the buttons
    QObject::connect(ui->btn_randomizeButton, &QPushButton::released, this, &DeformSphere::Randomize);
}

DeformSphere::~DeformSphere()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void DeformSphere::Run()
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);

    // VTK part.
    mpWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpVtkRenderWidget->setRenderWindow(mpWindow.Get());

    mpSphere = vtkSmartPointer<vtkSphereSource>::New();
    mpSphere->SetRadius(1.0);
    mpSphere->SetThetaResolution(100);
    mpSphere->SetPhiResolution(100);

    mpMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mpMapper->SetInputConnection(mpSphere->GetOutputPort());

    vtkSmartPointer<vtkActor> pActor = vtkSmartPointer<vtkActor>::New();
    pActor->SetMapper(mpMapper);
    pActor->GetProperty()->SetEdgeVisibility(true);
    pActor->GetProperty()->SetRepresentationToSurface();

    vtkSmartPointer<vtkRenderer> pRenderer = vtkSmartPointer<vtkRenderer>::New();
    // 在这里添加背景颜色设置 红色R、绿色G 蓝色B 的分量，取值范围是0.0到1.0
    //pRenderer->SetBackground(0.1, 0.2, 0.4); // 设置为深蓝色 (RGB值范围0-1)
    // 或者使用以下任意一种颜色：
    // pRenderer->SetBackground(1.0, 1.0, 1.0); // 白色
    // pRenderer->SetBackground(0.0, 0.0, 0.0); // 黑色（默认）
    pRenderer->SetBackground(0.2, 0.3, 0.4); // 蓝灰色
    pRenderer->AddActor(pActor);

    mpWindow->AddRenderer(pRenderer);

    // Setup initial status.
    mRandEng = std::mt19937(0);
    Randomize();
}

void DeformSphere::Randomize()
{
    // Generate randomness.
    double randAmp       = 0.2 + ((mRandEng() % 1000) / 1000.0) * 0.2;
    double randThetaFreq = 1.0 + (mRandEng() % 9);
    double randPhiFreq   = 1.0 + (mRandEng() % 9);

    // Extract and prepare data.
    mpSphere->Update();
    vtkSmartPointer<vtkPolyData> newSphere;
    newSphere.TakeReference(mpSphere->GetOutput()->NewInstance());
    newSphere->DeepCopy(mpSphere->GetOutput());
    vtkNew<vtkDoubleArray> height;
    height->SetName("Height");
    height->SetNumberOfComponents(1);
    height->SetNumberOfTuples(newSphere->GetNumberOfPoints());
    newSphere->GetPointData()->AddArray(height);

    // Deform the sphere.
    for (int iP = 0; iP < newSphere->GetNumberOfPoints(); iP++)
    {
        double pt[3] = {0.0};
        newSphere->GetPoint(iP, pt);
        double theta   = std::atan2(pt[1], pt[0]);
        double phi     = std::atan2(pt[2], std::sqrt(std::pow(pt[0], 2) + std::pow(pt[1], 2)));
        double thisAmp = randAmp * std::cos(randThetaFreq * theta) * std::sin(randPhiFreq * phi);
        height->SetValue(iP, thisAmp);
        pt[0] += thisAmp * std::cos(theta) * std::cos(phi);
        pt[1] += thisAmp * std::sin(theta) * std::cos(phi);
        pt[2] += thisAmp * std::sin(phi);
        newSphere->GetPoints()->SetPoint(iP, pt);
    }
    newSphere->GetPointData()->SetScalars(height);

    // Reconfigure the pipeline to take the new deformed sphere.
    mpMapper->SetInputDataObject(newSphere);
    mpMapper->SetScalarModeToUsePointData();
    mpMapper->ColorByArrayComponent("Height", 0);
    mpWindow->Render();
}
} // namespace Ithaca
