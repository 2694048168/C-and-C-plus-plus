#include "MedicalModel.h"

#include <QVTKInteractor.h>
#include <vtkAutoInit.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkFlyingEdges3D.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMarchingCubes.h>
#include <vtkNamedColors.h>
#include <vtkObject.h>
#include <vtkOutlineFilter.h>
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

MedicalModel::MedicalModel(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MedicalModelForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpReader       = vtkSmartPointer<vtkMetaImageReader>::New();
    mFilepath
        = R"(D:/Development/GitRepository/C-and-C-plus-plus/VTK_Learning/models/VTK-9.5.2/Testing/Data/HeadMRVolume.mhd)";

    ui->lineEdit_loadFilepath->setText(mFilepath.c_str());
    QObject::connect(ui->btn_loadFilepath, &QPushButton::clicked, this, &MedicalModel::sl_LoadModelFilepath);
}

MedicalModel::~MedicalModel()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void MedicalModel::Run()
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
            mCallbackFunc(false, "错误: 无法加载文件 " + mFilepath + " 请确保文件存在且路径正确");
        return;
    }
    if (mCallbackFunc)
        mCallbackFunc(true, "成功加载模型");

    auto pColors = vtkSmartPointer<vtkNamedColors>::New();

    std::array<unsigned char, 4> skinColor{
        {240, 184, 160, 255}
    };
    pColors->SetColor("SkinColor", skinColor.data());
    std::array<unsigned char, 4> backColor{
        {255, 229, 200, 255}
    };
    pColors->SetColor("BackfaceColor", backColor.data());
    std::array<unsigned char, 4> bkg{
        {51, 77, 102, 255}
    };
    pColors->SetColor("BkgColor", bkg.data());

    // Create the renderer, the render window, and the interactor. The renderer
    // draws into the render window, the interactor enables mouse- and
    // keyboard-based interaction with the data within the render window.
    auto pRenderer = vtkSmartPointer<vtkRenderer>::New();
    mpRenderWindow->AddRenderer(pRenderer);

    // An isosurface, or contour value of 500 is known to correspond to the
    // skin of the patient.
#ifdef USE_FLYING_EDGES
    auto pSkinExtractor = vtkSmartPointer<vtkFlyingEdges3D>::New();
#else
    auto pSkinExtractor = vtkSmartPointer<vtkMarchingCubes>::New();
#endif

    pSkinExtractor->SetInputConnection(mpReader->GetOutputPort());
    pSkinExtractor->SetValue(0, 500);

    auto pSkinMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pSkinMapper->SetInputConnection(pSkinExtractor->GetOutputPort());
    pSkinMapper->ScalarVisibilityOff();

    auto pSkinActor = vtkSmartPointer<vtkActor>::New();
    pSkinActor->SetMapper(pSkinMapper);
    pSkinActor->GetProperty()->SetDiffuseColor(pColors->GetColor3d("SkinColor").GetData());

    auto pBackProp = vtkSmartPointer<vtkProperty>::New();
    pBackProp->SetDiffuseColor(pColors->GetColor3d("BackfaceColor").GetData());
    pSkinActor->SetBackfaceProperty(pBackProp);

    // An outline provides context around the data.
    auto pOutlineData = vtkSmartPointer<vtkOutlineFilter>::New();
    pOutlineData->SetInputConnection(mpReader->GetOutputPort());

    auto pMapOutline = vtkSmartPointer<vtkPolyDataMapper>::New();
    pMapOutline->SetInputConnection(pOutlineData->GetOutputPort());

    auto pOutline = vtkSmartPointer<vtkActor>::New();
    pOutline->SetMapper(pMapOutline);
    pOutline->GetProperty()->SetColor(pColors->GetColor3d("Black").GetData());

    // It is convenient to create an initial view of the data. The FocalPoint
    // and Position form a vector direction. Later on (ResetCamera() method)
    // this vector is used to position the camera to look at the data in
    // this direction.
    auto pCamera = vtkSmartPointer<vtkCamera>::New();
    pCamera->SetViewUp(0, 0, -1);
    pCamera->SetPosition(0, -1, 0);
    pCamera->SetFocalPoint(0, 0, 0);
    pCamera->ComputeViewPlaneNormal();
    pCamera->Azimuth(30.0);
    pCamera->Elevation(30.0);

    // Actors are added to the renderer. An initial camera view is created.
    // The Dolly() method moves the camera towards the FocalPoint,
    // thereby enlarging the image.
    pRenderer->AddActor(pOutline);
    pRenderer->AddActor(pSkinActor);
    pRenderer->SetActiveCamera(pCamera);
    pRenderer->ResetCamera();
    pCamera->Dolly(1.5);

    // Set a background color for the renderer and set the size of the
    // render window (expressed in pixels).
    pRenderer->SetBackground(pColors->GetColor3d("BkgColor").GetData());
    //mpRenderWindow->SetSize(640, 480);
    //mpRenderWindow->SetWindowName("MedicalDemo1");

    // Note that when camera movement occurs (as it does in the Dolly()
    // method), the clipping planes often need adjusting. Clipping planes
    // consist of two planes: near and far along the view direction. The
    // near plane clips out objects in front of the plane; the far plane
    // clips out objects behind the plane. This way only what is drawn
    // between the planes is actually rendered.
    pRenderer->ResetCameraClippingRange();

    // Initialize the event loop and then start it.
    mpRenderWindow->Render();
}

void MedicalModel::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mCallbackFunc = callback;
}

void MedicalModel::sl_LoadModelFilepath()
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
