#include "QtVTKApp.h"

#include <vtkSmartPointer.h>
#include <vtkVersion.h>

#include <QtGlobal>

QtVTKApp::QtVTKApp(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QtVTKAppClass())
{
    ui->setupUi(this);

    std::string title = "Qt和VTK联合演示3D软件 " + GetVersion();
    this->setWindowTitle(title.c_str());

    Init();
    Connects();
    AddWidgetControl();

    RecordLog(true, "欢迎使用 " + title);
    MessageTip(true, "欢迎使用 " + title);
}

QtVTKApp::~QtVTKApp()
{
    if (mpStackedWidget)
    {
        delete mpStackedWidget;
        mpStackedWidget = nullptr;
    }

    delete ui;
}

void QtVTKApp::Init()
{
    // 创建 stackedWidget
    mpStackedWidget = new QStackedWidget(this);
    ui->gridLayoutCenter->addWidget(mpStackedWidget);

    // 设置最大块数（行数）
    ui->textEdit_log->document()->setMaximumBlockCount(1000);
    // 设置为纯色背景
    ui->textEdit_log->setStyleSheet("background-color: #f0f0f0;"); // 浅灰色
}

void QtVTKApp::Connects()
{
    // toolButton 连接 clicked() 信号
    connect(ui->toolButton_DeformSphere, &QToolButton::clicked, this, &QtVTKApp::sl_DeformSphere);
    connect(ui->toolButton_BarChart, &QToolButton::clicked, this, &QtVTKApp::sl_BarChart);
    connect(ui->toolButton_BorderDraw, &QToolButton::clicked, this, &QtVTKApp::sl_BorderDraw);
}

void QtVTKApp::AddWidgetControl()
{
    mpDeformSphereWidget = new Ithaca::DeformSphere;
    mpStackedWidget->addWidget(mpDeformSphereWidget);

    mpBarChartWidget = new Ithaca::BarChart;
    mpStackedWidget->addWidget(mpBarChartWidget);

    mpBorderDrawWidget = new Ithaca::BorderDraw;
    mpStackedWidget->addWidget(mpBorderDrawWidget);
}

void QtVTKApp::sl_DeformSphere()
{
    RecordLog(true, "点击了【变形球体】演示");

    mpStackedWidget->setCurrentWidget(mpDeformSphereWidget);
    mpDeformSphereWidget->Run();
}

void QtVTKApp::sl_BarChart()
{
    RecordLog(true, "点击了【条形图形】演示");

    mpStackedWidget->setCurrentWidget(mpBarChartWidget);
    mpBarChartWidget->Run();
}

void QtVTKApp::sl_BorderDraw()
{
    RecordLog(true, "点击了【绘制边框】演示");

    mpStackedWidget->setCurrentWidget(mpBorderDrawWidget);
    mpBorderDrawWidget->Run();
}

void QtVTKApp::MessageTip(bool flag, const std::string &message)
{
    if (flag)
    {
        // 设置背景颜色为 green, 字体颜色为 black
        ui->label_message->setStyleSheet("background-color: green; color: black;");
    }
    else
    {
        // 设置背景颜色为 red, 字体颜色为 black
        ui->label_message->setStyleSheet("background-color: red; color: black;");
    }

    ui->label_message->setText(message.c_str());
}

void QtVTKApp::RecordLog(bool flag, const std::string &message)
{
    if (false == flag)
    {
        ui->textEdit_log->setTextColor(Qt::red);
        //ui->textEdit_log->insertPlainText(message.c_str());
        ui->textEdit_log->append(message.c_str()); // auto "\n"
        ui->textEdit_log->setTextColor(Qt::black);
    }
    else
    {
        ui->textEdit_log->setTextColor(Qt::blue);
        //ui->textEdit_log->insertPlainText(message.c_str());
        ui->textEdit_log->append(message.c_str());
        ui->textEdit_log->setTextColor(Qt::black);
    }
}

std::string QtVTKApp::GetVersion()
{
    auto softwareVersionStr
        = "SoftWare Version: V" + std::to_string(mMajor) + "." + std::to_string(mMinor) + "." + std::to_string(mBuild);

    auto pVersion      = vtkSmartPointer<vtkVersion>::New();
    auto VTKVersionStr = "VTK Version: V" + std::to_string(pVersion->GetVTKMajorVersion()) + "."
                       + std::to_string(pVersion->GetVTKMinorVersion()) + "."
                       + std::to_string(pVersion->GetVTKBuildVersion());

    auto QTVersionStr = "QT Version: V" + std::string(qVersion());

    auto VersionStr = softwareVersionStr + " " + VTKVersionStr + " " + QTVersionStr;
    //RecordLog(true, VersionStr);
    return VersionStr;
}
