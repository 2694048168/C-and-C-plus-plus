#include "QtVTKApp.h"

QtVTKApp::QtVTKApp(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QtVTKAppClass())
{
    ui->setupUi(this);

    std::string title = "Qt和VTK联合演示3D软件";
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
}

void QtVTKApp::AddWidgetControl()
{
    mpDeformSphereWidget = new Ithaca::DeformSphere;
    mpStackedWidget->addWidget(mpDeformSphereWidget);
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
