#include "widget.h"

#include "ui_widget.h"

WindowWidget::WindowWidget(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::Window)
{
    ui->setupUi(this);
}

WindowWidget::~WindowWidget()
{
    delete ui;
}
