#include "GraphRender.h"

#include "ui_GraphRender.h"

GraphRender::GraphRender(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::FGraphRender)
{
    ui->setupUi(this);
}

GraphRender::~GraphRender()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}
