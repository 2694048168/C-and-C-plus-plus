#include "login_widget.h"

#include "ui_login_widget.h"

#include <QEvent.h>

#include <QApplication>
#include <QCheckBox>
#include <QDebug>
#include <QFile>
#include <QKeyEvent>
#include <QLabel>
#include <QWidget>

LoginWidget::LoginWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::FLoginWidget)
{
    ui->setupUi(this);

    this->setWindowFlag(Qt::FramelessWindowHint);
    this->installEventFilter(this);

    QFile file(":/icons/Login.qss");
    if (file.open(QFile::OpenModeFlag::ReadOnly))
    {
        this->setStyleSheet(file.readAll());
    }

    ui->ProfileLabel->setPixmap(QPixmap(":/icons/Profile.png")
                                    .scaled(ui->ProfileLabel->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    connect(ui->AuthorityCheckBox, &QCheckBox::stateChanged,
            [=](int state)
            {
                if (state == Qt::Checked)
                    ui->LoginButton->setEnabled(true);
                else
                    ui->LoginButton->setEnabled(false);
            });
    ui->LoginButton->setEnabled(false);

    connect(ui->CloseButton, &QPushButton::clicked, [=](int state) { this->close(); });
}

LoginWidget::~LoginWidget()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

void LoginWidget::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_F5)
    {
        // qDebug() << "========== F5 ==========\n";
        // qDebug() << qApp->applicationDirPath() + "/../src/icons/Login.qss";
        QFile file(qApp->applicationDirPath() + "/../src/icons/Login.qss");
        if (file.open(QFile::OpenModeFlag::ReadOnly))
        {
            this->setStyleSheet(file.readAll());
        }
    }

    return QWidget::keyPressEvent(event);
}

bool LoginWidget::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == this)
    {
        QMouseEvent *mouse_event = dynamic_cast<QMouseEvent *>(event);
        if (mouse_event)
        {
            static QPoint offset;
            if (mouse_event->type() == QEvent::Type::MouseButtonPress)
            {
                // 计算偏移量
                offset = mouse_event->globalPosition().toPoint() - this->pos();
            }
            else if (mouse_event->type() == QEvent::Type::MouseMove)
            {
                this->move(mouse_event->globalPosition().toPoint() - offset);
            }
        }
    }

    return QWidget::eventFilter(watched, event);
}
