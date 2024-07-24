#include "QGLWidgetTexture.h"
#include <QGridLayout>
#include <QTimer>

QGLWidgetTexture::QGLWidgetTexture(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QGLWidgetTextureClass())
{
    ui->setupUi(this);

    QGridLayout* mainLayout = new QGridLayout;

    for (int i = 0; i < NumRows; ++i) {
        for (int j = 0; j < NumColumns; ++j) {
            QColor clearColor;
            clearColor.setHsv(((i * NumColumns) + j) * 255
                / (NumRows * NumColumns - 1),
                255, 63);

            glWidgets[i][j] = new GLWidget;
            glWidgets[i][j]->setClearColor(clearColor);
            glWidgets[i][j]->rotateBy(+42 * 16, +42 * 16, -21 * 16);
            mainLayout->addWidget(glWidgets[i][j], i, j);

            connect(glWidgets[i][j], &GLWidget::clicked,
                this, &QGLWidgetTexture::setCurrentGlWidget);
        }
    }
    setLayout(mainLayout);

    currentGlWidget = glWidgets[0][0];

    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &QGLWidgetTexture::rotateOneStep);
    timer->start(20);

    setWindowTitle(tr("Textures"));
}

QGLWidgetTexture::~QGLWidgetTexture()
{
    delete ui;
}

void QGLWidgetTexture::setCurrentGlWidget()
{
    currentGlWidget = qobject_cast<GLWidget*>(sender());
}

void QGLWidgetTexture::rotateOneStep()
{
    if (currentGlWidget)
        currentGlWidget->rotateBy(+2 * 16, +2 * 16, -1 * 16);
}
