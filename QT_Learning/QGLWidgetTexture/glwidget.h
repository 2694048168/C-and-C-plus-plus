#pragma once

#include <QtGlobal>

// if (QT_VERSION >= QT_VERSION_CHECK(5.0.0))
// https://code.qt.io/cgit/qt/qtbase.git/tree/examples/opengl/textures?h=6.7

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    using QOpenGLWidget::QOpenGLWidget;
    ~GLWidget();

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;
    void rotateBy(int xAngle, int yAngle, int zAngle);
    void setClearColor(const QColor& color);

signals:
    void clicked();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void makeObject();

    QColor clearColor = Qt::black;
    QPoint lastPos;
    int xRot = 0;
    int yRot = 0;
    int zRot = 0;
    QOpenGLTexture* textures[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    QOpenGLShaderProgram* program = nullptr;
    QOpenGLBuffer vbo;
};
