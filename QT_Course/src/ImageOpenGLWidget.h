#ifndef __IAMGE_OPEN_GL_WIDGET_H__
#define __IAMGE_OPEN_GL_WIDGET_H__

#include <GL/gl.h>
#include <GL/glu.h>

#include <QImage>
#include <QObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>

class ImageOpenGLWidget
    : public QOpenGLWidget
    , protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit ImageOpenGLWidget(QWidget *parent = 0);

signals:

public slots:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void setImage(const QImage &image);
    void initTextures();
    void initShaders();

private:
    QVector<QVector3D>   vertices;
    QVector<QVector2D>   texCoords;
    QOpenGLShaderProgram program;
    QOpenGLTexture      *texture;
    QMatrix4x4           projection;
};

#endif /* __IAMGE_OPEN_GL_WIDGET_H__ */
