#ifndef __IMAGE_RENDER_WIDGET_H__
#define __IMAGE_RENDER_WIDGET_H__

/**
 * @file ImageRenderWidget.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-21
 * 
 * @copyright Copyright (c) 2024
 * 
 * *=============================== 
 * Qt自带的OpenGL来做这件事情的方法,图像的显示
 * OpenGL纹理怎么在Qt自带的OpenGL上贴纹理, 这在3D制作上是非常重要的.
 * 而对于只想要显示2D图像, 同样可以用纹理贴图来实现.
 * *原理非常简单,学完纹理知道只要用两个三角形直接填充满整个视图, 随后将图像当成纹理往上面贴就完事了.
 * ?不过由于这里不需要考虑空间精细度关系,因此直接使用四边形图元会更加方便.
 * 
 * 纹理贴图的方式显示图像,因此更新图像实际上是在更新纹理.
 * 
 */

#include <QDebug>
#include <QElapsedTimer>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>

class ImageRenderWidget : public QOpenGLWidget
{
    Q_OBJECT
public:
    ImageRenderWidget(QWidget *parent = nullptr);
    ~ImageRenderWidget();

    void setImage(const QImage &image);

protected:
    void initializeGL();         //初始化函数，在Widget刚加载时被调用
    void paintGL();              //绘图函数，每一次绘图请求产生，都会进行一次绘图
    void resizeGL(int w, int h); //用于处理窗口大小变化的情况

private:
    QOpenGLBuffer            *m_vbo;
    QOpenGLVertexArrayObject *m_vao;
    QOpenGLShaderProgram     *m_shader;     //渲染器程序对象
    QOpenGLFunctions         *func_pointer; //OpenGL函数对象
    QOpenGLTexture           *texture = nullptr;
};
#endif /* __IMAGE_RENDER_WIDGET_H__ */
