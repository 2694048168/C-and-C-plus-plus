#include "ImageRenderWidget.h"

ImageRenderWidget::ImageRenderWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
}

ImageRenderWidget::~ImageRenderWidget()
{
    //多线程调用保护
    makeCurrent();
    //对象释放
    m_vbo->destroy();
    m_vao->destroy();
    delete texture;
    //退出保护
    doneCurrent();
}

void ImageRenderWidget::initializeGL()
{
    func_pointer = this->context()->functions();
    func_pointer->glEnable(GL_DEPTH_TEST); // 三维绘图的关键！
    m_shader = new QOpenGLShaderProgram();
    m_shader->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaderProgram/textures.vert");
    m_shader->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaderProgram/textures.frag");
    if (m_shader->link())
    {
        qDebug("Shaders link success.");
    }
    else
    {
        qDebug("Shaders link failed!");
    }
    //VBO数据
    static const GLfloat vertices[] = {
        //位置                //纹理位置
        1.0f,  1.0f,  0.0f, 0.0f, 0.0f, // top right
        1.0f,  -1.0f, 0.0f, 0.0f, 1.0f, // bottom right
        -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // bottom left
        -1.0f, 1.0f,  0.0f, 1.0f, 0.0f  // top left
    };

    m_vao = new QOpenGLVertexArrayObject();
    m_vbo = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    m_vao->create();
    m_vao->bind();
    m_vbo->create();
    m_vbo->bind();
    m_vbo->allocate(vertices, 4 * 5 * sizeof(GLfloat));

    int attr = -1;
    //顶点属性设置
    attr = m_shader->attributeLocation("aPos");
    m_shader->setAttributeBuffer(attr, GL_FLOAT, 0, 3, sizeof(GLfloat) * 5);
    m_shader->enableAttributeArray(attr);
    //纹理属性设置
    attr = m_shader->attributeLocation("aTexCoord");
    m_shader->setAttributeBuffer(attr, GL_FLOAT, sizeof(GLfloat) * 3, 2, sizeof(GLfloat) * 5);
    m_shader->enableAttributeArray(attr);

    m_vao->release();
    m_vbo->release();

    //纹理读取
    // texture 1
    texture = new QOpenGLTexture(QImage("src/icons/Profile.png"),
                                 QOpenGLTexture::GenerateMipMaps); //直接生成绑定一个2d纹理, 并生成多级纹理MipMaps
    if (!texture->isCreated())
    {
        qDebug() << "Failed to load texture";
    }
    //设置纹理对象的环绕
    texture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::Repeat);
    texture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::Repeat);
    //设置纹理对象的采样方式
    texture->setMinificationFilter(QOpenGLTexture::Linear);  //缩小
    texture->setMagnificationFilter(QOpenGLTexture::Linear); //放大

    //设置纹理对应的单元
    m_shader->bind();
    m_shader->setUniformValue("texture1", 0);
    m_shader->setUniformValue("texture2", 1);
    m_shader->release();
}

void ImageRenderWidget::paintGL()
{
    //纹理绑定
    func_pointer->glActiveTexture(GL_TEXTURE0);
    texture->bind();

    func_pointer->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    func_pointer->glClearColor(0.0f, 0.2f, 0.0f, 1.0f);
    m_vao->bind();
    m_shader->bind();

    func_pointer->glDrawArrays(GL_QUADS, 0, 4); //绘制对象
    m_shader->release();
    m_vao->release();
    texture->release();
}

void ImageRenderWidget::resizeGL(int w, int h)
{
    func_pointer->glViewport(0, 0, w, h);
}

void ImageRenderWidget::setImage(const QImage &image)
{
    texture->setData(image); //设置纹理图像
    //设置纹理细节
    texture->setLevelofDetailBias(-1); //值越小，图像越清晰
    update();
}
