#ifndef __SHARPEN_PLUGIN_H__
#define __SHARPEN_PLUGIN_H__

#include <QObject>
#include <QtPlugin>

#include "editor_plugin_interface.h"

class SharpenPlugin :public QObject, public EditorPluginInterface
{
private:
	Q_OBJECT
		Q_PLUGIN_METADATA(IID EDIT_PLUGIN_INTERFACE_IID);
	Q_INTERFACES(EditorPluginInterface);

public:
	QString name() override;
	void edit(const cv::Mat& input, cv::Mat& output) override;
};

#endif // !__SHARPEN_PLUGIN_H__
