#include <QObject>
#include <QtPlugin>

#include "editor_plugin_interface.h"

#ifdef _WIN32
// Windows MSVC '__declspec' 特有的动态库装载方式
#ifdef ERODE_PLUGIN_API_EXPORT
#define ERODE_PLUGIN_API __declspec(dllexport) /* generate dynamic library */
#else
#define ERODE_PLUGIN_API __declspec(dllimport) /* using dynamic library */
#endif // ERODE_PLUGIN_API
#elif __linux__
#define ERODE_PLUGIN_API
#elif __APPLE__
#define ERODE_PLUGIN_API
#endif // _WIN32

#ifdef __cplusplus
extern "C" {
#endif 

	class  ErodePlugin : public QObject, public EditorPluginInterface
	{
	private:
		Q_OBJECT
			Q_PLUGIN_METADATA(IID EDIT_PLUGIN_INTERFACE_IID);
		Q_INTERFACES(EditorPluginInterface);

	public:
		ERODE_PLUGIN_API QString name() override;
		ERODE_PLUGIN_API void edit(const cv::Mat& input, cv::Mat& output) override;
	};

#ifdef __cplusplus
}
#endif 