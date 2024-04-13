#include "CameraParamsConf.hpp"

#include <direct.h>
#include <io.h>

CameraParamsConf *CameraParamsConf::m_pInstance = new CameraParamsConf;
std::string       CameraParamsConf::m_path      = "";
std::string       CameraParamsConf::m_filename  = "";

void CameraParamsConf::SaveParams()
{
    SAVE_CAMERA_PARAMS(m_cameraParams, m_path, m_filename);
}

void CameraParamsConf::LoadParams()
{
    createDirectory(m_path);
    if (!std::filesystem::exists(m_path + m_filename))
        SAVE_CAMERA_PARAMS(m_cameraParams, m_path, m_filename);

    LOAD_CAMERA_PARAMS(m_cameraParams, m_path, m_filename);
}

CameraParams *CameraParamsConf::getParams()
{
    return &m_cameraParams;
}

void CameraParamsConf::setParams(CameraParams &params)
{
    m_cameraParams = params;
}

void CameraParamsConf::createDirectory(std::string &path)
{
    int len = path.length();

    char tmpDirPath[256] = {0};
    for (int idx = 0; idx < len; ++idx)
    {
        tmpDirPath[idx] = path[idx];
        if (tmpDirPath[idx] == '\\' || tmpDirPath[idx] == '/')
        {
            if (_access(tmpDirPath, 0) == -1)
            {
                int ret = _mkdir(tmpDirPath);
                if (ret == -1)
                    break;
            }
        }
    }
}
