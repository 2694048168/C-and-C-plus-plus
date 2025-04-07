#include "semver.h"

#include <iostream>

// --------------------------------------
int main(int argc, const char *argv[])
{
    // 初始版本 1.0.0
    Version::SemVer initialRelease(1, 0, 0);
    std::cout << "Initial release: " << initialRelease.toString() << std::endl;

    // 修复了一些bug - 1.0.1
    Version::SemVer bugfixRelease(1, 0, 1);
    std::cout << "Bugfix release: " << bugfixRelease.toString() << std::endl;

    // 添加新功能但保持兼容 - 1.1.0
    Version::SemVer featureRelease(1, 1, 0);
    std::cout << "Feature release: " << featureRelease.toString() << std::endl;

    // 破坏性变更 - 2.0.0
    Version::SemVer breakingRelease(2, 0, 0);
    std::cout << "Breaking change release: " << breakingRelease.toString() << std::endl;

    // 预发布版本
    Version::SemVer prerelease(2, 1, 0, "alpha.1");
    std::cout << "Prerelease: " << prerelease.toString() << std::endl;

    // 带构建元数据的版本
    Version::SemVer buildMetadata(2, 1, 0, "", "20240315.1");
    std::cout << "With build metadata: " << buildMetadata.toString() << std::endl;

    // 版本比较
    if (initialRelease < bugfixRelease)
    {
        std::cout << initialRelease.toString() << " is older than " << bugfixRelease.toString() << std::endl;
    }

    // 从字符串解析
    try
    {
        Version::SemVer parsed = Version::SemVer::fromString("1.2.3-beta.4+sha.123456");
        std::cout << "Parsed version: " << parsed.toString() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing version: " << e.what() << std::endl;
    }

    return 0;
}
