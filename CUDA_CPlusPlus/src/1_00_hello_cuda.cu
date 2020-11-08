/** CUDA C 程序开发流程
  * 0、为主机内存和设备显存中的数据分配内存
  * 1、将数据从主机内存复制到设备显存
  * 2、通过指定并行度来启动内核
  * 3、所有线程完成之后，将数据从设备显存复制回主机内存
  * 4、释放主机和设备上使用的所有内存
  */

 /** CUDA 开发环境
 
  * 0、支持 CUDA 的硬件 GPU
  * 1、GPU 显卡驱动程序
  * 2、标准的C编译器 gcc or cl
  * 3、CUDA Toolkit 开发工具包
  */

 /** CUDA Linux for Debian and Ubuntu 
  * 0、使用 dpkg 安装指定版本的安装包
  *    sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
  * 1、安装 CUDA 公共 GPG 密钥
  *    sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
  * 2、更新 apt repository 仓库缓存
  *    sudo apt update
  * 3、安装 CUDA
  *    sudo apt install cuda
  * 4、修改 PATH 
  *    export PATH=/usr/local/cuda-x.x/bin${PATH:+:${PATH}}
  * 5、设定 LD_LIBRARY_PATH
  *    export LD_LIBRARY_PATH=/usr/local/cuda-x.x/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  * 6、或者使用 apt 直接安装 toolkit
  *    sudo apt install nvidia-cuda-toolkits
  * 7、IDE(NV)
  *    sudo apt install nvidia-nsight
  */

#include <iostream>

// the host code will be complied by stdandard C Complier such as GNU gcc or MSVC cl.
// the device code will be comoplied by Nvidia GPU nvcc complier.

/**device code
 * kernel function 内核函数
 * __global__ 是 CUDA C 在标准C中添加的限定符，告诉编译器该函数时定义在设备上，而不是主机上。
 */
__global__ void my_first_kernel(void)
{
  // Nothing to do.
}

/**host code
 * 
 */
int main(void)
{
  // kernel call 内核调用
  // 内核参数传递，资源分配，<< <block_numbers, thread_numbers> >>
  my_first_kernel << <1, 1> >> ();
  // printf("Hello, CUDA!\n");
  std::cout << "Hello, CUDA!" << std::endl;

  return 0;
}