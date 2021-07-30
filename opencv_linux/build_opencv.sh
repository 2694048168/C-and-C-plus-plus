mkdir -p build && cd build

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules -DOPENCV_ENABLE_NONFREE=ON -DCMAKE_BUILD_TYPE=Release -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_CUDA=OFF -DWITH_QT=OFF -DWITH_GTK=ON ../opencv-master

cmake --build . -j8