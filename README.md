# openvslam-learn

拆解openvslam，在拆解过程中理解代码，同时要求能获得每个求解器的独立输出结果，为以后快速集成做铺垫。

最终的目标是写出一个极简版的SLAM系统（性能差点没关系），方便大家理解3D视觉与SLAM的基本模块与原理，使得新手可以快速入门。

# How to build

sudo apt install libeigen3-dev libopencv-dev libyaml-cpp-dev

mkdir build
cd build
cmake ..
make -j

# How to run

./pnp_solver ../data/1.png ../data/2.png ../data/1_depth.png ../data/2_depth.png