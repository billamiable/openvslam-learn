# Triangulation求解

将OpenVSLAM中的triangulator部分单独拎出来，加上slambook中的特征提取代码，组成了一个单独的三角化求解模块。三角化求解的方法与OpenVSLAM中的一致，支持三种不同的输入。

- case 1: 已知pixel coordinate + 两个view各自的投影矩阵，其中P=K*[R t]
- case 2: 已知bearing + R,t，最常见的情况
- case 3: 已知bearing + 两个view各自相对于世界坐标系的pose