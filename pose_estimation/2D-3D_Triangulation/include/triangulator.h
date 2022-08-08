#ifndef SOLVE_TRIANGULATOR_H
#define SOLVE_TRIANGULATOR_H

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>

typedef Eigen::Matrix4d Mat44_t;
typedef Eigen::Matrix3d Mat33_t;
typedef Eigen::Vector4d Vec4_t;
typedef Eigen::Vector3d Vec3_t;
typedef Eigen::Vector2d Vec2_t;
typedef Eigen::MatrixXd MatX_t;
typedef Eigen::VectorXd VecX_t;
template<size_t R, size_t C>
using MatRC_t = Eigen::Matrix<double, R, C>;
using Mat34_t = MatRC_t<3, 4>;
using Mat22_t = MatRC_t<2, 2>;


// 三角化分了三种情况，分别是
// case 1: 已知pixel coordinate + 两个view各自的投影矩阵，其中P=K*[R t]
// case 2: 已知bearing + R,t，最常见的情况
// case 3: 已知bearing + 两个view各自相对于世界坐标系的pose
// 其中，case 1和3使用的是同一个SVD方法，case 2则使用十四讲里的矩阵求逆方法
// 通过类中函数的重载实现不同参数的输入
// 使用inline函数来加快速度，适用于小型计算场景
class triangulator {
public:
    /**
     * Triangulate using two points and two perspective projection matrices
     * @param pt_1
     * @param pt_2
     * @param P_1
     * @param P_2
     * @return triangulated point in the world reference
     */
    static inline Vec3_t triangulate(const cv::Point2d& pt_1, const cv::Point2d& pt_2, const Mat34_t& P_1, const Mat34_t& P_2);

    /**
     * Triangulate using two bearings and relative rotation & translation
     * @param bearing_1
     * @param bearing_2
     * @param rot_21
     * @param trans_21
     * @return triangulated point in the camera 1 coordinates
     */
    static inline Vec3_t triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat33_t& rot_21, const Vec3_t& trans_21);

    /**
     * Triangulate using two bearings and absolute camera poses
     * @param bearing_1
     * @param bearing_2
     * @param cam_pose_1
     * @param cam_pose_2
     * @return
     */
    static inline Vec3_t triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat44_t& cam_pose_1, const Mat44_t& cam_pose_2);
};

// SVD方法求解
// 这里的输入是像素坐标，即未经过归一化处理，与下面的bearing不同
Vec3_t triangulator::triangulate(const cv::Point2d& pt_1, const cv::Point2d& pt_2, const Mat34_t& P_1, const Mat34_t& P_2) {
    // 步骤1 构造Ax=0
    Mat44_t A;
    // 步骤1.1 构造A矩阵，4*4的矩阵
    A.row(0) = pt_1.x * P_1.row(2) - P_1.row(0);
    A.row(1) = pt_1.y * P_1.row(2) - P_1.row(1);
    A.row(2) = pt_2.x * P_2.row(2) - P_2.row(0);
    A.row(3) = pt_2.y * P_2.row(2) - P_2.row(1);

    // 步骤1.2 SVD法求解Ax=0
    const Eigen::JacobiSVD<Mat44_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // 最小特征值对应的特征向量，特征值从大到小排列
    const Vec4_t v = svd.matrixV().col(3);
    return v.block<3, 1>(0, 0) / v(3); // 齐次坐标转为非齐次坐标
}

// bearing的定义为[x-cx/fx, y-cy/fy, 1]/sqrt(1+(x-cx/fx)^2+(y-cy/fy)^2)
// 即为像素点的归一化平面坐标再做一次数值归一化
// TO-DO: 这里的结果非常不稳定，和case 1,3的差别很大，研究下原理和解决方案
Vec3_t triangulator::triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat33_t& rot_21, const Vec3_t& trans_21) {
    // 步骤1 先求深度值
    // lamda1*x1 = lamda2*R*x2+t
    // 步骤1.1 把lamda1,2作为未知数，建模成Ax=b的形式
    // 这里的21表示的是1->2，注意rot_21后面有一个transpose，相当于变成了rot_12
    const Vec3_t trans_12 = -rot_21.transpose() * trans_21;
    const Vec3_t bearing_2_in_1 = rot_21.transpose() * bearing_2; // 和十四讲的符号一致

    Mat22_t A;
    A(0, 0) = bearing_1.dot(bearing_1);
    A(1, 0) = bearing_1.dot(bearing_2_in_1);
    A(0, 1) = -A(1, 0);
    A(1, 1) = -bearing_2_in_1.dot(bearing_2_in_1);

    const Vec2_t b{bearing_1.dot(trans_12), bearing_2_in_1.dot(trans_12)};
    
    // 步骤1.2 求得深度值
    const Vec2_t lambda = A.inverse() * b;

    // 步骤2 再求三角化后的3D点坐标
    const Vec3_t pt_1 = lambda(0) * bearing_1;
    const Vec3_t pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
    // 取均值
    return (pt_1 + pt_2) / 2.0;
}

// 这里输入的是bearing，而不是像素值，里面隐含了K矩阵
// TO-DO: 这里的结果和case 1不完全一样，检查下精度问题
Vec3_t triangulator::triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat44_t& cam_pose_1, const Mat44_t& cam_pose_2) {
    // 同理，构造A矩阵
    Mat44_t A;
    A.row(0) = bearing_1(0) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(0);
    A.row(1) = bearing_1(1) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(1);
    A.row(2) = bearing_2(0) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(0);
    A.row(3) = bearing_2(1) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(1);

    // 特征值分解 (A = U S Vt)
    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    Eigen::JacobiSVD<Mat44_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Vec4_t singular_vector = svd.matrixV().block<4, 1>(0, 3); // 这里和svd.matrixV().col(3)效果一样

    return singular_vector.block<3, 1>(0, 0) / singular_vector(3);
}


#endif // SOLVE_TRIANGULATOR_H
