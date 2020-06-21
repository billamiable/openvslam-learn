#include "sim3_solver.h"
#include "random_array.h"

#include <cmath>

using namespace std;

// TO-DO: 改成读取两张图，可以测试下有scale差的，然后提feature匹配
sim3_solver::sim3_solver(const std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_1,
                         const std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_2,
                         const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_1,
                         const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_2,
                         cv::Mat K, int width, int height,
                         const bool fix_scale, const unsigned int min_num_inliers
                         )
    : common_pts_2d_1_(common_pts_2d_1), common_pts_2d_2_(common_pts_2d_2),
      common_pts_3d_1_(common_pts_3d_1), common_pts_3d_2_(common_pts_3d_2),
      K_(K), width_(width), height_(height), 
      fix_scale_(fix_scale), min_num_inliers_(min_num_inliers){

    // 卡方值，显著性水平为1％（2个自由度）
    chi_sq_2D_ = 9.21034;

    // 获取两帧的3D点匹配对个数
    num_common_pts_ = common_pts_2d_1_.size();

    // Only 3D
    reproject_to_same_image(common_pts_3d_1_, reprojected_1_);
    reproject_to_same_image(common_pts_3d_2_, reprojected_2_);
    // cout<<"first reproj "<< reprojected_2_.at(0)<<endl;

}

// TO-DO: 可以研究下RANSAC的写法，主要是random engine的写法，参考opencv的homography求解
void sim3_solver::find_via_ransac(const unsigned int max_num_iter) {
    // best model初始化
    unsigned int max_num_inliers = 0;
    solution_is_valid_ = false;
    best_rot_12_ = Mat33_t::Zero();
    best_trans_12_ = Vec3_t::Zero();
    best_scale_12_ = 0.0;
    best_reprojection_error_ = 0.0;

    if (num_common_pts_ < 3 || num_common_pts_ < min_num_inliers_) {
        solution_is_valid_ = false;
        return;
    }

    // RANSAC loop内使用的参数
    // in_sac表示正在RANSAC优化中，非最终结果
    Mat33_t rot_12_in_sac; // 1>2
    Vec3_t trans_12_in_sac;
    float scale_12_in_sac;
    Mat33_t rot_21_in_sac; // 2->1
    Vec3_t trans_21_in_sac;
    float scale_21_in_sac;

    // RANSAC loop
    for (unsigned int iter = 0; iter < max_num_iter; ++iter) {
        // 随机将3D点采样到矩阵中
        Mat33_t pts_1, pts_2;
        // TO-DO: 需要修改random这部分，看看大家都喜欢怎么用，我可以先用一个简单的
        // 关键是有一个选过之后去除的要求，这个比较难满足
        const auto random_indices = util::create_random_array(3, 0, static_cast<int>(num_common_pts_ - 1));
        // 按列排列
        // x1 x2 x3
        // y1 y2 y3
        // z1 z2 z3
        for (unsigned int i = 0; i < 3; ++i) {
            pts_1.block(0, i, 3, 1) = common_pts_3d_1_.at(random_indices.at(i));
            pts_2.block(0, i, 3, 1) = common_pts_3d_2_.at(random_indices.at(i));
        }

        // SIM3求解
        compute_Sim3(pts_1, pts_2,
                     rot_12_in_sac, trans_12_in_sac, scale_12_in_sac,
                     rot_21_in_sac, trans_21_in_sac, scale_21_in_sac);

        // 计算Inlier
        std::vector<bool> inliers;
        double reprojection_error;
        const auto num_inliers = count_inliers(rot_12_in_sac, trans_12_in_sac, scale_12_in_sac,
                                               rot_21_in_sac, trans_21_in_sac, scale_21_in_sac,
                                               inliers, reprojection_error);

        // 更新最佳模型
        if (max_num_inliers < num_inliers) {
            max_num_inliers = num_inliers;
            best_rot_12_ = rot_12_in_sac;
            best_trans_12_ = trans_12_in_sac;
            best_scale_12_ = scale_12_in_sac;
            best_reprojection_error_ = reprojection_error; // added
        }
    }

    cout<<"REPROJECTION ERROR, "<<best_reprojection_error_ <<endl;
    cout<<"MAX_NUM_INILERS, "<<max_num_inliers <<endl;
    cout<<"MIN_NUM_INLIERS, "<<min_num_inliers_<<endl;

     // 如果不满足最小内点数条件，则失败
    if (max_num_inliers < min_num_inliers_) {
        solution_is_valid_ = false;
        best_rot_12_ = Mat33_t::Zero();
        best_trans_12_ = Vec3_t::Zero();
        best_scale_12_ = 0.0;
        best_reprojection_error_ = 0.0; // added
        return;
    }
    else {
        solution_is_valid_ = true;
        return;
    }
}


void sim3_solver::compute_Sim3(const Mat33_t& pts_1, const Mat33_t& pts_2,
                               Mat33_t& rot_12, Vec3_t& trans_12, float& scale_12,
                               Mat33_t& rot_21, Vec3_t& trans_21, float& scale_21) {
    // Based on "Closed-form solution of absolute orientation using unit quaternions"
    // http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf

    // 查找每个点集的质心
    const Vec3_t centroid_1 = pts_1.rowwise().mean(); // 按列排列，所以取rowwise的mean
    const Vec3_t centroid_2 = pts_2.rowwise().mean();

    // 将分布的中心移到质心
    Mat33_t ave_pts_1 = pts_1;
    ave_pts_1.colwise() -= centroid_1;
    Mat33_t ave_pts_2 = pts_2;
    ave_pts_2.colwise() -= centroid_2;

    // 4.A Matrix of Sums of Products

    // 构造矩阵M，得到3*3矩阵
    const Mat33_t M = ave_pts_1 * ave_pts_2.transpose();

    // 求矩阵N，不是用旋转矩阵，而是四元数
    // TO-DO: 推导四元数的公式，直接看论文，用到了许多四元数的性质
    // 以下求解方法只在orbslam中应用，其他地方一般直接用M矩阵SVD分解
    const double& Sxx = M(0, 0);
    const double& Syx = M(1, 0);
    const double& Szx = M(2, 0);
    const double& Sxy = M(0, 1);
    const double& Syy = M(1, 1);
    const double& Szy = M(2, 1);
    const double& Sxz = M(0, 2);
    const double& Syz = M(1, 2);
    const double& Szz = M(2, 2);
    Eigen::Matrix4d N;
    N << (Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx),
        (Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz),
        (Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy),
        (Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz);

    // 4.B Eigenvector Maximizes Matrix Product

    // N的特征值分解
    Eigen::EigenSolver<Mat44_t> eigensolver(N);

    // 旋转的四元数即为最大特征值对应的特征向量
    // TO-DO: 为何要找最大特征值，在论文中解释了，需要通过推导得出
    const auto& eigenvalues = eigensolver.eigenvalues();
    int max_idx = -1;
    double max_eigenvalue = -INFINITY;
    for (int idx = 0; idx < 4; ++idx) {
        if (max_eigenvalue <= eigenvalues(idx, 0).real()) {
            max_eigenvalue = eigenvalues(idx, 0).real(); // TO-DO: 之后再研究下eigensolver
            max_idx = idx; // 找到对应的index
        }
    }
    const auto max_eigenvector = eigensolver.eigenvectors().col(max_idx);

    // 由于它是一个复数，因此仅提取实数，这个也是文中推导的
    Eigen::Vector4d eigenvector;
    eigenvector << max_eigenvector(0, 0).real(), max_eigenvector(1, 0).real(), max_eigenvector(2, 0).real(), max_eigenvector(3, 0).real();
    eigenvector.normalize(); // 最后不要忘了normalize！

    // 构造单元四元数
    Eigen::Quaterniond q_rot_21(eigenvector(0), eigenvector(1), eigenvector(2), eigenvector(3));

    // 转换为旋转矩阵
    rot_21 = q_rot_21.normalized().toRotationMatrix();

    // 2.D Finding the Scale

    if (fix_scale_) {
        scale_21 = 1.0;
    }
    // TO-DO: 这个不是最优，其实不能直接倒数
    else {
        // 1到2坐标系（仅旋转）
        const Mat33_t ave_pts_1_in_2 = rot_21 * ave_pts_1; // q是坐标系1，p是坐标系2

        // 分母
        // 因为||R*q||=||q||，所以直接用ave_pts_1就可以了
        const double denom = ave_pts_1.squaredNorm(); // sum of the square of all the matrix entries
        // 分子
        const double numer = ave_pts_2.cwiseProduct(ave_pts_1_in_2).sum();
        // scale
        scale_21 = numer / denom;
    }

    // 2.C Centroids of the Sets of Measurements

    trans_21 = centroid_2 - scale_21 * rot_21 * centroid_1;

    // 旋转矩阵逆变换
    rot_12 = rot_21.transpose();
    scale_12 = 1.0 / scale_21; // TO-DO: NOT CORRECT
    trans_12 = -scale_12 * rot_12 * trans_21;
}

// 两帧互相投影，在误差范围内统计内点个数
unsigned int sim3_solver::count_inliers(const Mat33_t& rot_12, const Vec3_t& trans_12, const float scale_12,
                                        const Mat33_t& rot_21, const Vec3_t& trans_21, const float scale_21,
                                        std::vector<bool>& inliers, double& reprojection_error) {
    // 使用估计的SIM3将一个3D点重新投影到另一幅图像上并计算距离
    unsigned int num_inliers = 0;
    // Inlier最多为匹配对个数，false表示n超过现在容器大小时默认的填充元素
    inliers.resize(num_common_pts_, false); // 实际分配内存，与reserve不同

    // 将坐标系1中的3D点投影到坐标系2中的图像上
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_1_in_cam_2;
    reproject_to_other_image(common_pts_3d_1_, reprojected_1_in_cam_2, rot_21, trans_21, scale_21);

    // 将坐标系2中的3D点投影到坐标系1中的图像上
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_2_in_cam_1;
    reproject_to_other_image(common_pts_3d_2_, reprojected_2_in_cam_1, rot_12, trans_12, scale_12);

    double r_error = 0.0;
    for (unsigned int i = 0; i < num_common_pts_; ++i) {
        // 计算残差向量
        const Vec2_t dist_in_2 = (reprojected_1_in_cam_2.at(i) - common_pts_2d_2_.at(i));
        const Vec2_t dist_in_1 = (reprojected_2_in_cam_1.at(i) - common_pts_2d_1_.at(i));

        // 计算平方误差
        const double error_in_2 = dist_in_2.dot(dist_in_2);
        const double error_in_1 = dist_in_1.dot(dist_in_1);

        // inlier检查
        // TO-DO: 研究chi卡方的参数是怎么给定的
        if (error_in_2 < chi_sq_2D_ && error_in_1 < chi_sq_2D_) {
            inliers.at(i) = true;
            ++num_inliers;
        }

        // in pixel
        // TO-DO: should only calculate inlier reprojection error
        r_error += 0.5*(sqrt(error_in_1)+sqrt(error_in_2));  
    }

    reprojection_error = r_error / num_common_pts_;
    // cout << reprojection_error<<endl;

    return num_inliers;
}

// 多了一个SIM3变换
void sim3_solver::reproject_to_other_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam_1,
                                           std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected_in_cam_2,
                                           const Mat33_t& rot_21, const Vec3_t& trans_21, const float scale_21) {
    reprojected_in_cam_2.clear();
    reprojected_in_cam_2.reserve(lm_coords_in_cam_1.size()); // 只是预留，与resize不同

    for (const auto& lm_coord_in_cam_1 : lm_coords_in_cam_1) {
        Vec2_t reproj_in_cam_2;
        float x_right; // 这里暂时没用，双目的时候会用到
        reproject_to_image(scale_21 * rot_21, trans_21, lm_coord_in_cam_1, reproj_in_cam_2, x_right);

        reprojected_in_cam_2.push_back(reproj_in_cam_2);
    }
}

// Only 3D
void sim3_solver::reproject_to_same_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam,
                                          std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected) {
    reprojected.clear();
    reprojected.reserve(lm_coords_in_cam.size());

    for (const auto& lm_coord_in_cam : lm_coords_in_cam) {
        Vec2_t reproj;
        float x_right;
        reproject_to_image(Mat33_t::Identity(), Vec3_t::Zero(), lm_coord_in_cam, reproj, x_right);

        reprojected.push_back(reproj);
    }
}

// 这个只是出发的角度不同，之前是先有2D，再到3D
// OPENVSLAM中是以3D点保存，再转成2D
bool sim3_solver::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    // convert to camera-coordinates
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (pos_c(2) <= 0.0) {
        return false;
    }

    // reproject onto the image
    const auto z_inv = 1.0 / pos_c(2);
    // x = 1/Z*(fx*X+cx*Z) = 1/z*fx*X + cx
    reproj(0) = K_.at<double> ( 0,0 ) * pos_c(0) * z_inv + K_.at<double> ( 0,2 );
    reproj(1) = K_.at<double> ( 1,1 ) * pos_c(1) * z_inv + K_.at<double> ( 1,2 );
    // x_right = reproj(0) - focal_x_baseline_ * z_inv;
    x_right = reproj(0); // TO-DO: change later, used in stereo

    // check if the point is visible
    // 这个还是蛮重要的，如果超过界限就没有意义了
    return (0 < reproj(0) && reproj(0) < width_
            && 0 < reproj(1) && reproj(1) < height_);
}
