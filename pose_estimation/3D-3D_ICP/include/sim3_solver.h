#ifndef SOLVE_SIM3_SOLVER_H
#define SOLVE_SIM3_SOLVER_H

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
typedef Eigen::Vector3d Vec3_t;
typedef Eigen::Vector2d Vec2_t;

class sim3_solver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    sim3_solver(const std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_1,
                const std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_2,
                const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_1,
                const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_2,
                cv::Mat K, int width, int height,
                const bool fix_scale = true, const unsigned int min_num_inliers = 20);

    // Destructor
    virtual ~sim3_solver() = default;

    // Find the most reliable Sim3 matrix via RANSAC
    void find_via_ransac(const unsigned int max_num_iter);

    // Check if the solution is valid or not
    // 专门做了一个接口来确定是否有解
    bool solution_is_valid() const {
        return solution_is_valid_;
    }

    // Get the most reliable rotation from keyframe 2 to keyframe 1
    Mat33_t get_best_rotation_12() {
        return best_rot_12_;
    }

    // Get the most reliable translation from keyframe 2 to keyframe 1
    Vec3_t get_best_translation_12() {
        return best_trans_12_;
    }

    // Get the most reliable scale from keyframe 2 to keyframe 1
    float get_best_scale_12() {
        return best_scale_12_;
    }

    // reproject points in camera (local) coordinates to the other image (as undistorted keypoints)
    void reproject_to_other_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam_1,
                                  std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected_in_cam_2,
                                  const Mat33_t& rot_21, const Vec3_t& trans_21, const float scale_21);
    
    // Only 3D
    // reproject points in camera (local) coordinates to the same image (as undistorted keypoints)
    void reproject_to_same_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam,
                                              std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected);

    // reprojection implementation
    bool reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const;

protected:
    // compute Sim3 from three common points
    // 在输入矩阵中，每列为[x_i，y_i，z_i]^T，并且在行方向上总共排列了3列
    void compute_Sim3(const Mat33_t& pts_1, const Mat33_t& pts_2,
                      Mat33_t& rot_12, Vec3_t& trans_12, float& scale_12,
                      Mat33_t& rot_21, Vec3_t& trans_21, float& scale_21);

    // count up inliers
    unsigned int count_inliers(const Mat33_t& rot_12, const Vec3_t& trans_12, const float scale_12,
                               const Mat33_t& rot_21, const Vec3_t& trans_21, const float scale_21,
                               std::vector<bool>& inliers, double& reprojection_error);

protected:
    // 3d-3d pairs
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_1_;
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_3d_2_;

    // 2d-2d pairs
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_1_;
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> common_pts_2d_2_;

    // 2d-2d pairs using projection
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_1_;
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_2_;

    // true: Sim3, false: SE3
    bool fix_scale_;

    // 自由度为2时重投影误差方差的卡方
    float chi_sq_2D_;

    //! 共同的3D点数
    unsigned int num_common_pts_ = 0;

    // solution is valid or not
    bool solution_is_valid_ = false;
    // most reliable rotation from keyframe 2 to keyframe 1
    Mat33_t best_rot_12_;
    // most reliable translation from keyframe 2 to keyframe 1
    Vec3_t best_trans_12_;
    // most reliable scale from keyframe 2 to keyframe 1
    float best_scale_12_;
    // reprojection error
    double best_reprojection_error_;

    // RANSAC参数
    unsigned int min_num_inliers_;

    // 相机参数
    cv::Mat K_;
    int height_;
    int width_;
};


#endif // SOLVE_SIM3_SOLVER_H
