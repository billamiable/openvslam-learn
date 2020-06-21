#ifndef SOLVE_PNP_SOLVER_H
#define SOLVE_PNP_SOLVER_H

// 下面的代码基本和orbslam是一样的

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
typedef Eigen::MatrixXd MatX_t;
typedef Eigen::VectorXd VecX_t;
template<size_t R, size_t C>
using MatRC_t = Eigen::Matrix<double, R, C>;


class pnp_solver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Constructor
    pnp_solver(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_3d_w,
               const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_2d_c,
               const unsigned int min_num_inliers = 10);

    //! Destructor
    virtual ~pnp_solver();

    //! Find the most reliable camera pose via RANSAC
    void find_via_ransac(const unsigned int max_num_iter, const bool recompute = true);

    //! Check if the solution is valid or not
    bool solution_is_valid() const {
        return solution_is_valid_;
    }

    //! Get the most reliable rotation (as the world reference)
    Mat33_t get_best_rotation() const {
        return best_rot_cw_;
    }

    //! Get the most reliable translation (as the world reference)
    Vec3_t get_best_translation() const {
        return best_trans_cw_;
    }

    //! Get the most reliable camera pose (as the world reference)
    Mat44_t get_best_cam_pose() const {
        Mat44_t cam_pose = Mat44_t::Identity();
        cam_pose.block<3, 3>(0, 0) = best_rot_cw_;
        cam_pose.block<3, 1>(0, 3) = best_trans_cw_;
        return cam_pose;
    }

    //! Get the inlier flags estimated via RANSAC
    // 问题：这是干嘛的？没用到
    std::vector<bool> get_inlier_flags() const {
        return is_inlier_match;
    }

private:
    //! Check inliers of 2D-3D matches
    //! (Note: inlier flags are set to_inlier_match and the number of inliers is returned)
    unsigned int check_inliers(const Mat33_t& rot_cw, const Vec3_t& trans_cw, std::vector<bool>& is_inlier);

    //! the number of 2D-3D matches
    const unsigned int num_matches_;
    // the following vectors are corresponded as element-wise
    //! bearing vector
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_2d_c_;
    //! 3D point
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_3d_w_;
    //! acceptable maximum error
    // std::vector<float> max_cos_errors_;
    float max_cos_error_;

    //! minimum number of inliers
    //! (Note: if the number of inliers is less than this, solution is regarded as invalid)
    const unsigned int min_num_inliers_;

    //! solution is valid or not
    bool solution_is_valid_ = false;
    //! most reliable rotation
    Mat33_t best_rot_cw_;
    //! most reliable translation
    Vec3_t best_trans_cw_;
    //! inlier matches computed via RANSAC
    std::vector<bool> is_inlier_match;

    //-----------------------------------------
    // quoted from EPnP implementation
    // 下面的函数与基本的EPnP方法是一样的

    void reset_correspondences();

    void set_max_num_correspondences(const unsigned int max_num_correspondences);

    void add_correspondence(const Vec3_t& pos_w, const Vec3_t& bearing);

    double compute_pose(Mat33_t& rot_cw, Vec3_t& trans_cw);

    double reprojection_error(const double R[3][3], const double t[3]);

    void choose_control_points();

    void compute_barycentric_coordinates();

    void fill_M(MatX_t& M, const int row, const double* alphas, const double u, const double v);

    void compute_ccs(const double* betas, const MatX_t& ut);

    void compute_pcs();

    void solve_for_sign();

    void find_betas_approx_1(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas);

    void find_betas_approx_2(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas);

    void find_betas_approx_3(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas);

    void qr_solve(MatRC_t<6, 4>& A, MatRC_t<6, 1>& b, MatRC_t<4, 1>& X);

    double dot(const double* v1, const double* v2);

    double dist2(const double* p1, const double* p2);

    void compute_rho(MatRC_t<6, 1>& Rho);

    void compute_L_6x10(const MatX_t& Ut, MatRC_t<6, 10>& L_6x10);

    void gauss_newton(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double current_betas[4]);

    void compute_A_and_b_gauss_newton(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double cb[4], MatRC_t<6, 4>& A, MatRC_t<6, 1>& b);

    double compute_R_and_t(const MatX_t& Ut, const double* betas, double R[3][3], double t[3]);

    void estimate_R_and_t(double R[3][3], double t[3]);

    double* pws_ = nullptr;
    double* us_ = nullptr;
    double* alphas_ = nullptr;
    double* pcs_ = nullptr;
    int* signs_ = nullptr;

    // 相机参数
    // TO-DO: 研究下constexpr的用处
    static constexpr float fx_ = 1.0, fy_ = 1.0, cx_ = 0.0, cy_ = 0.0;

    double cws[4][3], ccs[4][3];

    unsigned int num_correspondences_ = 0;
    unsigned int max_num_correspondences_ = 0;
};


#endif // SOLVE_PNP_SOLVER_H
