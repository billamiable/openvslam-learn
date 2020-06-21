#include "pnp_solver.h"
#include "random_array.h"
// TO-DO: 下面这个是计算sin,cos的，为何要专门写一个？是为了速度吗?
// 先用基本的cos，之后再看是否要改
// #include "openvslam/util/trigonometric.h" 

// 求解的是X_c = R_cw * X_w + t_cw
// 通过引入bearing，实现单目，双目，rgbd，全景图的兼容
// valid_bearing：对于perspective相机模型，间接表示2D点的[x, y, 1]/sqrt(x^2+y^2)
// landmark是路标点（3D），keypts是特征点（2D），后两者组成3D-2D匹配对
// TO-DO: 使用时须确定3D-2D匹配对是否有效
pnp_solver::pnp_solver(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_3d_w,
                       const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> pts_2d_c,
                       const unsigned int min_num_inliers)
    : num_matches_(pts_3d_w.size()), pts_3d_w_(pts_3d_w),
      pts_2d_c_(pts_2d_c), min_num_inliers_(min_num_inliers) {
    
    // max_cos_errors_.clear();
    // max_cos_errors_.resize(num_matches_);

    // TO-DO: 涉及金字塔没有完全理解
    // 在RANSAC检查Inlier时需要检查cos角度的值
    // 
    constexpr double max_rad_error = 1.0 * M_PI / 180.0;
    max_cos_error_ = cos(max_rad_error);
    // for (unsigned int i = 0; i < num_matches_; ++i) {
    //     const auto max_rad_error_with_scale = scale_factors.at(valid_keypts.at(i).octave) * max_rad_error;
    //     max_cos_errors_.at(i) = cos(max_rad_error_with_scale); // util::cos()
    // }

    assert(num_matches_ == pts_2d_c_.size());
    // assert(num_matches_ == valid_keypts.size());
    assert(num_matches_ == pts_3d_w_.size());
    // assert(num_matches_ == max_cos_errors_.size());
}

pnp_solver::~pnp_solver() {
    // 用数组指针表示更方便
    delete[] pws_; // 3D点在世界坐标系下的坐标，这个是已知的，给定的3D点坐标
    delete[] us_; // 图像坐标系下的2D点坐标，这个是已知的，给定的2D点坐标
    delete[] alphas_; // 真实3D点用4个虚拟控制点表达时的系数
    delete[] pcs_; // 3D点在camera坐标系下的坐标，这个是未知的，待求
    delete[] signs_; // 表征正负
}

// 开始ransac求解
void pnp_solver::find_via_ransac(const unsigned int max_num_iter, const bool recompute) {
    // 1. Prepare for RANSAC

    // minimum number of samples (= 4)
    // TO-DO: 研究下constexpr
    static constexpr unsigned int min_set_size = 4;
    // 都是unsigned int比较
    if (num_matches_ < min_set_size || num_matches_ < min_num_inliers_) {
        solution_is_valid_ = false;
        return;
    }

    // RANSAC variables
    unsigned int max_num_inliers = 0;
    is_inlier_match = std::vector<bool>(num_matches_, false);

    // shared variables in RANSAC loop
    // rotation from world to camera
    // world已知，camera未知，所以从world到camera
    Mat33_t rot_cw_in_sac;
    // translation from world to camera
    Vec3_t trans_cw_in_sac;
    // inlier/outlier flags
    std::vector<bool> is_inlier_match_in_sac;

    // 2. RANSAC loop
    // 这里开始应该是和传统的都一样，可以快速过一遍
    for (unsigned int iter = 0; iter < max_num_iter; ++iter) { // 同为unsigned int比较
        // 2-1. Create a minimum set
        const auto random_indices = util::create_random_array(min_set_size, 0U, num_matches_ - 1); // 0U表示unsigned
        assert(random_indices.size() == min_set_size); // 多加了一些assert语句
        // 下面是获取3D-2D匹配对
        reset_correspondences(); // number_of_correspondences置零
        set_max_num_correspondences(min_set_size); // 只要分配最小set的大小即可
        for (const auto i : random_indices) {
            const Vec3_t& bearing = pts_2d_c_.at(i);
            const Vec3_t& pos_w = pts_3d_w_.at(i);
            add_correspondence(pos_w, bearing); // 组成匹配对
        }

        // 2-2. Compute a camera pose
        // 核心函数，利用EPNP方法求解位姿
        compute_pose(rot_cw_in_sac, trans_cw_in_sac);

        // 2-3. Check inliers and compute a score
        const auto num_inliers = check_inliers(rot_cw_in_sac, trans_cw_in_sac, is_inlier_match_in_sac);

        // 2-4. Update the best model
        if (max_num_inliers < num_inliers) {
            max_num_inliers = num_inliers;
            best_rot_cw_ = rot_cw_in_sac;
            best_trans_cw_ = trans_cw_in_sac;
            is_inlier_match = is_inlier_match_in_sac;
        }
    }

    if (max_num_inliers > min_num_inliers_) {
        solution_is_valid_ = true;
    }

    if (!recompute || !solution_is_valid_) {
        return;
    }

    // 这个不错，可以作为sim3_solver的参考
    // 3. Recompute a camera pose only with the inlier matches

    const auto num_inliers = std::count(is_inlier_match.begin(), is_inlier_match.end(), true);
    reset_correspondences();
    set_max_num_correspondences(num_inliers);
    for (unsigned int i = 0; i < num_matches_; ++i) {
        if (!is_inlier_match.at(i)) {
            continue;
        }
        const Vec3_t& bearing = pts_2d_c_.at(i);
        const Vec3_t& pos_w = pts_3d_w_.at(i);
        add_correspondence(pos_w, bearing);
    }
    // 这里就不再用ransac了
    compute_pose(best_rot_cw_, best_trans_cw_);
}

unsigned int pnp_solver::check_inliers(const Mat33_t& rot_cw, const Vec3_t& trans_cw, std::vector<bool>& is_inlier) {
    unsigned int num_inliers = 0;

    is_inlier.resize(num_matches_);
    for (unsigned int i = 0; i < num_matches_; ++i) {
        const Vec3_t& pos_w = pts_3d_w_.at(i);
        const Vec3_t& bearing = pts_2d_c_.at(i);

        const Vec3_t pos_c = rot_cw * pos_w + trans_cw; // 转换到相机坐标系

        const auto cos = pos_c.dot(bearing) / pos_c.norm(); // 值越大说明夹角小，效果越好

        // cos值超过阈值时，说明效果好，设为内点
        // if (max_cos_errors_.at(i) < cos) {
        if (max_cos_error_ < cos) {
            is_inlier.at(i) = true;
            ++num_inliers;
        }
        else {
            is_inlier.at(i) = false;
        }
    }

    return num_inliers;
}

void pnp_solver::reset_correspondences() {
    num_correspondences_ = 0;
}

void pnp_solver::set_max_num_correspondences(const unsigned int max_num_correspondences) {
    // TO-DO: 这样写不太elegant
    // 防止指针未释放
    delete[] pws_;
    delete[] us_;
    delete[] alphas_;
    delete[] pcs_;
    delete[] signs_;

    max_num_correspondences_ = max_num_correspondences; // 为4
    // 初始化数组指针，一个指针指向一个数组
    pws_ = new double[3 * max_num_correspondences_];
    us_ = new double[2 * max_num_correspondences_];
    alphas_ = new double[4 * max_num_correspondences_];
    pcs_ = new double[3 * max_num_correspondences_];
    signs_ = new int[max_num_correspondences_];
}

void pnp_solver::add_correspondence(const Vec3_t& pos_w, const Vec3_t& bearing) {
    // 根据定义这个很难为0
    if (bearing(2) == 0) {
        return;
    }

    // 赋值3D坐标
    pws_[3 * num_correspondences_] = pos_w(0);
    pws_[3 * num_correspondences_ + 1] = pos_w(1);
    pws_[3 * num_correspondences_ + 2] = pos_w(2);

    // 间接得到2D坐标
    us_[2 * num_correspondences_] = bearing(0) / bearing(2);
    us_[2 * num_correspondences_ + 1] = bearing(1) / bearing(2);

    // 这个应该是为了兼容双目，全景图输入而加的
    if (0.0 < bearing(2)) {
        signs_[num_correspondences_] = 1;
    }
    else {
        signs_[num_correspondences_] = -1;
    }

    ++num_correspondences_;
}

double pnp_solver::compute_pose(Mat33_t& rot_cw, Vec3_t& trans_cw) {
    // 获得4个控制点坐标
    choose_control_points();
    // 求解4个控制点的系数alphas
    compute_barycentric_coordinates();

    MatX_t M(2 * num_correspondences_, 12); // 8*12

    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        // 填充最小二乘的M矩阵，根据刚体变换不变性求取3D点在控制坐标系下的坐标
        fill_M(M, 2 * i, alphas_ + 4 * i, us_[2 * i], us_[2 * i + 1]); // 合理
    }

    // 目标：求解相机坐标系下控制点坐标
    // 求解Mx=0
    const MatX_t MtM = M.transpose() * M; // 12*12
    Eigen::JacobiSVD<MatX_t> SVD(MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
    // 这里的U其实和V就是转置关系，因为输入是对称矩阵，Ut即为V
    // TO-DO: 测试一下大小
    const MatX_t Ut = SVD.matrixU().transpose(); // 特征向量，12*12，组成相机坐标系下4个控制点坐标

    MatRC_t<6, 10> L_6x10;
    MatRC_t<6, 1> Rho;

    // 步骤1 假设相机坐标系下控制点坐标由特征向量的加权和得到
    // 步骤1.1 构建加权系数的线性方程组
    compute_L_6x10(Ut, L_6x10); // 计算并填充矩阵L
    compute_rho(Rho); // 计算4个控制点任意两点间的距离，总共6个距离

    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];

    // 步骤1.2 近似法求解加权系数
    // 建模为除B11、B12、B13、B14四个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
    find_betas_approx_1(L_6x10, Rho, Betas[1]);
    gauss_newton(L_6x10, Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(Ut, Betas[1], Rs[1], ts[1]); // TO-DO: 为何用到Ut??

    // 建模为除B00、B01、B11三个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
    find_betas_approx_2(L_6x10, Rho, Betas[2]);
    gauss_newton(L_6x10, Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(Ut, Betas[2], Rs[2], ts[2]);

    // 建模为除B00、B01、B11、B02、B12五个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
    find_betas_approx_3(L_6x10, Rho, Betas[3]);
    gauss_newton(L_6x10, Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(Ut, Betas[3], Rs[3], ts[3]);

    unsigned int N = 1;
    if (rep_errors[2] < rep_errors[1]) {
        N = 2;
    }
    if (rep_errors[3] < rep_errors[N]) {
        N = 3;
    }

    for (unsigned int r = 0; r < 3; ++r) {
        for (unsigned int c = 0; c < 3; ++c) {
            rot_cw(r, c) = Rs[N][r][c];
        }
    }

    trans_cw(0) = ts[N][0];
    trans_cw(1) = ts[N][1];
    trans_cw(2) = ts[N][2];

    return rep_errors[N];
}

void pnp_solver::choose_control_points() {
    // Take C0 as the reference points centroid:
    cws[0][0] = cws[0][1] = cws[0][2] = 0;

    // 求和平均取质心
    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            cws[0][j] += pws_[3 * i + j];
        }
    }
    for (unsigned int j = 0; j < 3; ++j) {
        cws[0][j] /= num_correspondences_;
    }

    // Take C1, C2, and C3 from PCA on the reference points:
    MatX_t PW0(num_correspondences_, 3);
    // 减去质心归一化
    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            PW0(i, j) = pws_[3 * i + j] - cws[0][j];
        }
    }

    // PCA找到三个正交的基方向
    const MatX_t PW0tPW0 = PW0.transpose() * PW0;
    Eigen::JacobiSVD<MatX_t> SVD(PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const MatX_t D = SVD.singularValues();
    const MatX_t Ut = SVD.matrixU().transpose(); // 即为特征向量

    // cws仍为4*3的矩阵，因此其实没有压缩特征维度
    for (unsigned int i = 1; i < 4; ++i) { // 从1开始的
        const double k = sqrt(D(i - 1, 0) / num_correspondences_); // sqrt(lamda/n)
        for (unsigned int j = 0; j < 3; ++j) {
            // TO-DO: 下面的计算公式应该是工程上的经验方程，最重要的是特征向量，scale不重要
            cws[i][j] = cws[0][j] + k * Ut((i - 1), j); // 质心+k*特征向量
        }
    }
}

void pnp_solver::compute_barycentric_coordinates() {
    Mat33_t CC;

    // 这里其实写得不好，memory indexing时i仍然会转成Unsigned int类型
    for (int i = 0; i < 3; ++i) {
        for (unsigned int j = 1; j < 4; ++j) {
            CC(i, j - 1) = cws[j][i] - cws[0][i]; // 隐含了转置，变成了按列摆放
        }
    }

    const Mat33_t CC_inv = CC.inverse();

    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        double* pi = pws_ + 3 * i; // 输入3D点，3表示3维，指针指向数组
        double* a = alphas_ + 4 * i; // 赋值alpha系数，4表示4个系数

        for (unsigned int j = 0; j < 3; ++j) {
            a[1 + j] = CC_inv(j, 0) * (pi[0] - cws[0][0])   // x
                       + CC_inv(j, 1) * (pi[1] - cws[0][1]) // y
                       + CC_inv(j, 2) * (pi[2] - cws[0][2]);// z
        }

        a[0] = 1.0f - a[1] - a[2] - a[3]; // 根据公式可得
    }
}

void pnp_solver::fill_M(MatX_t& M, const int row, const double* as, const double u, const double v) {
    // 根据推导赋值即可
    for (unsigned int i = 0; i < 4; ++i) {
        M(row, 3 * i) = as[i] * fx_;
        M(row, 3 * i + 1) = 0.0;
        M(row, 3 * i + 2) = as[i] * (cx_ - u);

        M(row + 1, 3 * i) = 0.0;
        M(row + 1, 3 * i + 1) = as[i] * fy_;
        M(row + 1, 3 * i + 2) = as[i] * (cy_ - v);
    }
}

void pnp_solver::compute_ccs(const double* betas, const MatX_t& ut) {
    for (unsigned int i = 0; i < 4; ++i) {
        ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0;
    }

    for (unsigned int i = 0; i < 4; ++i) { // 表示N=4
        for (unsigned int j = 0; j < 4; ++j) { // 表示有4个控制点
            for (unsigned int k = 0; k < 3; ++k) {
                // ut按行排列特征向量
                ccs[j][k] += betas[i] * ut(11 - i, 3 * j + k); // 这里reference的效率比较高！
            }
        }
    }
}

void pnp_solver::compute_pcs() {
    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        double* a = alphas_ + 4 * i; // 权重系数
        double* pc = pcs_ + 3 * i;

        for (unsigned int j = 0; j < 3; ++j) {
            // 每个控制点在相机坐标系下的坐标
            pc[j] = a[0] * ccs[0][j]
                    + a[1] * ccs[1][j]
                    + a[2] * ccs[2][j]
                    + a[3] * ccs[3][j];
        }
    }
}

double pnp_solver::dist2(const double* p1, const double* p2) {
    return (p1[0] - p2[0]) * (p1[0] - p2[0])
           + (p1[1] - p2[1]) * (p1[1] - p2[1])
           + (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double pnp_solver::dot(const double* v1, const double* v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// 这个写得挺好的，可以给sim3_solver借鉴
double pnp_solver::reprojection_error(const double R[3][3], const double t[3]) {
    double sum2 = 0.0;

    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        double* pw = pws_ + 3 * i;
        double Xc = dot(R[0], pw) + t[0]; // world to camera
        double Yc = dot(R[1], pw) + t[1];
        double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
        double ue = cx_ + fx_ * Xc * inv_Zc;
        double ve = cy_ + fy_ * Yc * inv_Zc;
        double u = us_[2 * i], v = us_[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve)); // Pixel级别的squared error
    }

    return sum2 / num_correspondences_;
}

void pnp_solver::estimate_R_and_t(double R[3][3], double t[3]) {
    // 表示中心
    double pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        const double* pc = pcs_ + 3 * i;
        const double* pw = pws_ + 3 * i;

        for (unsigned int j = 0; j < 3; ++j) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }
    for (unsigned int j = 0; j < 3; ++j) {
        pc0[j] /= num_correspondences_;
        pw0[j] /= num_correspondences_;
    }

    // 下面的方法和之前的sim3_solver是一样的
    // 先求得去质心后的3D点，然后构造3*3的M矩阵，即为这里的Abt
    // 只不过sim3_solver是用四元数表示的，这里是用旋转矩阵表示
    MatX_t Abt(3, 3);

    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            Abt(i, j) = 0.0;
        }
    }

    for (unsigned int i = 0; i < num_correspondences_; ++i) {
        const double* pc = pcs_ + 3 * i;
        const double* pw = pws_ + 3 * i;

        for (unsigned int j = 0; j < 3; ++j) {
            // camera * world
            // 这里camera放在前面，因此最后得到的是X_c = R_cw * X_w + t_cw
            Abt(j, 0) += (pc[j] - pc0[j]) * (pw[0] - pw0[0]); // 这里为何不写成都是j？
            Abt(j, 1) += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            Abt(j, 2) += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    Eigen::JacobiSVD<MatX_t> SVD(Abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const MatX_t& Abt_u = SVD.matrixU();
    const MatX_t& Abt_v = SVD.matrixV();

    // R = U * V^T
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            R[i][j] = Abt_u.row(i) * Abt_v.row(j).transpose(); // 这里是不是写得复杂了？
        }
    }

    // 求行列式
    const double det = R[0][0] * R[1][1] * R[2][2]
                       + R[0][1] * R[1][2] * R[2][0]
                       + R[0][2] * R[1][0] * R[2][1]
                       - R[0][2] * R[1][1] * R[2][0]
                       - R[0][1] * R[1][0] * R[2][2]
                       - R[0][0] * R[1][2] * R[2][1];

    //change 1: negative determinant problem is solved by changing Abt_v, not R

    if (det < 0) {
        MatX_t Abt_v_prime = Abt_v;
        Abt_v_prime.col(2) = -Abt_v.col(2); // TO-DO: 不理解V为何变为-V？
        for (unsigned int i = 0; i < 3; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                R[i][j] = Abt_u.row(i) * Abt_v_prime.row(j).transpose();
            }
        }
    }
    
    // R表示world->camera, 即R_cw
    t[0] = pc0[0] - dot(R[0], pw0);
    t[1] = pc0[1] - dot(R[1], pw0);
    t[2] = pc0[2] - dot(R[2], pw0);
}

void pnp_solver::solve_for_sign() {
    // 整体符号调整
    //change to this (using original depths)
    if ((pcs_[2] < 0.0 && signs_[0] > 0) || (pcs_[2] > 0.0 && signs_[0] < 0)) {
        for (unsigned int i = 0; i < 4; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                ccs[i][j] = -ccs[i][j];
            }
        }

        for (unsigned int i = 0; i < num_correspondences_; ++i) {
            pcs_[3 * i] = -pcs_[3 * i];
            pcs_[3 * i + 1] = -pcs_[3 * i + 1];
            pcs_[3 * i + 2] = -pcs_[3 * i + 2];
        }
    }
}

double pnp_solver::compute_R_and_t(const MatX_t& Ut, const double* betas, double R[3][3], double t[3]) {
    // 每一个控制点在相机坐标系下都表示为特征向量乘以beta的形式，EPnP论文的公式16
    compute_ccs(betas, Ut);
    // 用四个控制点作为单位向量表示任意3D点，将其转化到相机坐标系下
    compute_pcs();
    // 随便取一个相机坐标系下3D点，如果z < 0，则表明3D点都在相机后面，则3D点坐标整体取负号
    solve_for_sign();
    // 3D-3D svd方法求解ICP获得R，t
    // 见《视觉SLAM十四讲从理论到实践》 7.9.1
    estimate_R_and_t(R, t);
    // 获得R，t后计算所有3D点的重投影误差平均值
    return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void pnp_solver::find_betas_approx_1(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas) {
    MatX_t L_6x4(6, 4);
    // 只要保存近似要求解的即可
    for (unsigned int i = 0; i < 6; ++i) {
        L_6x4(i, 0) = L_6x10(i, 0);
        L_6x4(i, 1) = L_6x10(i, 1);
        L_6x4(i, 2) = L_6x10(i, 3);
        L_6x4(i, 3) = L_6x10(i, 6);
    }
    // SVD求解线性方程
    Eigen::JacobiSVD<MatX_t> SVD(L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const VecX_t Rho_temp = Rho;
    const VecX_t b4 = SVD.solve(Rho_temp); // 得到beta_l

    // 调整符号
    if (b4[0] < 0) {
        // 根据beta_l和beta的关系式反解出beta
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    }
    else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void pnp_solver::find_betas_approx_2(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas) {
    MatX_t L_6x3(6, 3);

    for (unsigned int i = 0; i < 6; ++i) {
        L_6x3(i, 0) = L_6x10(i, 0);
        L_6x3(i, 1) = L_6x10(i, 1);
        L_6x3(i, 2) = L_6x10(i, 2);
    }

    Eigen::JacobiSVD<MatX_t> SVD(L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const VecX_t Rho_temp = Rho;
    const VecX_t b3 = SVD.solve(Rho_temp);

    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) {
        betas[0] = -betas[0];
    }

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void pnp_solver::find_betas_approx_3(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double* betas) {
    MatX_t L_6x5(6, 5);

    for (unsigned int i = 0; i < 6; ++i) {
        L_6x5(i, 0) = L_6x10(i, 0);
        L_6x5(i, 1) = L_6x10(i, 1);
        L_6x5(i, 2) = L_6x10(i, 2);
        L_6x5(i, 3) = L_6x10(i, 3);
        L_6x5(i, 4) = L_6x10(i, 4);
    }

    Eigen::JacobiSVD<MatX_t> SVD(L_6x5, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const VecX_t Rho_temp = Rho;
    const VecX_t b5 = SVD.solve(Rho_temp);

    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) {
        betas[0] = -betas[0];
    }

    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

void pnp_solver::compute_L_6x10(const MatX_t& Ut, MatRC_t<6, 10>& L_6x10) {
    // 4表示特征向量取前4个，即N=4
    // 6表示两两相连的边有6条
    // 3表示每个控制点有三维坐标
    double dv[4][6][3]; // 中间变量

    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int a = 0, b = 1;
        for (unsigned int j = 0; j < 6; ++j) {
            // 由于本身求得的12维向量由4个控制点组成，所以每个控制点占有3维
            // 取特征值最小的4个所对应的特征向量，这里选择行，因为之前转置了
            dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
            dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
            dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);

            // 下面的部分是为了实现4条边的6组不重复取样
            // 实现方式还是挺巧妙的
            ++b; // 实现(0,1),(0,2),(0,3)   (1,3)
            if (b > 3) {
                ++a;
                b = a + 1; // 实现(1,2)  (2,3)
            }
        }
    }

    for (unsigned int i = 0; i < 6; ++i) {
        // 按照公式赋值即可
        L_6x10(i, 0) = dot(dv[0][i], dv[0][i]); // 注意模的平方(dot)得到的是数值，不再是向量
        L_6x10(i, 1) = 2.0f * dot(dv[0][i], dv[1][i]); // 注意这里有的已经乘以2了
        L_6x10(i, 2) = dot(dv[1][i], dv[1][i]);
        L_6x10(i, 3) = 2.0f * dot(dv[0][i], dv[2][i]);
        L_6x10(i, 4) = 2.0f * dot(dv[1][i], dv[2][i]);
        L_6x10(i, 5) = dot(dv[2][i], dv[2][i]);
        L_6x10(i, 6) = 2.0f * dot(dv[0][i], dv[3][i]);
        L_6x10(i, 7) = 2.0f * dot(dv[1][i], dv[3][i]);
        L_6x10(i, 8) = 2.0f * dot(dv[2][i], dv[3][i]);
        L_6x10(i, 9) = dot(dv[3][i], dv[3][i]);
    }
}

void pnp_solver::compute_rho(MatRC_t<6, 1>& Rho) {
    // 下面的顺序很重要，要和前面的一致
    Rho[0] = dist2(cws[0], cws[1]);
    Rho[1] = dist2(cws[0], cws[2]);
    Rho[2] = dist2(cws[0], cws[3]);
    Rho[3] = dist2(cws[1], cws[2]);
    Rho[4] = dist2(cws[1], cws[3]);
    Rho[5] = dist2(cws[2], cws[3]);
}

void pnp_solver::compute_A_and_b_gauss_newton(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho,
                                              double betas[4], MatRC_t<6, 4>& A, MatRC_t<6, 1>& b) {
    for (unsigned int i = 0; i < 6; ++i) {
        // 前面已经有乘以2的了，所以这里不用乘
        // 按照公式来赋值即可
        A(i, 0) = 2 * L_6x10(i, 0) * betas[0] + L_6x10(i, 1) * betas[1]
                  + L_6x10(i, 3) * betas[2] + L_6x10(i, 6) * betas[3];
        A(i, 1) = L_6x10(i, 1) * betas[0] + 2 * L_6x10(i, 2) * betas[1]
                  + L_6x10(i, 4) * betas[2] + L_6x10(i, 7) * betas[3];
        A(i, 2) = L_6x10(i, 3) * betas[0] + L_6x10(i, 4) * betas[1]
                  + 2 * L_6x10(i, 5) * betas[2] + L_6x10(i, 8) * betas[3];
        A(i, 3) = L_6x10(i, 6) * betas[0] + L_6x10(i, 7) * betas[1]
                  + L_6x10(i, 8) * betas[2] + 2 * L_6x10(i, 9) * betas[3];

        // 10项相乘，表示Dk_w-Dk_c(n-1)
        b(i, 0) = Rho[i] - (L_6x10(i, 0) * betas[0] * betas[0] + L_6x10(i, 1) * betas[0] * betas[1] + L_6x10(i, 2) * betas[1] * betas[1] + L_6x10(i, 3) * betas[0] * betas[2] + L_6x10(i, 4) * betas[1] * betas[2] + L_6x10(i, 5) * betas[2] * betas[2] + L_6x10(i, 6) * betas[0] * betas[3] + L_6x10(i, 7) * betas[1] * betas[3] + L_6x10(i, 8) * betas[2] * betas[3] + L_6x10(i, 9) * betas[3] * betas[3]);
    }
}

void pnp_solver::gauss_newton(const MatRC_t<6, 10>& L_6x10, const MatRC_t<6, 1>& Rho, double betas[4]) {
    const int iterations_number = 5;

    // AX=B
    MatRC_t<6, 4> A;
    MatRC_t<6, 1> B;
    MatRC_t<4, 1> X;

    // 建模成迭代循环优化的最小二乘法模型
    for (unsigned int k = 0; k < iterations_number; ++k) {
        // 构造Ax = B中的A和B，A为目标函数关于待优化变量（B0、B1、B2、B3）的雅克比
        // B为目标函数当前残差（相机坐标系下控制点之间的平方距离与世界坐标系下控制点之间的平方距离之差）
        compute_A_and_b_gauss_newton(L_6x10, Rho, betas, A, B);
        // QR分解法求解Ax=b
        qr_solve(A, B, X);

        // 由于以上求解得到的是β的变化量，所以需要用+=
        for (unsigned int i = 0; i < 4; ++i) {
            betas[i] += X[i];
        }
    }
}

// TO-DO: 这里应该借鉴了别人的写法，之后研究
void pnp_solver::qr_solve(MatRC_t<6, 4>& A_orig, MatRC_t<6, 1>& b, MatRC_t<4, 1>& X) {
    MatRC_t<4, 6> A = A_orig.transpose();

    static int max_nr = 0;
    static double *A1, *A2;

    const int nr = A_orig.rows();
    const int nc = A_orig.cols();

    if (max_nr != 0 && max_nr < nr) {
        delete[] A1;
        delete[] A2;
    }
    if (max_nr < nr) {
        max_nr = nr;
        A1 = new double[nr];
        A2 = new double[nr];
    }

    double* pA = A.data();
    double* ppAkk = pA;
    for (int k = 0; k < nc; ++k) {
        double* ppAik = ppAkk;
        double eta = fabs(*ppAik);
        for (int i = k + 1; i < nr; ++i) {
            const double elt = fabs(*ppAik);
            if (eta < elt) {
                eta = elt;
            }
            ppAik += nc;
        }

        if (eta == 0) {
            A1[k] = A2[k] = 0.0;
            return;
        }
        else {
            double* ppAik = ppAkk;
            double sum = 0.0;
            const double inv_eta = 1.0 / eta;
            for (int i = k; i < nr; ++i) {
                *ppAik *= inv_eta;
                sum += *ppAik * *ppAik;
                ppAik += nc;
            }
            double sigma = sqrt(sum);
            if (*ppAkk < 0) {
                sigma = -sigma;
            }
            *ppAkk += sigma;
            A1[k] = sigma * *ppAkk;
            A2[k] = -eta * sigma;
            for (int j = k + 1; j < nc; ++j) {
                double* ppAik = ppAkk;
                double sum = 0.0;
                for (int i = k; i < nr; i++) {
                    sum += *ppAik * ppAik[j - k];
                    ppAik += nc;
                }
                const double tau = sum / A1[k];
                ppAik = ppAkk;
                for (int i = k; i < nr; ++i) {
                    ppAik[j - k] -= tau * *ppAik;
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;
    }

    // b <- Qt b
    double* ppAjj = pA;
    double* pb = b.data();
    for (int j = 0; j < nc; ++j) {
        double *ppAij = ppAjj, tau = 0;
        for (int i = j; i < nr; i++) {
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1[j];
        ppAij = ppAjj;
        for (int i = j; i < nr; ++i) {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b
    double* pX = X.data();
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for (int i = nc - 2; i >= 0; --i) {
        double *ppAij = pA + i * nc + (i + 1), sum = 0;

        for (int j = i + 1; j < nc; ++j) {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2[i];
    }
}
