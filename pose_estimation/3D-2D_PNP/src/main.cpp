#include "pnp_solver.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../../../feature/feature.h"

using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    

    //-- 读取图像
    Mat img_1 = imread ( argv[1], 1 );
    Mat img_2 = imread ( argv[2], 1 );
    int width = img_1.cols;
    int height = img_1.rows;

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    bool viewer = true;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches, viewer );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat depth1 = imread ( argv[3], -1 );       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread ( argv[4], -1 );       // 深度图为16位无符号数，单通道图像
    // fx, cx, fy, cy
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    
    // aligned_allocated has to be used to align memory
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> all_pts_3d_1, all_pts_3d_2;
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> all_pts_2d_1, all_pts_2d_2;
    for ( DMatch m:matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        // 保存2D点
        all_pts_2d_1.emplace_back ( Vec2_t ( double(keypoints_1[m.queryIdx].pt.x), double(keypoints_1[m.queryIdx].pt.y) ) );
        all_pts_2d_2.emplace_back ( Vec2_t ( double(keypoints_2[m.trainIdx].pt.x), double(keypoints_2[m.trainIdx].pt.y) ) );
        // pixel2cam: 2d->2d，相当于乘了K^-1
        // lamda*p = K*P -> P = lamda*K^-1*p
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /5000.0;
        float dd2 = float ( d2 ) /5000.0;
        // 保存3D点
        all_pts_3d_1.emplace_back ( Vec3_t ( double(p1.x*dd1), double(p1.y*dd1), double(dd1) ) );
        all_pts_3d_2.emplace_back ( Vec3_t ( double(p2.x*dd2), double(p2.y*dd2), double(dd2) ) );
    }
    // cout << "first " << all_pts_2d_2.at(0)<<endl;

    // convert 2d keypoints to bearings
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> bearing_pts_2d_2;
    bearing_pts_2d_2.resize(all_pts_2d_2.size());
    double cx = K.at<double> ( 0,2 );
    double cy = K.at<double> ( 1,2 );
    double fx = K.at<double> ( 0,0 );
    double fy = K.at<double> ( 1,1 );
    for (unsigned long idx = 0; idx < all_pts_2d_2.size(); ++idx) {
        const auto x_normalized = (all_pts_2d_2[idx](0) - cx) / fx;
        const auto y_normalized = (all_pts_2d_2[idx](1) - cy) / fy;
        const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
        bearing_pts_2d_2.at(idx) = Vec3_t{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
    }

    // TO-DO: 这里原版用了resample_by_indices，需要看下是否需要，我可以先来个最简单的
    // 与sim3_solver是类对象不同，pnp_solver是指针
    auto solver = new pnp_solver(all_pts_3d_1, bearing_pts_2d_2);
    solver->find_via_ransac(30);
    cout<<"end of PNP estimation: "<<solver->solution_is_valid()<<endl;
    Vec3_t t = solver->get_best_translation();
    Mat33_t R = solver->get_best_rotation();
    Mat44_t cam_pose = solver->get_best_cam_pose();
    cout<<"end of PNP rotation: "<<R<<endl;
    cout<<"end of PNP translation: "<<t<<endl;
    cout<<"end of PNP pose: "<<cam_pose<<endl;

    double avg_err;
    double error = 0;
    // X_c = R_cw * X_w + t_cw, world->camera
    for ( int i=0; i<all_pts_3d_1.size(); i++ )
    {
        Vec3_t p_3d_w2c = R * all_pts_3d_1[i] + t;
        Vec2_t p_2d_w2c = cam2pixel(p_3d_w2c, K);
        // calculate error
        double e = abs(p_2d_w2c(0) - all_pts_2d_2[i](0)) + 
                   abs(p_2d_w2c(1) - all_pts_2d_2[i](1));
        error += e;
    }
    avg_err = error / all_pts_3d_1.size();
    cout<<"average error is "<<avg_err<<endl;

}