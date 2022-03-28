#include "sim3_solver.h"
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
    // TO-DO: 之后试下不同scale的图，测试下Sim3的效果
    Mat img_1 = imread ( argv[1], 1 );
    Mat img_2 = imread ( argv[2], 1 );
    int width = img_1.cols;
    int height = img_1.rows;

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    bool viewer = true;
    // 这部分没法省略，因为需要找到correspondence
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches, viewer );
    // 这里有81个匹配对，后面inlier有60个，是说的同一个
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
        // 得到3D点，这里已经有2D点了
        // TO-DO: 看下怎么简化代码？
        all_pts_2d_1.emplace_back ( Vec2_t ( double(keypoints_1[m.queryIdx].pt.x), double(keypoints_1[m.queryIdx].pt.y) ) );
        all_pts_2d_2.emplace_back ( Vec2_t ( double(keypoints_2[m.trainIdx].pt.x), double(keypoints_2[m.trainIdx].pt.y) ) );
        // pixel2cam: 2d->2d，相当于乘了K^-1
        // lamda*p = K*P -> P = lamda*K^-1*p
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /5000.0;
        float dd2 = float ( d2 ) /5000.0;
        all_pts_3d_1.emplace_back ( Vec3_t ( double(p1.x*dd1), double(p1.y*dd1), double(dd1) ) );
        all_pts_3d_2.emplace_back ( Vec3_t ( double(p2.x*dd2), double(p2.y*dd2), double(dd2) ) );
    }
    // cout << "first " << all_pts_2d_2.at(0)<<endl;

    // 这里获得的Inlier个数就是匹配对的子集
    // 总共81组，一般可以到60组内点，还是不错的！
    sim3_solver solver(all_pts_2d_1, all_pts_2d_2, all_pts_3d_1, all_pts_3d_2, K, width, height, true, 20);
    solver.find_via_ransac(200);
    Vec3_t t = solver.get_best_translation_12();
    Mat33_t R = solver.get_best_rotation_12();
    float s = solver.get_best_scale_12();
    cout<<"end of ICP estimation: "<<solver.solution_is_valid()<<endl;
    cout<<"end of ICP rotation: "<<R<<endl;
    cout<<"end of ICP translation: "<<t<<endl;
    cout<<"end of ICP scale: "<<s<<endl;

    double avg_err;
    double error = 0;
    // 2->1
    for ( int i=0; i<all_pts_3d_1.size(); i++ )
    {
        Vec3_t p2_1 = R * all_pts_3d_2[i] + t;
        // calculate error
        double e = abs(p2_1(0) - all_pts_3d_1[i](0)) + 
                   abs(p2_1(1) - all_pts_3d_1[i](1)) +
                   abs(p2_1(2) - all_pts_3d_1[i](2));
        error += e;
    }
    avg_err = error / all_pts_3d_1.size();
    cout<<"average error is "<<avg_err<<endl;

}
