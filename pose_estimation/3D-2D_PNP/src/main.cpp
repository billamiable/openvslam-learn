#include "pnp_solver.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

Vec2_t cam2pixel ( const Vec3_t& p, const Mat& K );

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    

    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    int width = img_1.cols;
    int height = img_1.rows;

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat depth1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread ( argv[4], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
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
    // TO-DO: 下面的需要修改下，变成3D-2D匹配对
    // TO-DO: 研究下valid_indices等输入变量，其他应该就没问题了。
    auto solver = new pnp_solver(all_pts_3d_1, bearing_pts_2d_2);
    // auto pnp_solver = setup_pnp_solver(valid_indices, curr_frm.bearings_, curr_frm.keypts_, matched_landmarks.at(i), curr_frm.scale_factors_);
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

Vec2_t cam2pixel ( const Vec3_t& p, const Mat& K )
{
    return Vec2_t
           (
               p(0) * K.at<double> ( 0,0 ) / p(2) + K.at<double> ( 0,2 ),
               p(1) * K.at<double> ( 0,0 ) / p(2) + K.at<double> ( 1,2 )
           );
}


void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}