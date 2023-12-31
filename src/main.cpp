/*
Copyright 2011 Nghia Ho. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BY NGHIA HO OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Nghia Ho.
*/

#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <cassert>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/cuda.hpp>

#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>


#include "CUDA_RANSAC_Homography.h"
//#include "timing.h"

#ifdef PLATFORM_WINDOWS

#include <windows.h>
struct timeval {
    time_t		tv_sec;		/* seconds */
    long tv_usec;	/* and microseconds */
};
/* FILETIME of Jan 1 1970 00:00:00. */
/* FILETIME of Jan 1 1970 00:00:00. */
static const unsigned __int64 epoch = UINT64CONST(116444736000000000);

/*
* timezone information is stored outside the kernel so tzp isn't used anymore.
*
* Note: this function is not for Win32 high precision timing purpose. See
* elapsed_time().
*/
int gettimeofday(struct timeval* tp, struct timezone* tzp) {
	FILETIME    file_time;
	SYSTEMTIME  system_time;
	ULARGE_INTEGER ularge;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	ularge.LowPart = file_time.dwLowDateTime;
	ularge.HighPart = file_time.dwHighDateTime;

	tp->tv_sec = (long)((ularge.QuadPart - epoch) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);

	return 0;
}
#else
#include <sys/time.h>
#endif

using namespace std;
using namespace cv;

bool g_bExit = false;
bool g_bDrawCircles = false;
bool g_bUseGPUHomography = false;
bool g_bTrack = false;
bool g_bUseGpuSURF = true;
bool g_bAccumulate = false;


// Calc the theoretical number of iterations using some conservative parameters
const double CONFIDENCE = 0.99;
const double INLIER_RATIO = 0.18; // Assuming lots of noise in the data!
const double INLIER_THRESHOLD = 20.0; //3.0  // pixel distance
const int MIN_GOOD_MATCHES = 4;

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

Rect g_rectangle;
Rect g_orig_track_rectangle;
bool g_bDrawingBox = false;
bool g_bStartedDrawingBox = false;
bool g_bHaveBox = false;
int g_BoxStartX=0, g_BoxStartY=0;
RNG g_rng(0);  // Generate random number


void DrawRectangle(Mat& img, Rect box, Scalar color = Scalar(g_rng.uniform(0, 255),
					g_rng.uniform(0,255),g_rng.uniform(0,255)), int thickness = 1)
{
	//Draw a rectangle with random color
	rectangle(img, box.tl(), box.br(), color, thickness);
}

void on_MouseHandle(int event, int x, int y, int flags, void* param) {
	Mat& image = *(cv::Mat*) param;
    switch (event) {
        case EVENT_MOUSEMOVE: {    // When mouse moves, get the current rectangle's width and height
                                  if (g_bStartedDrawingBox) {
                                      g_rectangle.width = abs(g_BoxStartX - x);
                                      g_rectangle.height = abs(g_BoxStartY  - y);
                                      g_rectangle.x = x < g_BoxStartX ? x : g_BoxStartX;
                                      g_rectangle.y = y < g_BoxStartY ? y : g_BoxStartY;
                                  }
                              }
                              break;
        case EVENT_LBUTTONDOWN: {  // when the left mouse button is pressed down,
                                   //get the starting corner's coordinates of the rectangle
                                    g_bStartedDrawingBox = true;
                                    g_BoxStartX = x;
                                    g_BoxStartY = y;
                                }
                                break;
        case EVENT_LBUTTONUP: {   //when the left mouse button is released,
                                  //draw the rectangle
                                  if(g_bStartedDrawingBox && g_rectangle.width>1 && g_rectangle.height>1) {
                                      g_bHaveBox = true;
                                      g_orig_track_rectangle = g_rectangle;
                                  }
                                  g_bStartedDrawingBox = false;
                              }
                              break;
    }
}
#if 0
Mat myDrawMatches(cv::cuda::GpuMat a, cv::cuda::GpuMat b, const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {

    Mat grey1(a), grey2(b);

    int h = grey1.rows + grey2.rows;
    int w = max(grey1.cols, grey2.cols);

    Mat result(h, w, CV_8UC3);

    for(int y=0; y < grey1.rows; y++) {
        for(int x=0; x < grey1.cols; x++) {
            result.at<Vec3b>(y,x)[0] = grey1.at<uchar>(y,x);
            result.at<Vec3b>(y,x)[1] = grey1.at<uchar>(y,x);
            result.at<Vec3b>(y,x)[2] = grey1.at<uchar>(y,x);
        }
    }

    for(int y=0; y < grey2.rows; y++) {
        for(int x=0; x < grey2.cols; x++) {
            result.at<Vec3b>(y+grey1.rows,x)[0] = grey2.at<uchar>(y,x);
            result.at<Vec3b>(y+grey1.rows,x)[1] = grey2.at<uchar>(y,x);
            result.at<Vec3b>(y+grey1.rows,x)[2] = grey2.at<uchar>(y,x);        }
    }

    std::vector<bool> inlier_mask(src.size());

    RNG match_color_rng(0);  // Generate random number

    for(unsigned int i=0; i < inlier_mask.size(); i++) {
        //if(inlier_mask[i]) {
        //
        int r = match_color_rng.uniform(0, 255);
        int g = match_color_rng.uniform(0, 255);
        int b = match_color_rng.uniform(0, 255);
        line(result, Point(src[i].x, src[i].y), Point(dst[i].x, grey1.rows + dst[i].y), CV_RGB(r,g,b), 1, cv::LINE_AA);
        //}
    }

    return result;
}
#endif

void myDrawMatches(Mat result, Mat grey1, Mat grey2, const std::vector<DMatch> matches, 
        const std::vector<KeyPoint>& kp1, const std::vector<KeyPoint>& kp2, bool b_draw_area, const std::vector<char>* mask) {

    //Mat grey1(a), grey2(b);

    int h = grey1.rows + grey2.rows;
    int w = max(grey1.cols, grey2.cols);

    for(int y=0; y < grey1.rows; y++) {
        for(int x=0; x < grey1.cols; x++) {
            result.at<Vec3b>(y,x)[0] = grey1.at<uchar>(y,x);
            result.at<Vec3b>(y,x)[1] = grey1.at<uchar>(y,x);
            result.at<Vec3b>(y,x)[2] = grey1.at<uchar>(y,x);
        }
    }

    for(int y=0; y < grey2.rows; y++) {
        for(int x=0; x < grey2.cols; x++) {
            result.at<Vec3b>(y+grey1.rows,x)[0] = grey2.at<uchar>(y,x);
            result.at<Vec3b>(y+grey1.rows,x)[1] = grey2.at<uchar>(y,x);
            result.at<Vec3b>(y+grey1.rows,x)[2] = grey2.at<uchar>(y,x);
        }
    }

    RNG match_color_rng(0);  // Generate random number

    for(unsigned int i=0; i < matches.size(); i++) {
        int r = match_color_rng.uniform(0, 155);
        int g = match_color_rng.uniform(0, 155);
        int b = match_color_rng.uniform(0, 155);

        const int idx1 = matches[i].queryIdx;
        const int idx2 = matches[i].trainIdx;
        Point pt1 = Point(kp1[idx1].pt.x, kp1[idx1].pt.y);
        Point pt2 = Point(kp2[idx2].pt.x, grey1.rows + kp2[idx2].pt.y);
        Scalar c = CV_RGB(r,g,b);

        line(result, pt1, pt2, ( mask==0 || (*mask)[i] ) ? c : CV_RGB(255,255,255),1, cv::LINE_AA);
        if(b_draw_area) {
            circle(result, pt1, kp1[idx1].size/2, c, 1, cv::LINE_AA);
            circle(result, pt2, kp2[idx2].size/2, c, 1, cv::LINE_AA);
        }
    }
}

std::vector<Point2f> transformBox(Rect r, Mat H) {
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(r.x, r.y);
    obj_corners[1] = Point2f(r.x + r.width, r.y);
    obj_corners[2] = Point2f(r.x + r.width, r.y + r.height);
    obj_corners[3] = Point2f(r.x, r.y + r.height);

    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);
    return scene_corners;
}

void drawPolyline(Mat out, std::vector<Point2f> pts, Point2f off, Scalar colour, int width) {
    uint32_t count = pts.size();
    for(int i=0;i<count; ++i) {
        line(out, pts[i] + off, pts[(i+1)%count] + off, colour, width);
    }
}

void reconstructPose(Mat H, Mat& RotM, Mat& TrVec) {
    // Normalization to ensure that ||c1|| = 1
    double norm = sqrt(H.at<double>(0,0)*H.at<double>(0,0) +
            H.at<double>(1,0)*H.at<double>(1,0) +
            H.at<double>(2,0)*H.at<double>(2,0));
    H /= norm;
    Mat c1 = H.col(0);
    Mat c2 = H.col(1);
    Mat c3 = c1.cross(c2);
    TrVec = H.col(2);
    Mat R(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
    {
        R.at<double>(i,0) = c1.at<double>(i,0);
        R.at<double>(i,1) = c2.at<double>(i,0);
        R.at<double>(i,2) = c3.at<double>(i,0);
    }
    String mat_str; mat_str << R;
    String t_str; t_str << TrVec;
    printf("R (before polar decomposition): %s\n det(R): %.2f\n", mat_str.c_str(), determinant(R));
    Mat_<double> W, U, Vt;
    SVDecomp(R, W, U, Vt);
    R = U*Vt;
    double det = determinant(R);
    if (det < 0) {
        Vt.at<double>(2,0) *= -1;
        Vt.at<double>(2,1) *= -1;
        Vt.at<double>(2,2) *= -1;
        R = U*Vt;
    }
    mat_str << R;
    printf("R (after polar decomposition): %s\n det(R): %.2f\n", mat_str.c_str(), determinant(R));
    printf("T: %s\n", t_str.c_str());
    RotM = R;
}

void drawPose(Mat out, Mat RotM, Mat TrVec, int w, int h) {
    Mat rvec;
    cv::Rodrigues(RotM, rvec);

    cv::Mat distCoeffs = (cv::Mat_<float>(4,1)<<0,0,0,0);
#if 0
    distCoeffs.at<float>(0) = 0.016478045550785764; // k1
    distCoeffs.at<float>(1) = -0.02176317098941869; // k2
    distCoeffs.at<float>(2) = 0.003739511470855945; // p1
    distCoeffs.at<float>(3) = -0.0026895016023160863;// p2
#endif
    float squareSize = 20;
    Mat cam_mat = Mat::eye(3,3, CV_32F);
#if 0
    cam_mat.at<float>(0,0) = 395.54921897113553;// fx
    cam_mat.at<float>(1,1) = 395.470648141951; // fy
    cam_mat.at<float>(0,2) = 325.31980911255465;
    cam_mat.at<float>(1,2) = 245.2448967915069;
#else
    cam_mat.at<float>(0,2) = w; // cx
    cam_mat.at<float>(1,2) = h; // cy
#endif
    cv::drawFrameAxes(out, cam_mat, distCoeffs, rvec, TrVec, 2*squareSize);
}

Rect rect_from_points(const std::vector<Point2f>& pts) {
    float minx = pts[0].x;
    float maxx = pts[0].x;
    float miny = pts[0].y;
    float maxy = pts[0].y;

    for(int i=1;i<(int)pts.size();++i) {
        minx = minx < pts[i].x ? minx : pts[i].x;
        maxx = maxx > pts[i].x ? maxx : pts[i].x;
        miny = miny < pts[i].y ? miny : pts[i].y;
        maxy = maxy > pts[i].y ? maxy : pts[i].y;
    }
    return Rect(minx, miny, maxx - minx, maxy - miny);
}

void usage() {
    fprintf(stderr, "usage: cuda-flow [--left img --right img] [--video file]\n");
}

int main(int argc, char **argv)
{
    // TEMP:
    //
    //if(argc != 4) {
     //   printf("Usage: CUDA_RANSAC_Homography [img.jpg] [target.jpg] [results.png]\n");
  //      return 0;
    //}

    //timing::init();

    timeval start_time, t1, t2;
    cv::cuda::GpuMat img1, img2;
    cv::cuda::GpuMat obj_subimg;
    vector<KeyPoint> kp1, kp2;
    Mat grey1, grey2;
    int best_inliers;
    float best_H[9];
    vector <char> inlier_mask;
    Mat cpu_img1, cpu_img2;
    VideoCapture cap;
    bool bUseVideo = false;
    bool bAdvanceFrame = true;


    assert(cv::cuda::getCudaEnabledDeviceCount());
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    gettimeofday(&start_time, NULL);

    printf("--------------------------------\n");
    int img_loaded = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--left")
        {
            cpu_img1 = imread(argv[++i], IMREAD_GRAYSCALE);
            img1.upload(cpu_img1);
            CV_Assert(!img1.empty());
            img_loaded++;
        }
        else if (string(argv[i]) == "--right")
        {
            cpu_img2 = imread(argv[++i], IMREAD_GRAYSCALE);
            img2.upload(cpu_img2);
            CV_Assert(!img2.empty());
            img_loaded++;
        }
        else if (string(argv[i]) == "--video" && argc>i+1)
        {
            if(!cap.open(argv[++i])) {
                fprintf(stderr, "Cannot open video input file: %s\n", argv[i]);
                usage();
                return -1;
            }
            bUseVideo = true;
        }
        else if (string(argv[i]) == "--help")
        {
            usage();
            return -1;
        }
    }

    if(img_loaded!=2 && !bUseVideo) {
        usage();
        return -1;
    }

    namedWindow("matches", 0);
    setMouseCallback("matches", on_MouseHandle, (void*)0);

    int out_w = bUseVideo ? cap.get(cv::CAP_PROP_FRAME_WIDTH) : cpu_img1.cols;   
    int out_h = bUseVideo ? 2*cap.get(cv::CAP_PROP_FRAME_HEIGHT): cpu_img2.rows + cpu_img2.rows;   
    Mat out(out_h, out_w, CV_8UC3);

    cv::cuda::SURF_CUDA surf(250, 2, 4, 1, 0.1);
    cv::cuda::GpuMat gpu_kp1, gpu_kp2;
    cv::cuda::GpuMat gpu_desc1, gpu_desc2;
    std::vector<cv::cuda::GpuMat> train_desc;
    std::vector<cv::cuda::GpuMat> train_desc_mask;
    std::vector<int> train_kp_offsets;

    Mat cpu_desc1, cpu_desc2;
    cv::cuda::GpuMat gpu_ret_idx, gpu_ret_dist, gpu_all_dist;

    Ptr<AKAZE> akaze = AKAZE::create();


    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    Ptr<DescriptorMatcher> flann_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    BFMatcher bf_hamming_matcher(NORM_HAMMING);

    vector<DMatch> goodMatches;
    vector<float> descriptors1, descriptors2;

    Mat OpenCV_H;
    Mat refined_H;
    int opencv_inliers;
    int K;

    Mat f;
    int frame_idx = -1;
    while(!g_bExit && (!bUseVideo || cap.isOpened()))
    {
        if(bUseVideo) {
            if(bAdvanceFrame) {

                if(!g_bTrack) {

                    g_orig_track_rectangle = g_rectangle;

                    cpu_img2.copyTo(cpu_img1);
                    img2.copyTo(img1);

                    gpu_kp2.copyTo(gpu_kp1);
                    gpu_desc2.copyTo(gpu_desc1);

                    if(!g_bAccumulate) {
                        kp1.clear();
                        train_desc.clear();
                        //train_desc_mask.clear();
                        train_kp_offsets.clear();
                    }

                    // accumulate keypoints
                    train_kp_offsets.push_back(kp1.size());
                    //Mat mask(kp2.size(), kp2.size(), CV_8UC1);
                    kp1.insert(kp1.end(), kp2.begin(), kp2.end());
                    //for(int i=0;i<kp2.size();++i) {
                    //   mask.at<char>(i, 0) = !g_bHaveBox || g_orig_track_rectangle.contains(kp2[i].pt);
                    //}
                    //cv::cuda::GpuMat gpu_mask;
                    //gpu_mask.upload(mask);
                    //train_desc_mask.push_back(gpu_mask);

                    // accumulate descriptors
                    cv::cuda::GpuMat m;
                    gpu_desc1.copyTo(m);
                    train_desc.push_back(m);

                    // do not accumulate these because not using them really
                    descriptors1 = descriptors2;

                    // TODO: use descriptors1/2 instead of duplicating data
                    cpu_desc2.copyTo(cpu_desc1);

                }

                if(!cap.read(f)) {
                    fprintf(stderr, "Error reading frame");
                    exit(-1);
                }

                cv::cvtColor(f, cpu_img2, cv::COLOR_RGB2GRAY);
                if(g_bUseGpuSURF) {
                    // OPT: if we have box only use subset of an image centered on a box
                    if(g_bHaveBox && 0) {
                        Point2f c = (g_orig_track_rectangle.br() + g_orig_track_rectangle.tl())/2;
                        Rect r(c - Point2f(320, 240), c+ Point2f(320, 240));
                        r.x = r.x < 0 ? 0 : r.x;
                        r.y = r.y < 0 ? 0 : r.y;
                        //w = w - (w + x - c) = c - x
                        r.width = r.width + r.x > cpu_img2.cols ? cpu_img2.cols - r.x : r.width;
                        r.height = r.height + r.y > cpu_img2.rows ? cpu_img2.rows - r.y : r.height;
                        img2.upload(Mat(cpu_img2, r));
                    } else {
                        img2.upload(cpu_img2);
                    }
                    surf(img2, cv::cuda::GpuMat(), gpu_kp2, gpu_desc2);
                    surf.downloadKeypoints(gpu_kp2, kp2);
                    surf.downloadDescriptors(gpu_desc2, descriptors2);
                    // TODO: actually create Mat from descriptors2 instead
                    // (used only for flann matcher)
                    gpu_desc2.download(cpu_desc2);
                } else {
                    akaze->detectAndCompute(cpu_img2, noArray(), kp2, cpu_desc2);
                }

                frame_idx++;

                if(frame_idx<1)
                    continue;

                bAdvanceFrame = false;
            }
        }

        // SURF
        vector <Point2Df> src, dst;
        vector <float> match_score;
        vector<vector<DMatch>> knnMatches;

        {
            gettimeofday(&t1, NULL);
            // SURF does not work for particularly small area, so we use whole image, and then select interesting points
            //if(g_bHaveBox && 0) { 
            //    obj_subimg = img1(g_rectangle);
            //    surf(obj_subimg, cv::cuda::GpuMat(), gpu_kp1, gpu_desc1);
            //} 

            if(!bUseVideo) {
                surf(img1, cv::cuda::GpuMat(), gpu_kp1, gpu_desc1);
                surf(img2, cv::cuda::GpuMat(), gpu_kp2, gpu_desc2);

                surf.downloadKeypoints(gpu_kp1, kp1);
                surf.downloadDescriptors(gpu_desc1, descriptors1);
                surf.downloadKeypoints(gpu_kp2, kp2);
                surf.downloadDescriptors(gpu_desc2, descriptors2);
            }

            if(g_bUseGpuSURF) {
                // TODO: only match subset which falls in rectangle, but how to do it on GPU?
                // by using mask? too much hassle as it should contain every match for src and dst so it will be src x dst size quite big
                for(int i=0;i<(int)train_desc.size();++i) {
                    vector<vector<DMatch>> matches;
                    matcher->knnMatch(train_desc[i], gpu_desc2, matches, 2);//, train_desc_mask[i]);
                    for(vector<DMatch>& ms: matches) {
                        for(DMatch& m: ms) {
                            m.queryIdx += train_kp_offsets[i];
                        }
                    }
                    knnMatches.insert(knnMatches.end(), matches.begin(), matches.end());
                }
                //flann_matcher->knnMatch(cpu_desc1, cpu_desc2, knnMatches, 2);
            } else {
                bf_hamming_matcher.knnMatch(cpu_desc1, cpu_desc2, knnMatches, 2);
            }

            printf("GPU SURF: %g ms\n", TimeDiff(t1,t2));
        }

        {
            float min_val = FLT_MAX;
            float max_val = 0.0f;

            goodMatches.clear();

            //-- Filter matches using the Lowe's ratio test
            int max_count = 1000;
            float ratio_test = 0.75f;
            for(int i=0; i< knnMatches.size(); ++i) {
                const vector<DMatch>& m = knnMatches[i];
                if(m.size() == 0) {
                    continue;
                }
                const float dist = m[0].distance;
                if(dist < ratio_test * m[1].distance) {

                    Point2f src_pt = kp1[m[0].queryIdx].pt;
                    Point2f dst_pt = kp2[m[0].trainIdx].pt;

                    if(!g_bHaveBox || g_orig_track_rectangle.contains(src_pt)) {

                        goodMatches.push_back(m[0]);

                        src.push_back({src_pt.x, src_pt.y});
                        dst.push_back({dst_pt.x, dst_pt.y});
                        match_score.push_back(dist);

                        min_val = dist < min_val ? dist : min_val;
                        max_val = dist > max_val ? dist : max_val;
                    }

                    if(goodMatches.size() >= max_count) {
                        break;
                    }
                }
            }

            // Flip score
            for(unsigned int i=0; i < match_score.size(); i++) {
                match_score[i] = max_val - match_score[i] + min_val;
            }

            // drawing the results
            //Mat img_matches;
            //vector<vector<char>> matches_mask;
            //drawMatches(Mat(img1), kp1, Mat(img2), kp2, knnMatches, img_matches, Scalar::all(-1), Scalar::all(0), matches_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
        }

        // OpenCV homography
        vector<char> open_cv_status;
        if(goodMatches.size()>=MIN_GOOD_MATCHES)
        {
            gettimeofday(&t1, NULL);

            Mat src2(src.size(), 2, CV_32F);
            Mat dst2(dst.size(), 2, CV_32F);

            for(unsigned int i=0; i < src.size(); i++) {
                src2.at<float>(i,0) = src[i].x;
                src2.at<float>(i,1) = src[i].y;
            }

            for(unsigned int i=0; i < dst.size(); i++) {
                dst2.at<float>(i,0) = dst[i].x;
                dst2.at<float>(i,1) = dst[i].y;
            }

            // LMEDS/RHO
            OpenCV_H = findHomography(src2, dst2, open_cv_status, cv::RANSAC, INLIER_THRESHOLD);
            opencv_inliers = accumulate(open_cv_status.begin(), open_cv_status.end(), 0);

            if(!OpenCV_H.empty()) {
                opencv_inliers  = 0;
                vector<DMatch> good_matches;
                vector<KeyPoint> inliers1, inliers2;
                for(size_t i = 0; i < src.size(); i++) {
                    Mat col = Mat::ones(3, 1, CV_64F);
                    col.at<double>(0) = src[i].x;
                    col.at<double>(1) = src[i].y;
                    col = OpenCV_H * col;
                    col /= col.at<double>(2);
                    float x = col.at<double>(0);
                    float y = col.at<double>(1);
                    double dist = sqrt( pow(x - dst[i].x, 2) +
                            pow(y - dst[i].y, 2));
                    const bool b_good_match = dist < INLIER_THRESHOLD;
                    open_cv_status[i] = b_good_match;
                    opencv_inliers += b_good_match ? 1 : 0;
                }
            }

            gettimeofday(&t2, NULL);
            printf("RANSAC Homography (OpenCV): %g ms\n", TimeDiff(t1,t2));
        }

        if(goodMatches.size()>=MIN_GOOD_MATCHES)
        {
            // Homography
            {
                gettimeofday(&t1, NULL);

                K = (int)(log(1.0 - CONFIDENCE) / log(1.0 - pow(INLIER_RATIO, 4.0)));

                CUDA_RANSAC_Homography(src, dst, match_score, INLIER_THRESHOLD, K, &best_inliers, best_H, &inlier_mask);

                gettimeofday(&t2, NULL);

                printf("RANSAC Homography (GPU): %g ms\n", TimeDiff(t1,t2));
            }

            // Refine homography
            if(best_inliers >= MIN_GOOD_MATCHES)
            {
                gettimeofday(&t1, NULL);

                Mat src2(best_inliers, 2, CV_32F);
                Mat dst2(best_inliers, 2, CV_32F);

                int k = 0;
                for(unsigned int i=0; i < src.size(); i++) {
                    if(inlier_mask[i] == 0) {
                        continue;
                    }

                    src2.at<float>(k,0) = src[i].x;
                    src2.at<float>(k,1) = src[i].y;

                    dst2.at<float>(k,0) = dst[i].x;
                    dst2.at<float>(k,1) = dst[i].y;

                    k++;
                }

                vector<uchar> status;
                //refined_H = findHomography(src2, dst2, status, 0 /* Least square */);
                refined_H = findHomography(src2, dst2, status, cv::RANSAC, INLIER_THRESHOLD);

                // do not update best_H if findHomography failed
                if(!refined_H.empty()) {
                    k =0;
                    for(int y=0; y < 3; y++) {
                        for(int x=0; x < 3; x++) {
                            best_H[k] = refined_H.at<double>(y,x);
                            k++;
                        }
                    }
                }

                best_inliers = 0;
                for(int i=0; i < src.size(); i++) {
                    float x = best_H[0]*src[i].x + best_H[1]*src[i].y + best_H[2];
                    float y = best_H[3]*src[i].x + best_H[4]*src[i].y + best_H[5];
                    float z = best_H[6]*src[i].x + best_H[7]*src[i].y + best_H[8];

                    x /= z;
                    y /= z;

                    float dist_sq = (dst[i].x - x)*(dst[i].x- x) + (dst[i].y - y)*(dst[i].y - y);

                    if(dist_sq < INLIER_THRESHOLD*INLIER_THRESHOLD) {
                        best_inliers++;
                    } else {
                        // modify inlier mask after refining
                        inlier_mask[i] = 0;
                    }
                }

                gettimeofday(&t2, NULL);
            }
            printf("Refine homography: %g ms\n", TimeDiff(t1,t2));
        }


        if(g_bUseGPUHomography) {
            myDrawMatches(out, cpu_img1, cpu_img2, goodMatches, kp1, kp2, g_bDrawCircles, &inlier_mask);
        } else {
            myDrawMatches(out, cpu_img1, cpu_img2, goodMatches, kp1, kp2, g_bDrawCircles, 
                open_cv_status.size() == goodMatches.size()? &open_cv_status : nullptr);
        }

        if (g_bHaveBox) {
            DrawRectangle(out, g_orig_track_rectangle, g_bTrack?Scalar(255,0,0):Scalar(0,255,0));
        }

        gettimeofday(&t2, NULL);

        printf("--------------------------------\n");
        printf("Total time: %g ms\n", TimeDiff(start_time,t2));

        printf("\n");
        printf("Features extracted: %d %d\n", (int)kp1.size(), (int)kp2.size());
        printf("OpenCV Inliers: %d\n", opencv_inliers);
        printf("GPU Inliers: %d\n", best_inliers);
        printf("GPU RANSAC iterations: %d\n", K);
        printf("\n");
        printf("RANSAC parameters:\n");
        printf("Confidence: %g\n", CONFIDENCE);
        printf("Inliers ratio: %g\n", INLIER_RATIO);
#ifdef NORMALISE_INPUT_POINTS
        printf("Data is normalised: yes\n");
#else
        printf("Data is normalised: no\n");
#endif

#ifdef BIAS_RANDOM_SELECTION
        printf("Bias random selection: yes\n");
#else
        printf("Bias random selection: no\n");
#endif

        printf("\n");
        printf("Homography matrix\n");

        for(int i=0; i < 9; i++) {
            printf("%g ", best_H[i]/best_H[8]);

            if((i+1) % 3 == 0 && i > 0) {
                printf("\n");
            }
        }

        if(g_bHaveBox) {
            const Point2f y_off = Point2f(0, cpu_img1.rows);
            const Rect src_rect = g_orig_track_rectangle;

            std::vector<Point2f> opencv_pts;
            std::vector<Point2f> gpu_pts;

            if(!OpenCV_H.empty()) {
                opencv_pts = transformBox(src_rect, OpenCV_H);
                drawPolyline(out, opencv_pts, y_off, Scalar(255, 0, 0), 1);

                if(g_bTrack && !g_bUseGPUHomography) {
                    g_rectangle = rect_from_points(opencv_pts);
                    g_rectangle.y += y_off.y;
                    DrawRectangle(out, g_rectangle, Scalar(200,200,200), 2);
                }

            }
            if(!refined_H.empty()) {
                gpu_pts = transformBox(src_rect, refined_H);
                drawPolyline(out, gpu_pts, y_off, Scalar(0, 0, 255), 1);

                if(g_bTrack && g_bUseGPUHomography) {
                    g_rectangle = rect_from_points(gpu_pts);
                    g_rectangle.y += y_off.y;
                    DrawRectangle(out, g_rectangle, Scalar(200,200,200), 2);
                }
            }

        }

        // pose reconstruction
        if(!OpenCV_H.empty()) {
            Mat RotM, TrVec; // 3x3, 3x1
            reconstructPose(OpenCV_H, RotM, TrVec);
            drawPose(out, RotM, TrVec, cpu_img1.cols/2, cpu_img1.rows/2); 
        }

        // drawText
        {
            String text = g_bUseGPUHomography ? "GPU" : "OpenCV";
            int fontFace = FONT_HERSHEY_PLAIN;
            double fontScale = 1;
            int thickness = 1;

            int baseline=0;
            Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            baseline += thickness;

            // center the text
            Point textOrg((out.cols - textSize.width)/2, out.rows - textSize.height);
            Point status_coords(0, out.rows - 2*textSize.height);

            putText(out, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
            char status[256] = {0};
            sprintf(status, "%s %s", g_bAccumulate?"Accum.":"", g_bTrack?"Track":"");
            String st = status;
            putText(out, st, status_coords, fontFace, fontScale, Scalar::all(255), thickness, 8);
        }

        imshow("matches", out);

        const int Key = waitKey(0);
        switch(Key) {
            case 27: // ESC
                     g_bExit = true;
                     break;
            case ' ':
                     bAdvanceFrame = true;
                     break;
            case 'c':
                     g_bDrawCircles  = !g_bDrawCircles;
                     break;
            case 'g':
                     g_bUseGPUHomography = !g_bUseGPUHomography;
                     break;
            case 'a':
                     g_bAccumulate = !g_bAccumulate;
                     break;
            case 't':
                     g_bTrack = !g_bTrack;
                     break;
            case 'b':
                     g_bHaveBox = false;
                     break;
        }
    }

    return 0;
}
