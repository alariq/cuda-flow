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


// Calc the theoretical number of iterations using some conservative parameters
const double CONFIDENCE = 0.99;
const double INLIER_RATIO = 0.18; // Assuming lots of noise in the data!
const double INLIER_THRESHOLD = 3.0; // pixel distance

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

Rect g_rectangle;
bool g_bDrawingBox = false;
bool g_bStartedDrawingBox = false;
bool g_bHaveBox = false;
RNG g_rng(0);  // Generate random number

void DrawRectangle(Mat& img, Rect box)
{
	//Draw a rectangle with random color
	rectangle(img, box.tl(), box.br(), Scalar(g_rng.uniform(0, 255),
					g_rng.uniform(0,255),g_rng.uniform(0,255)));
}

void on_MouseHandle(int event, int x, int y, int flags, void* param) {
	Mat& image = *(cv::Mat*) param;
    switch (event) {
        case EVENT_MOUSEMOVE: {    // When mouse moves, get the current rectangle's width and height
                                  if (g_bStartedDrawingBox) {
                                      g_rectangle.width = x - g_rectangle.x;
                                      g_rectangle.height = y - g_rectangle.y;
                                  }
                              }
                              break;
        case EVENT_LBUTTONDOWN: {  // when the left mouse button is pressed down,
                                   //get the starting corner's coordinates of the rectangle
                                   printf("started drawing\n");
                                    g_bStartedDrawingBox = true;
                                    g_rectangle = Rect(x, y, 0, 0);
                                }
                                break;
        case EVENT_LBUTTONUP: {   //when the left mouse button is released,
                                  //draw the rectangle
                                  if(g_bStartedDrawingBox) {
                                      g_bHaveBox = true;
                                       printf("have box\n");
                                  }
                                  g_bStartedDrawingBox = false;
                                  if (g_rectangle.width < 0) {
                                      g_rectangle.x += g_rectangle.width;
                                      g_rectangle.width *= -1;
                                  }

                                  if (g_rectangle.height < 0) {
                                      g_rectangle.y += g_rectangle.height;
                                      g_rectangle.height *= -1;
                                  }
                                  //DrawRectangle(image, g_rectangle);
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

Mat myDrawMatches(Mat grey1, Mat grey2, const std::vector<DMatch> matches, 
        const std::vector<KeyPoint>& kp1, const std::vector<KeyPoint>& kp2, bool b_draw_area = false) {

    //Mat grey1(a), grey2(b);

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

    RNG match_color_rng(0);  // Generate random number

    for(unsigned int i=0; i < matches.size(); i++) {
        int r = match_color_rng.uniform(0, 255);
        int g = match_color_rng.uniform(0, 255);
        int b = match_color_rng.uniform(0, 255);

        const int idx1 = matches[i].queryIdx;
        const int idx2 = matches[i].trainIdx;
        Point pt1 = Point(kp1[idx1].pt.x, kp1[idx1].pt.y);
        Point pt2 = Point(kp2[idx2].pt.x, grey1.rows + kp2[idx2].pt.y);
        Scalar c = CV_RGB(r,g,b);

        line(result, pt1, pt2, c, 1, cv::LINE_AA);
        if(b_draw_area) {
            circle(result, pt1, kp1[idx1].size/2, c, 1, cv::LINE_AA);
            circle(result, pt2, kp2[idx2].size/2, c, 1, cv::LINE_AA);
        }
    }

    return result;
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
            cap.open(argv[++i]);
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
    Mat out;

    cv::cuda::SURF_CUDA surf;
    cv::cuda::GpuMat gpu_kp1, gpu_kp2;
    cv::cuda::GpuMat gpu_desc1, gpu_desc2;
    cv::cuda::GpuMat gpu_ret_idx, gpu_ret_dist, gpu_all_dist;
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());

    vector<DMatch> goodMatches;

    Mat OpenCV_H;
    Mat refined_H;
    int opencv_inliers;
    int K;

    while(!g_bExit)
    {

        // SURF
        vector <Point2Df> src, dst;
        vector <float> match_score;
        vector<vector<DMatch>> knnMatches;

        {
            gettimeofday(&t1, NULL);

            if(g_bHaveBox && 0 ) { // SURF does nor work for particularly small area, so we use whole image, and then select interesting points
                obj_subimg = img1(g_rectangle);
                surf(obj_subimg, cv::cuda::GpuMat(), gpu_kp1, gpu_desc1);
            } else {
                surf(img1, cv::cuda::GpuMat(), gpu_kp1, gpu_desc1);
            }
            surf(img2, cv::cuda::GpuMat(), gpu_kp2, gpu_desc2);

            matcher->knnMatch(gpu_desc1, gpu_desc2, knnMatches, 2);

            vector<float> descriptors1, descriptors2;
            surf.downloadKeypoints(gpu_kp1, kp1);
            surf.downloadKeypoints(gpu_kp2, kp2);
            surf.downloadDescriptors(gpu_desc1, descriptors1);
            surf.downloadDescriptors(gpu_desc2, descriptors2);

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
                const float dist = m[0].distance;
                if(dist < ratio_test * m[1].distance) {

                    Point2f src_pt = kp1[m[0].queryIdx].pt;
                    Point2f dst_pt = kp2[m[0].trainIdx].pt;

                    if(!g_bHaveBox || g_rectangle.contains(src_pt))
                        goodMatches.push_back(m[0]);

                    src.push_back({src_pt.x, src_pt.y});
                    dst.push_back({dst_pt.x, dst_pt.y});
                    match_score.push_back(dist);
                    min_val = dist < min_val ? dist : min_val;
                    max_val = dist > max_val ? dist : max_val;

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
            
            //vector<char> matches_mask;
            //RNG match_color_rng(0);  // Generate random number
            //drawMatches(Mat(img1), kp1, Mat(img2), kp2, goodMatches, img_matches, match_color_rng.uniform(0, 255), Scalar::all(0), matches_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            //imshow("matches", img_matches);
            //waitKey();

            out = myDrawMatches(cpu_img1, cpu_img2, goodMatches, kp1, kp2, g_bDrawCircles);
            if (g_bHaveBox) {
                DrawRectangle(out, g_rectangle);
            }
        }

        // OpenCV homography
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

            vector<uchar> status;

            // LMEDS/RHO
            OpenCV_H = findHomography(src2, dst2,status, cv::RANSAC , INLIER_THRESHOLD);
            opencv_inliers = accumulate(status.begin(), status.end(), 0);


            gettimeofday(&t2, NULL);
            printf("RANSAC Homography (OpenCV): %g ms\n", TimeDiff(t1,t2));
        }

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
                refined_H = findHomography(src2, dst2, status, 0 /* Least square */);

                k =0;
                for(int y=0; y < 3; y++) {
                    for(int x=0; x < 3; x++) {
                        best_H[k] = refined_H.at<double>(y,x);
                        k++;
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

                    if(dist_sq < INLIER_THRESHOLD) {
                        best_inliers++;
                    }
                }

                gettimeofday(&t2, NULL);
            }
            printf("Refine homography: %g ms\n", TimeDiff(t1,t2));
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

        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(g_rectangle.x, g_rectangle.y);
        obj_corners[1] = Point2f(g_rectangle.x + g_rectangle.width, g_rectangle.y);
        obj_corners[2] = Point2f(g_rectangle.x + g_rectangle.width, g_rectangle.y + g_rectangle.height);
        obj_corners[3] = Point2f(g_rectangle.x, g_rectangle.y + g_rectangle.height);

        Mat H = g_bUseGPUHomography ? refined_H : OpenCV_H;
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform(obj_corners, scene_corners, H);
        Point2f y_off = Point2f(0, cpu_img1.rows);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( out, scene_corners[0] + y_off, scene_corners[1] + y_off, Scalar(0, 255, 0), 4 );
        line( out, scene_corners[1] + y_off, scene_corners[2] + y_off, Scalar( 0, 255, 0), 4 );
        line( out, scene_corners[2] + y_off, scene_corners[3] + y_off, Scalar( 0, 255, 0), 4 );
        line( out, scene_corners[3] + y_off, scene_corners[0] + y_off, Scalar( 0, 255, 0), 4 );


        {
        String text = g_bUseGPUHomography ? "GPU" : "OpenCV";
        int fontFace = FONT_HERSHEY_PLAIN;
        double fontScale = 1;
        int thickness = 1;

        int baseline=0;
        Size textSize = getTextSize(text, fontFace,
                                    fontScale, thickness, &baseline);
        baseline += thickness;

        // center the text
        Point textOrg((out.cols - textSize.width)/2,
                      out.rows - textSize.height);

        putText(out, text, textOrg, fontFace, fontScale,
            Scalar::all(255), thickness, 8);
        }

        imshow("matches", out);

        const int Key = waitKey(0);
        switch(Key) {
            case 27: // ESC
                     g_bExit = true;
                     break;
            case ' ':
                     g_bDrawCircles  = !g_bDrawCircles;
                     break;
            case 'g':
                     g_bUseGPUHomography = !g_bUseGPUHomography;
                     break;
            case 'b':
                     g_bHaveBox = false;
                     break;
        }
    }

    return 0;
}
