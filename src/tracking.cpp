#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "tracker_nano.h"
 
using namespace cv;
using namespace std;
 
int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[8] = {//"BOOSTING", "TLD","MEDIANFLOW", "MOSSE", 
                              "MIL", "KCF", "GOTURN", "CSRT", "NANO", "DaSiamRPN"};
                           //  0        1       2       3         4        5
 
    // Create a tracker
    string trackerType = trackerTypes[4];
 
    Ptr<Tracker> tracker;

    my::TrackerNano::Params p; 
    p.backbone = "./data/nanotrack_backbone_sim.onnx";
    p.neckhead = "./data/nanotrack_head_sim.onnx";
    p.backend = cv::dnn::Backend::DNN_BACKEND_CUDA;
    p.target = cv::dnn::Target::DNN_TARGET_CUDA_FP16;
    Ptr<my::TrackerNano> my_tracker = my::TrackerNano::create(p);
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        //if (trackerType == "BOOSTING")
         //   tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        //if (trackerType == "TLD")
         //   tracker = TrackerTLD::create();
        //if (trackerType == "MEDIANFLOW")
         //   tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        //if (trackerType == "MOSSE")
        //    tracker = TrackerMOSSE::create();
        if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
        if (trackerType == "NANO") {
            TrackerNano::Params p; 
            p.backbone = "./data/nanotrack_backbone_sim.onnx";
            p.neckhead = "./data/nanotrack_head_sim.onnx";
            p.backend = cv::dnn::Backend::DNN_BACKEND_CUDA;
            p.target = cv::dnn::Target::DNN_TARGET_CUDA_FP16;
            tracker = TrackerNano::create(p);
        }
        if (trackerType == "DaSiamRPN") {
            TrackerDaSiamRPN::Params p; 
            p.model = "./data/dasiamrpn_model.onnx";
            p.kernel_r1 = "./data/dasiamrpn_kernel_r1.onnx";
            p.kernel_cls1 = "./data/dasiamrpn_kernel_cls1.onnx";
            p.backend = cv::dnn::Backend::DNN_BACKEND_CUDA;
            p.target = cv::dnn::Target::DNN_TARGET_CUDA_FP16;
            tracker = TrackerDaSiamRPN::create(p);
        }
    }
    #endif

    VideoCapture cap;
    bool bFoundVideo = false;
    bool bWriteVideo = true;


    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--video" && argc>i+1)
        {
            if(!cap.open(argv[++i])) {
                fprintf(stderr, "Cannot open video input file: %s\n", argv[i]);
                return -1;
            }
            bFoundVideo = true;
        }
    }

    if(!bFoundVideo) { 
        fprintf(stderr, "No video could be opened\n");
        return 1;
    }

    const int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    VideoWriter video;
    if(bWriteVideo) {
        if(!video.open("out.avi", cv::VideoWriter::fourcc('M','J','P','G'), 60, Size(frame_width,frame_height))) {
            fprintf(stderr, "Failed to open video file for writing\n");
            return 1;
        }
    }

    // Read video
     
    // Exit if video is not opened
    if(!cap.isOpened())
    {
        cout << "Could not read video file" << endl; 
        return 1; 
    } 
 
    // Read first frame 
    Mat frame; 
    bool ok = cap.read(frame); 
 
    // Define initial bounding box 
    Rect bbox(287, 23, 86, 320); 
    Rect my_bbox; 
 
    // Uncomment the line below to select a different bounding box 
    bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 
 
    imshow("Tracking", frame); 
    tracker->init(frame, bbox);
    my_tracker->init(frame, bbox);
    bool bIsInitialized = true;
    //
     
    while(cap.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = false;
        if(bIsInitialized) {
            ok = tracker->update(frame, bbox);
            ok = my_tracker->update(frame, my_bbox);
        }
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok) {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            rectangle(frame, my_bbox, Scalar(255, 255, 255 ), 2, 1 );
        } else if(bIsInitialized) {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        } else {
            putText(frame, "Please select ROI by pressing 's'", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,255,0),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        char buf[128] = {0};
        sprintf(buf, "FPS: %.3f", fps);
        putText(frame, buf, Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        if(bWriteVideo && video.isOpened()) {
            video.write(frame);
        }
 
        // Display frame.
        imshow("Tracking", frame);


         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        if(k == 's') {
            bbox = selectROI("selection", frame, false); 
            tracker->init(frame, bbox);
            my_tracker->init(frame, bbox);
            bIsInitialized = true;
        }
 
    }
}

