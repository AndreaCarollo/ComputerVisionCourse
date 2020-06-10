// /*--------------------------------------------
// * Usage:
// * example_tracking_multitracker <video_name> [algorithm]
// *
// * example:
// * example_tracking_multitracker Bolt/img/%04d.jpg
// * example_tracking_multitracker faceocc2.webm KCF
// *--------------------------------------------------*/

// #include <opencv2/core/utility.hpp>
// #include <opencv2/tracking/tracker.hpp>
// #include <opencv2/tracking/tracking.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/highgui.hpp>
// #include <iostream>
// #include <cstring>
// #include <ctime>

// using namespace std;
// using namespace cv;

// int main(int argc, char **argv)
// {
//     // show help
//     if (argc < 2)
//     {
//         cout << " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
//                 " examples:\n"
//                 " example_tracking_multitracker Bolt/img/%04d.jpg\n"
//                 " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
//              << endl;
//         return 0;
//     }

//     // set the default tracking algorithm
//     std::string trackingAlg = "KCF";

//     // set the tracking algorithm from parameter
//     // if (argc > 2)
//     //     trackingAlg = argv[2];

//     // create the tracker
//     // MultiTracker trackers(trackingAlg);
//     MultiTracker trackers;

//     // container of the tracked objects
//     vector<Rect2d> objects;

//     // set input video
//     std::string video = argv[1];
//     VideoCapture cap(video);

//     Mat frame;

//     // get bounding box
//     cap >> frame;
//     selectROI("tracker", frame, objects);

//     //quit when the tracked object(s) is not provided
//     if (objects.size() < 1)
//         return 0;

//     // initialize the tracker
//     trackers.add(frame, objects);

//     // do the tracking
//     printf("Start the tracking process, press ESC to quit.\n");
//     for (;;)
//     {
//         // get frame from the video
//         cap >> frame;

//         // stop the program if no more images
//         if (frame.rows == 0 || frame.cols == 0)
//             break;

//         //update the tracking result
//         trackers.update(frame);

//         // draw the tracked object
//         for (unsigned i = 0; i < trackers.objects.size(); i++)
//             rectangle(frame, trackers.objects[i], Scalar(255, 0, 0), 2, 1);

//         // show image with the tracked object
//         imshow("tracker", frame);

//         //quit on ESC button
//         if (waitKey(1) == 27)
//             break;
//     }
// }
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking_by_matching.hpp>
#include <opencv2/tracking.hpp>
// #include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/highgui.hpp>
#include <iostream>
// #include <cstring>
using namespace std;
using namespace cv;
int main(int argc, char **argv)
{
    // show help
    // if (argc < 2)
    // {
    //     cout << " Usage: tracker <video_name>\n"
    //             " examples:\n"
    //             " example_tracking_kcf Bolt/img/%04d.jpg\n"
    //             " example_tracking_kcf faceocc2.webm\n"
    //          << endl;
    //     return 0;
    // }
    // declares all required variables
    Rect2d roi;
    Mat frame;
    // create a tracker object
    Ptr<Tracker> MyTracker = TrackerCSRT::create();
    // set input video
    //std::string video = argv[1];
    //VideoCapture cap(video);
    // get bounding box
    //string path_to_img = "./videos/run.mp4";
    string path_to_img = "./Video/img1/%06d.jpg";
    //string path_to_vid = ".Video/people.mp4";
    // load the sequence of images
    VideoCapture sequence(path_to_img);
    sequence>>frame;
    roi = selectROI("tracker", frame);
    //quit if ROI was not selected
    if (roi.width == 0 || roi.height == 0)
        return 0;
    // initialize the tracker
    MyTracker->init(frame, roi);
    // perform the tracking process
    printf("Start the tracking process, press ESC to quit.\n");
    for (int i = 0; i<795; i++)
    {
        // get frame from the video
        sequence >> frame;
        // stop the program if no more images
        if (!sequence.isOpened())
            break;
        // update the tracking result
        MyTracker->update(frame, roi);
        // draw the tracked object
        rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
        // show image with the tracked object
        imshow("tracker", frame);
        //quit on ESC button
        if (waitKey(1) == 27)
            break;
    }
    return 0;
}