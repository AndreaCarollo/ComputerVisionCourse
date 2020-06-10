//************************************************//
//            University of Trento                //
//                                                //
//    Department of Industrial Engineering        //
//    Course:  Computer Vision                    //
//                                                //
//    Student: Carollo Andrea                     //
//    Mat:     206343                             //
//    email:   andrea.carollo@studenti.unitn.it   //
//                                                //
//************************************************//

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ctime>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>

#include "./Hungarian.h"

using namespace cv;
using namespace std;

void ss(const char *ground_truth, vector<Rect> *exact_points, int i);

Point center_rect(Rect Rettangolo);

int main()
{
    // set timer to calculate performances
    time_t tstart, tend;
    tstart = time(0);

    /************ Generic Parameters ************/
    // TMP refresh tracker
    int refresh_tracker = 6;
    int threshold_distance = 9;

    // threshold of overlapping area of rectangles
    float threshold_kill_BG = 0.01;
    float threshold_intersection_ratio = 0.90;

    // Output detection
    // one row each detection structured as following:
    // <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>
    FILE *detectionFile;
    detectionFile = fopen("./Output/detection.txt", "w");

    // Output tracking
    FILE *trackingFile;
    trackingFile = fopen("./Output/tracking.txt", "w");

    FILE *pFile;
    pFile = fopen("./Output/log.txt", "w");

    // set input video
    string path_to_img = "./Video/img1/%06d.jpg";
    VideoCapture sequence(path_to_img);

    namedWindow("tracker + detector", WINDOW_NORMAL);
    // cv::resizeWindow("tracker + detector", 1920, 1080);

    // False positive vector, each element indicate number of false positive into frame
    vector<int> FP;
    vector<int> Missed;
    vector<float> error_average;

    // create matrix for frames
    Mat frame;
    Mat frame_det; // just used to plot rectangles of detection

    /************ Parameters for the Detector ************/
    // set parameters for the HOG descriptor
    HOGDescriptor hog(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    // load Daimler People Detector pre trained classifier with previous parameters
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    vector<Rect> pedestrian;

    /************ Variables for MOG2 ************/
    Mat frame_gray;
    Mat motion_mask;
    Mat motion_mask_ed;
    Mat bg;
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    int history = 100;
    int nmixture = 5;
    double backgroundRatio = 0.7;
    double learningRate = 0.1;
    double noiseSignal = 10;
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0), false);

    /************ Parameters for Erosion & Dilation ************/
    int hsize = 16;
    float hranges[] = {0, 180};
    const float *phranges = hranges;
    int erosion_elem = 1;
    int erosion_size = 1;
    int dilation_elem = 3;
    int dilation_size = 3;
    int const max_elem = 4;
    int const max_kernel_size = 21;

    /***** Tracker Initialization *****/
    // initialize the multitracker
    MultiTracker trackers;
    std::vector<Ptr<Tracker>> algorithms;

    // create container of the tracked objects
    vector<Rect2d> objects;

    // tracker's Region of Interest vector
    vector<Rect> ROIs;

    // declare vector of points for trajectories
    vector<vector<Point>> trajectories;
    vector<int> IDs;

    Mat frame_test;
    /************ Begin analysis of video ************/
    // Start loop processing
    printf("Start the tracking process, press ESC to quit.\n");
    for (int i = 0; i < 1000; i++)
    {
        // load frame
        sequence >> frame;
        frame.copyTo(frame_det);
        frame.copyTo(frame_test);

        // exit from the for cycle if the video is finished
        if (frame.empty())
        {
            cout << "End of the video\n"
                 << endl;
            break;
        }

        /***** Compute Motion Mask *****/
        pMOG2->apply(frame, motion_mask, learningRate);
        pMOG2->getBackgroundImage(bg);

        /***** Erosion & Dilation of the mask ****/
        // clone motion_mask to compute erosion and dilation
        motion_mask.copyTo(motion_mask_ed);
        Mat element = getStructuringElement(MORPH_ELLIPSE,
                                            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            Point(erosion_size, erosion_size));
        erode(motion_mask_ed, motion_mask_ed, element);
        Mat element2 = getStructuringElement(MORPH_RECT,
                                             Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                             Point(dilation_size, dilation_size));
        dilate(motion_mask_ed, motion_mask_ed, element2);

        /***** Compute Detector *****/
        hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(16, 16), 1.05, 2, true);
        vector<float> descriptorValues;
        hog.compute(frame, descriptorValues);

        // parse detected pedestrian for delete too big rectangles -> error of detection
        // create temporary vector to fill with accepted rectangles
        vector<Rect> tmp_pedestrian;
        for (int k = 0; k < pedestrian.size(); k++)
        {
            if (!(pedestrian[k].width > 90) || !(pedestrian[k].height > 180))
            {
                tmp_pedestrian.push_back(pedestrian[k]);
            }
        }
        pedestrian = tmp_pedestrian;

        // stamp detections on output "detection.txt" file
        // <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>
        for (int k = 0; k < pedestrian.size(); k++)
        {
            int bb_left = pedestrian[k].x;
            int bb_top = pedestrian[k].y;
            int bb_width = pedestrian[k].width;
            int bb_height = pedestrian[k].height;

            fprintf(detectionFile, "%d,%03d,%03d,%03d,%03d\n", i + 1, bb_left, bb_top, bb_width, bb_height);
        }

        /***** Refresh of the MultiTracker *****/
        // first save all statement of the tracker

        // if track something not on the mog2 -> kill it
        // use  >> countwhite(Mat mask, Rect src) <<
        trackers.update(frame);
        bool flag_refresh = false;

        // create a policy to refresh the tracker if there are some new entries from the detector
        // simply if the new entries do not overlap the already tracked pedestrain
        for (int k = 0; k < pedestrian.size(); k++)
        {
            for (int p = 0; p < trackers.getObjects().size(); p++)
            {
                if (!(((Rect)pedestrian[k] & (Rect)trackers.getObjects()[p]).area() > 0))
                {
                    // set true and break cycle
                    flag_refresh = true;
                    break;
                }
            }
            if (flag_refresh == true)
            {
                break;
            }
        }

        // refresh the tracker every refresh rate frames or if the previous policy is applied
        // to avoid bug, do not refresh if the detector has not find pedestrian
        if ((i % refresh_tracker == 0 || flag_refresh == true) && !(pedestrian.empty()))
        {

            // Let's look for trackers to remove
            vector<int> tkr_to_remove;
            for (int k = 0; k < trackers.getObjects().size(); k++)
            {
                // Using MOG2 -> information of motion
                // taken the rectangle of the tracker and calculate number of white pixels in the motion mask
                // the calculate the ratio of white over the overall area of the crop
                Mat motion_mask_ed_copy = motion_mask_ed;
                Mat cropedImage;
                Point2i tmp_tl;
                tmp_tl.x = (int)trackers.getObjects()[k].tl().x;
                tmp_tl.y = (int)trackers.getObjects()[k].tl().y;
                Point2i tmp_br;
                tmp_br.x = (int)trackers.getObjects()[k].br().x;
                tmp_br.y = (int)trackers.getObjects()[k].br().y;
                Rect crop = Rect(tmp_tl, tmp_br);
                //printf("f %03d, tl(%03d, %03d) - br(%03d, %03d) - ", i, tmp_tl.x, tmp_tl.y, tmp_br.x, tmp_br.y);
                // cropedImage = motion_mask_ed_copy( crop );

                /* 
                    Those lines of "try & catch" are needed because sometime the crop action do not work properly
                    and create some exception on the dimensions of the cropped rectangles from
                    the motion mask 
                */
                Mat *ROI;
                try
                {
                    ROI = new Mat(motion_mask_ed_copy, crop);
                }
                catch (cv::Exception exc)
                {
                    continue;
                }
                if (ROI == NULL)
                    continue;
                ROI->copyTo(cropedImage);
                delete ROI;

                // count white pixels inside the cropped region of interest
                int white = countNonZero(cropedImage);
                // measure the area of the cropped region
                int area_crop = cropedImage.rows * cropedImage.cols;

                // policy to remove tracker if the quantity of white pixel are under a certain percentage
                // this means that the tracker maybe has drift or lost the subject
                if ((float)white / (float)area_crop < (float)threshold_kill_BG)
                {
                    tkr_to_remove.push_back(k);
                }
            }

            // re-initialization of the multitracker with new vector of objects and algorithms
            std::vector<Ptr<Tracker>> new_algorithms;
            vector<Rect> new_ROIs;
            vector<Rect> old_ROIs;
            vector<Rect2d> new_objects;

            // new_ROIs.clear();    // TODO: check if the code work without
            // old_ROIs.clear();    // TODO: check if the code work without

            // find new pedestrian respect to the already tracked pedestrian
            for (int k = 0; k < trackers.getObjects().size(); k++)
            {
                // jump the value of the to_remove_tkr (that are drifted somewhere)
                if (!tkr_to_remove.empty())
                {
                    /* v is non-empty */
                    // control to jump if the element is already on the tracked elements
                    // if the index is present on the tkr_to_remove vector, jump that index
                    if (std::find(tkr_to_remove.begin(), tkr_to_remove.end(), k) == tkr_to_remove.end())
                    {
                        // collect "old" ROIs from the tracker, by rebuilding rectangles
                        int tmp_x = trackers.getObjects()[k].x;
                        int tmp_y = trackers.getObjects()[k].y;
                        int tmp_w = trackers.getObjects()[k].width;
                        int tmp_h = trackers.getObjects()[k].height;
                        Rect tmp_rect = Rect(tmp_x, tmp_y, tmp_w, tmp_h);
                        old_ROIs.push_back(tmp_rect);
                    }
                }
                else
                {
                    // collect ALL "old" ROIs from the tracker, by rebuilding rectangles
                    int tmp_x = trackers.getObjects()[k].x;
                    int tmp_y = trackers.getObjects()[k].y;
                    int tmp_w = trackers.getObjects()[k].width;
                    int tmp_h = trackers.getObjects()[k].height;
                    Rect tmp_rect = Rect(tmp_x, tmp_y, tmp_w, tmp_h);
                    old_ROIs.push_back(tmp_rect);
                }
            }

            // Compare the rectangles tracking to looks for duplicates
            // comparison is done using theirs areas
            vector<Rect> tmp_tmp_new_ROIs;
            new_ROIs = old_ROIs;
            for (int k = 0; k < new_ROIs.size(); k++)
            {
                bool add_flag = true;
                Rect r1;
                Rect r2;
                Rect r3;
                int r4;
                int r5;
                for (int l = k + 1; l < new_ROIs.size(); l++)
                {
                    r1 = new_ROIs[k];
                    r2 = new_ROIs[l];
                    // intersection of r1 and r2
                    r3 = r2 & r1;
                    // r4 = not overlapped area
                    r4 = r1.area() + r2.area() - 2 * r3.area();
                    // smallest rectangle including r1 and r2
                    r5 = (r1 | r2).area();

                    //  if the intersection is greater than zero -> are overlapped
                    if (r3.area() > 0)
                    {
                        // if the rectangle including r1 and r2 is == r1
                        if (r5 == r1.area())
                        {
                            // r2 is inside r1
                            add_flag = false;
                            break;
                        }
                        else if (r5 == r2.area())
                        {
                            // r1 is inside r2
                            add_flag = true;
                        }
                        else if (r4 / (float)r3.area() > 0.5)
                        {
                            // Overlapping Rectangles
                            // keep the smallest
                            if (r1.area() > r2.area())
                            {

                                add_flag = false;
                                break;
                            }
                            else
                            {
                                add_flag = true;
                            }
                        }
                        else
                        {
                            // keep the smallest area
                            if ((float)r1.area() < (float)r2.area())
                            {
                                add_flag = false;
                                break;
                            }
                            else
                            {
                                add_flag = true;
                            }
                        }
                    }
                    else
                    {

                        // Non-overlapping Rectangles , maybe consider to add it, but need to run all the for cycle.
                        add_flag = true;
                    }
                }
                // after the previous consideration, if add is true, save the current rectangle r1
                if (add_flag == true)
                {
                    tmp_tmp_new_ROIs.push_back(new_ROIs[k]);
                }
            }
            new_ROIs = tmp_tmp_new_ROIs;
            tmp_tmp_new_ROIs.clear();

            // erase the tracker, so initialize it
            trackers.clear();
            old_ROIs = new_ROIs;

            // for loop for comparison of new pedestrian and old trackers pedestrian, and add the new one
            if (old_ROIs.size() != 0 & pedestrian.size() != 0)
            {
                bool add_flag = false;
                for (int k = 0; k < pedestrian.size(); k++)
                {
                    for (int j = 0; j < old_ROIs.size(); j++)
                    {
                        Point old_ROI_centre = (old_ROIs[j].br() + old_ROIs[j].tl()) / 2.0;
                        Point pedestrian_centre = (pedestrian[k].br() + pedestrian[k].tl()) / 2;
                        float distance = sqrt((old_ROI_centre.x - pedestrian_centre.x) ^ 2 + (old_ROI_centre.y - pedestrian_centre.y) ^ 2);
                        if (distance < threshold_distance)
                        {
                            add_flag = false;
                            // break;
                        }
                        else
                        {
                            add_flag = true;
                        }
                        // calculation of the intersaction area
                        Rect intersection = pedestrian[k] & old_ROIs[j];
                        float intersection_ratio = (float)intersection.area() / (float)pedestrian[k].area();
                        if (intersection.area() / (float)old_ROIs[j].area() > 0.70)
                        {
                            add_flag = false;
                            break;
                        }
                        else if (intersection.area() > 10 && intersection.area() / (float)old_ROIs[j].area() < 0.70)
                        {
                            // exist an intersection so control all the cases:
                            if (intersection.area() == pedestrian[k].area())
                            {
                                // prev ROI is inside pedestrain[k] -> tracker better than the detector
                                add_flag = false;
                                break;
                                // it is already condidered, break if.
                            }
                            else if (intersection.area() == old_ROIs[j].area())
                            {
                                // pedestrian[k] is inside previous ROI
                                // subs the old ROI with the new one -> detection better than the tracker
                                new_ROIs[j] = pedestrian[k];
                                add_flag = false;
                                break;
                                // it is already condidered, break if.
                            }
                            else if (old_ROIs[j].area() == pedestrian[k].area())
                            {
                                add_flag = false;
                                break;
                            }
                            else
                            {
                                // Overlapping Rectangles -> consider area of intersection_ratio // threshold_intersection
                                if (intersection_ratio > threshold_intersection_ratio)
                                {
                                    // overlap is greater, means that rectangle is already consider by the tracker
                                    // new_ROIs[j] = pedestrian[k];
                                    if (old_ROIs[j].area() < pedestrian[k].area())
                                    {
                                        add_flag = false;
                                        break;
                                    }
                                    else
                                    {
                                        add_flag = true;
                                    }
                                    // it is already condidered, break if.
                                }
                                else
                                {
                                    // olverlap is lower, means that rectangle is not already consider -> new person!
                                    add_flag = true;
                                }
                            }
                        }
                        // there is no intersection so add the new rectangle
                        else
                        {
                            add_flag = true;
                        }
                    }
                    if (add_flag == true)
                    {
                        // cout << "add" << k << endl;
                        new_ROIs.push_back(pedestrian[k]);
                        // algorithms.push_back(TrackerCSRT::create());
                    }
                }
            }
            else
            {
                new_ROIs = pedestrian;
            }

            algorithms.clear();
            objects.clear();

            ROIs = new_ROIs;
            MultiTracker newtrackers;

            for (int p = 0; p < ROIs.size(); p++)
            {
                algorithms.push_back(TrackerCSRT::create());
                objects.push_back(ROIs[p]);
            }

            newtrackers.add(algorithms, frame, objects);
            trackers = newtrackers;
        }

        /******************** Metric & Print in log file *********************/

        // read the metric true file
        vector<Rect> exact_points;
        ss("./Video/gt/gt.txt", &exact_points, i);

        // Build Cost Matrix, the elements are the square distances
        // of the center of ROIs & the Exact_Point from the ground truth
        vector<vector<double>> costMatrix;
        if (ROIs.size() > 0)
        {
            for (int j = 0; j < ROIs.size(); j++)
            {
                vector<double> tmp_row;
                Point ROIs_j = (ROIs[j].tl() + ROIs[j].br()) / 2.0;
                for (int k = 0; k < exact_points.size(); k++)
                {
                    Point exact_k = (exact_points[k].tl() + exact_points[k].br()) / 2.0;
                    double tmp_distance = sqrt(pow(ROIs_j.x - exact_k.x, 2) + pow(ROIs_j.y - exact_k.y, 2));
                    if (tmp_distance > 250)
                    {
                        tmp_distance = 250;
                    }
                    tmp_row.push_back(tmp_distance);
                }
                costMatrix.push_back(tmp_row);
            }
        }

        // use Hungarian Algorithm to find best combination to minimize the costs
        // return a vector of assignments element by elements
        // if the element is "-1" means that there is no association
        HungarianAlgorithm HungAlgo;
        vector<int> assignment;
        double cost = HungAlgo.Solve(costMatrix, assignment);

        int FP_frame = 0;
        int count_sum = 0;
        int error_sum = 0;
        // calculation of missed tracked
        int Missed_frame = exact_points.size() - ROIs.size();
        if (Missed_frame > 0)
        {
            Missed.push_back(Missed_frame);
        }
        else
        {
            Missed.push_back(0);
        }
        // calculation of average error in position of the tracked object
        for (int x = 0; x < costMatrix.size(); x++)
        {
            int tmp_x_error;
            int tmp_y_error;
            if (assignment[x] == -1)
            {
                FP_frame++;
            }
            else
            {
                if (costMatrix[x][assignment[x]] < 250)
                {
                    error_sum = error_sum + costMatrix[x][assignment[x]];
                    count_sum++;
                }
            }
        }
        float error_frame = (float)error_sum / (float)count_sum;
        error_average.push_back(error_frame);
        FP.push_back(FP_frame);

        // update trajectory
        vector<vector<Point>> new_trajectories;
        vector<Point> new_center_list;
        HungarianAlgorithm HungAlgo_traj;
        vector<int> assignment_traj;

        vector<vector<double>> costMatrix_traj;
        if (i != 0 && !(trajectories.empty()) && !(ROIs.empty()))
        {
            for (int j = 0; j < ROIs.size(); j++)
            {
                new_center_list.push_back(center_rect(ROIs[j]));
            }

            for (int k = 0; k < trajectories.size(); k++)
            {
                vector<double> tmp_row;
                for (int j = 0; j < new_center_list.size(); j++)
                {
                    double tmp_distance = sqrt(pow(new_center_list[j].x - trajectories[k].back().x, 2) + pow(new_center_list[j].y - trajectories[k].back().y, 2));
                    if (tmp_distance > 20)
                    {
                        tmp_distance = 200;
                    }
                    tmp_row.push_back(tmp_distance);
                }
                costMatrix_traj.push_back(tmp_row);
            }
            cout << "frame: " << i ;
            std::cout << " matrix size " << costMatrix_traj.size() << " - " << costMatrix_traj[1].size() << endl;
            // for (unsigned int x = 0; x < costMatrix_traj.size(); x++)
            //     std::cout << x << "," << assignment_traj[x] << "\t";

            // std::cout << "\ncost: " << cost << std::endl;

            // solve matches with Hungarian
            double cost_traj = HungAlgo_traj.Solve(costMatrix_traj, assignment_traj);

            for (int j = 0; j < assignment_traj.size(); j++)
            {
                int tmp_index = assignment_traj[j];
                if (tmp_index != -1 && costMatrix_traj[j][tmp_index] < 18 )
                {
                    vector<Point> tmp_vec_traj;
                    tmp_vec_traj = trajectories[j];
                    tmp_vec_traj.push_back(new_center_list[tmp_index]);
                    new_trajectories.push_back(tmp_vec_traj);
                }
                // else if ( tmp_index != -1 && costMatrix_traj[j][tmp_index] > 20 )
                // {
                //     vector<Point> tmp_vec_traj;
                //     tmp_vec_traj.push_back(new_center_list[0]);
                //     new_trajectories.push_back(tmp_vec_traj);
                // }
            }
            for ( int j = 0; j < new_center_list.size(); j++ ){
                if( find( assignment_traj.begin(), assignment_traj.end(), j ) != assignment_traj.end()){
                    vector<Point> tmp_new_vec_traj;
                    tmp_new_vec_traj.push_back(new_center_list[j]);
                    new_trajectories.push_back(tmp_new_vec_traj);
                }
            }
        }
        else
        {
            for (int j = 0; j < ROIs.size(); j++)
            {
                new_center_list.push_back(center_rect(ROIs[j]));
            }
            for (int j = 0; j < new_center_list.size(); j++)
            {
                vector<Point> tmp_vec_traj;
                tmp_vec_traj.push_back(new_center_list[j]);
                new_trajectories.push_back(tmp_vec_traj);
            }
        }
        trajectories.clear();

        trajectories = new_trajectories;

        new_trajectories.clear();

        // plot each trajectories
        for (int k = 0; k < trajectories.size(); k++)
        {
            if (trajectories[k].size() > 1)
            {
                for (int j = 1; j < trajectories[k].size(); j++)
                {
                    Point pt_1 = trajectories[k][j - 1];
                    Point pt_2 = trajectories[k][j];
                    // cout << "here" <<endl;

                    line(frame, pt_1, pt_2, Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }
        // imshow("track", frame_test);

        // stamp tracking on output tracking.txt file
        // <frame>, <id>, <x_center>, <y_center>
        for (int k = 0; k < trackers.getObjects().size(); k++)
        {
            int id = trackers.getDefaultName()[k];
            Point tmp_center = (trackers.getObjects()[k].br() + trackers.getObjects()[k].tl()) / 2;
            int x_center = tmp_center.x;
            int y_center = tmp_center.y;

            fprintf(trackingFile, "%d,%03d,%03d,%03d\n", i + 1, id, x_center, y_center);
        }

        /*************************** GRAPHICAL STUFF *******************************/
        // draw the tracked object
        for (int j = 0; j < trackers.getObjects().size(); j++)
        {
            rectangle(frame, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            rectangle(motion_mask_ed, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            putText(frame, to_string(trackers.getDefaultName()[j]),
                    Point(trackers.getObjects()[j].x, trackers.getObjects()[j].y),
                    FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
        }
        // plot detection
        for (int j = 0; j < pedestrian.size(); j++)
        {
            rectangle(frame_det, pedestrian[j], Scalar(0, 255, 0), 2, 1);
        }

        // show detection results
        cv::imshow("detection", frame_det);

        // export image det
        char nomefile_d[30];
        std::sprintf(nomefile_d, "./Output/PedDet/%06d.jpg", i);
        cv::imwrite(nomefile_d, frame_det);

        // show image with the tracked object
        cv::imshow("tracker + detector", frame);

        // export image tracker + det
        char nomefile_w[30];
        std::sprintf(nomefile_w, "./Output/Videoout/%06d.jpg", i);
        cv::imwrite(nomefile_w, frame);

        // show BG
        // imshow("background", bg);

        // show motion mask
        // imshow("MOG", motion_mask);

        // show motion mask after erosion & dilation
        // cv::imshow("MOG ED", motion_mask_ed);

        // print information on the terminal
        // cout << "frame: " << i << " , displacement error: " << error_frame << endl;
        fprintf(pFile, "frame: %03d  - displacement error: %f\n", i, error_frame);

        //quit on ESC button
        if (waitKey(5) == 27)
        {
            cout << "\n***************************\n";
            cout << "\nbreaked cycle\n"
                 << endl;

            fprintf(pFile, "\nbreaked cycle\n");
            break;
        }
    }

    cout << "\n***************************\n"
         << "\n_Final evaluation_" << endl;

    fprintf(pFile, "Final Evaluation:\n");

    // evaluation of the algorithm
    // calculation of average error
    float error_sum = 0;
    for (unsigned int i = 1; i < error_average.size(); i++)
    {
        error_sum = error_sum + error_average[i];
    }
    float final_error_average = error_sum / ((float)error_average.size() - 1);
    cout << "Average error displacement: " << final_error_average << "\n"
         << endl;
    fprintf(pFile, "Average error displacement: %f\n", final_error_average);

    tend = time(0);
    cout << "Time of execution for " << error_average.size() << " frames: " << difftime(tend, tstart) << " second(s)." << endl;

    fprintf(pFile, "Time of execution for %d frames: %d second(s).", (int)error_average.size(), (int)difftime(tend, tstart));

    cout << "End of program" << endl;
}

// Functions

// calculate center of the rectangle
Point center_rect(Rect Rettangolo)
{
    Point center;
    center = (Rettangolo.br() + Rettangolo.tl()) / 2;
    return center;
}

// read the ground truth
void ss(const char *ground_truth, vector<Rect> *exact_points, int i)
{
    int line_len;
    size_t n;
    char *line = NULL;
    char *token = NULL;
    bool flag = true;
    vector<vector<int>> truth_people;
    int index = 0;
    int count = 0;
    FILE *truth = fopen(ground_truth, "r"); // "../Video/gt/gt.csv"
    assert(truth != NULL);
    while (flag)
    {
        vector<int> truth_person;
        count = 0;
        line_len = getline(&line, &n, truth);
        for (int k = 0; k <= 5; k++)
        {
            token = strsep(&line, ",");
            if (count == 0)
            {
                index = atoi(token);
            }
            if (index == i) // if the index is related to the current frame
            {
                truth_person.push_back(atoi(token));
            }
            else if (atoi(token) > i & count == 0) // if the index is greater than the current frame
            {
                flag = false;
            }
            count++;
        }
        if (flag & index == i)
        {
            truth_people.push_back(truth_person); // save the truth person
        }
        while ((token = strsep(&line, ",")) != NULL)
        {
        } // finish the line to reset the token
    }
    fclose(truth);
    for (int k = 0; k < truth_people.size(); k++)
    {
        int tl = truth_people[k][2];
        int tp = truth_people[k][3];
        int w = truth_people[k][4];
        int l = truth_people[k][5];
        Rect ROI(tl, tp, w, l);
        //Point center = ( ROI.br() + ROI.tl() )/2.0 ;
        exact_points->push_back(ROI);
        // printf("person:%i -> ID:%2i lf:%i tp:%i w:%i l:%i\n",i,truth_people[k][1],truth_people[k][2],truth_people[k][3],truth_people[k][4],truth_people[k][5]);
    }
}