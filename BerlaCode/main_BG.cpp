#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library
#include <opencv2/bgsegm.hpp>  // library for the bg segmentation
#include "opencv2/imgproc.hpp"
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

int N = 795;

int main(){

    Mat frame_BGR, frame_split[3];
    Mat frame_buff_B, frame_buff_G, frame_buff_R;
    Mat mean_B, mean_G, mean_R, mean;

    for (int i=1; i <= N; i++){

      char path [23];
      sprintf(path, "Video/img1/%06i.jpg",i);
      frame_BGR = imread(path,1); //load the image (0 = gray scale)
      //fprintf(stdout,"%i\n",i);

      if(i==1){
        split(frame_BGR, frame_split);
        frame_buff_B = frame_split[0].clone();
        frame_buff_G = frame_split[1].clone();
        frame_buff_R = frame_split[2].clone();
        frame_buff_B.convertTo(frame_buff_B,CV_32F, 1.0/255.0);
        frame_buff_G.convertTo(frame_buff_G,CV_32F, 1.0/255.0);
        frame_buff_R.convertTo(frame_buff_R,CV_32F, 1.0/255.0);
      } else{
        split(frame_BGR, frame_split);
        frame_split[0].convertTo(frame_split[0],CV_32F, 1.0/255.0);
        frame_split[1].convertTo(frame_split[1],CV_32F, 1.0/255.0);
        frame_split[2].convertTo(frame_split[2],CV_32F, 1.0/255.0);
        frame_buff_B = frame_split[0] + frame_buff_B;
        frame_buff_G = frame_split[1] + frame_buff_G;
        frame_buff_R = frame_split[2] + frame_buff_R;
      }
    }

    mean_B = frame_buff_B/float(N);
    mean_G = frame_buff_G/float(N);
    mean_R = frame_buff_R/float(N);
    //mean_B.convertTo(mean_B,CV_8UC1);


    vector<Mat> mean_tot = {mean_B, mean_G, mean_R};

    merge(mean_tot, mean);
    Mat bg_out;
    mean.convertTo(bg_out, CV_8UC3, 255.0);
    imwrite( "background.jpg", bg_out);

    imshow("mean",mean);
    if(waitKey(0)==27){
      return 0;
    }
}


