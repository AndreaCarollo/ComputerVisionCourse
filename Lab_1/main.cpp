#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;


int main(){
    Mat image; // set matrix variable for the image
    image = imread("Google.jpg",1); //load the image (0 = gray scale)

    imshow("Image window",image);
    waitKey(0); // wait until the user press a key


    return 0;
}