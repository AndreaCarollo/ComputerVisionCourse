#include "bg.h"  //openCV library

static int ctr = 1;

void bg_train(Mat frame, Mat* bg){ //train function

    if (ctr==1){ // runned only the first time

        //initial background storage
        frame.copyTo(*bg);

    }
    ctr++;
};

void bg_update(Mat frame, Mat* bg, float alpha){
    *bg = alpha* frame + (1.0-alpha)*(*bg);
};
