#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <assert.h>

using namespace cv;
using namespace std;


void exact_points(const char * path_truth_data, vector<Rect> *exact_points, int i){
    int line_len;
    size_t n;
    char * line = NULL;
    char * token = NULL;
    bool flag = true;
    vector<vector<int>> truth_people;
    int index = 0;
    int count = 0;
    FILE* truth = fopen(path_truth_data, "r"); // "../Video/gt/gt.csv"
    assert(truth!=NULL);
    while (flag)
    {
      vector<int> truth_person;
      count = 0;
      line_len = getline(&line, &n, truth);
      for(int k = 0; k <=5; k++)
      {
        token = strsep(&line, ",");
        if(count==0){
          index = atoi(token);
        }
        if(index == i) // if the index is related to the current frame
        {
          truth_person.push_back(atoi(token));
        } else if(atoi(token) > i & count == 0) // if the index is greater than the current frame
        {
          flag = false;
        }
        count++;
      }
      if(flag & index == i){
        truth_people.push_back(truth_person); // save the truth person
      }
      while((token = strsep(&line, ","))!=NULL){} // finish the line to reset the token
    }
    fclose(truth);
    for(int k=0; k<truth_people.size(); k++){
      int tl = truth_people[k][2];
      int tp = truth_people[k][3];
      int w  = truth_people[k][4];
      int l  = truth_people[k][5];
      Rect ROI(tl, tp, w, l);
      exact_points->push_back(ROI);
      // printf("person:%i -> ID:%2i lf:%i tp:%i w:%i l:%i\n",i,truth_people[k][1],truth_people[k][2],truth_people[k][3],truth_people[k][4],truth_people[k][5]);
    }
}