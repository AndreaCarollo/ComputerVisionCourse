*** Assignment 2: Computer Vision ***
*** 
*** Student Andrea Carollo
*** 206343
*** andrea.carollo@studenti.unitn.it
***

To compile the code need:
- cmake(version 3.1)
- c++ 11 as standard
- openCV libraries
- openCV dev ( also the non-free packages )
- code Hungarian.cpp & libray Hungarain.h
( see CMakeLists.txt for more )

Put inside this directory, the folder Video containing:
./Video/gt/gt.txt        # ground truth
./Video/img1/%06d.jpeg   # frames, called numerically

Create in this directory, if not exist, the folder Output:
./Output                # here will be created the detection.txt, tracking.txt, log.txt
./Output/Videoout       # here will be saved the frames with bounding boxes of detection & tracking
                          detection in green, treacking in blue
./Output PedDet		# save frames of the detections
./PedTrack		# save frame of the tracking wuth traces

The name of the executable code is ./A1

log.txt will contain the average error on tracking for each frame.
At the end of the file is written time to compute code and final average error.

While the code is running, if press Esc the code will be blocked, and calculate the average error of the analyzed frames.




