To compile the code need:
- cmake(version 3.1)
- c++ 11 as standard
- libray Hungarain.h
- code Hungarian.cpp
( see CMakeLists.txt )

Put inside this directory, the folder Video containing:
./Video/gt/gt.txt        # ground truth
./Video/img1/%06d.jpeg   # frames, called numerically

Create in this directory, if not exist, the folder Output:
./Output                # here will be created the detection.txt, tracking.txt, log.txt
./Output/Videoout       # here will be saved the frames with bounding boxes of detection & tracking
                          detection in green, treacking in blue

The name of the executable code is ./A1


