# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mmlab/workspace/C++/Lab_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mmlab/workspace/C++/Lab_1

# Include any dependencies generated for this target.
include CMakeFiles/Intro.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Intro.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Intro.dir/flags.make

CMakeFiles/Intro.dir/main.cpp.o: CMakeFiles/Intro.dir/flags.make
CMakeFiles/Intro.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mmlab/workspace/C++/Lab_1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Intro.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Intro.dir/main.cpp.o -c /home/mmlab/workspace/C++/Lab_1/main.cpp

CMakeFiles/Intro.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Intro.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mmlab/workspace/C++/Lab_1/main.cpp > CMakeFiles/Intro.dir/main.cpp.i

CMakeFiles/Intro.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Intro.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mmlab/workspace/C++/Lab_1/main.cpp -o CMakeFiles/Intro.dir/main.cpp.s

CMakeFiles/Intro.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/Intro.dir/main.cpp.o.requires

CMakeFiles/Intro.dir/main.cpp.o.provides: CMakeFiles/Intro.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Intro.dir/build.make CMakeFiles/Intro.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Intro.dir/main.cpp.o.provides

CMakeFiles/Intro.dir/main.cpp.o.provides.build: CMakeFiles/Intro.dir/main.cpp.o


# Object files for target Intro
Intro_OBJECTS = \
"CMakeFiles/Intro.dir/main.cpp.o"

# External object files for target Intro
Intro_EXTERNAL_OBJECTS =

Intro: CMakeFiles/Intro.dir/main.cpp.o
Intro: CMakeFiles/Intro.dir/build.make
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_gapi.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_stitching.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_aruco.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_bgsegm.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_bioinspired.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_ccalib.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_cvv.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_dnn_objdetect.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_dnn_superres.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_dpm.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_face.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_freetype.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_fuzzy.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_hfs.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_img_hash.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_line_descriptor.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_quality.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_reg.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_rgbd.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_saliency.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_stereo.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_structured_light.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_superres.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_surface_matching.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_tracking.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_videostab.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_xfeatures2d.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_xobjdetect.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_xphoto.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_shape.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_highgui.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_datasets.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_plot.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_text.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_dnn.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_ml.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_phase_unwrapping.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_optflow.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_ximgproc.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_video.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_videoio.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_imgcodecs.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_objdetect.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_calib3d.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_features2d.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_flann.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_photo.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_imgproc.so.4.1.2
Intro: /home/mmlab/installation/OpenCV-/lib/libopencv_core.so.4.1.2
Intro: CMakeFiles/Intro.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mmlab/workspace/C++/Lab_1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Intro"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Intro.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Intro.dir/build: Intro

.PHONY : CMakeFiles/Intro.dir/build

CMakeFiles/Intro.dir/requires: CMakeFiles/Intro.dir/main.cpp.o.requires

.PHONY : CMakeFiles/Intro.dir/requires

CMakeFiles/Intro.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Intro.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Intro.dir/clean

CMakeFiles/Intro.dir/depend:
	cd /home/mmlab/workspace/C++/Lab_1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mmlab/workspace/C++/Lab_1 /home/mmlab/workspace/C++/Lab_1 /home/mmlab/workspace/C++/Lab_1 /home/mmlab/workspace/C++/Lab_1 /home/mmlab/workspace/C++/Lab_1/CMakeFiles/Intro.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Intro.dir/depend

