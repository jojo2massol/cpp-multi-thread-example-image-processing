Georges de Massol
[@jojo2massol](https://github.com/jojo2massol)

# multi-threads example
The goal of this is to learn how to do multiprocessing. 
The Idea is to process an image (png) to compute the average of the 9 neighbor pixels, in multiple threads.
The code have to :
- Load the image, and its dimensions
- Process the image in as much treads as there are cores,
- save the image in a file.

You can download the picture i used [here](https://stsci-opo.org/STScI-01GGF8H15VZ09MET9HFBRQX4S3.png)
or [here](https://webbtelescope.org/contents/media/images/2022/052/01GF423GBQSK6ANC89NTFJW8VM)

This code is a program that processes an image (in PNG format) by computing the average of the 9 neighbor pixels of each pixel in the image. The code uses the OpenCV library for image processing and the `std::thread` library for multi-threading. It also uses the `std::chrono` library to measure the time taken by the parallel and sequential versions of the image processing algorithm.

The code first loads the image using the `cv::imread()` function. Then, it creates a matrix to store the result of the image processing. Next, it measures the time taken by the parallel version of the image processing algorithm by calling the `ParallelProcess()` function. This function divides the image into equal parts and creates a separate thread for each part. Each thread processes its part of the image by calling the `processImage()` function. Finally, the code saves the processed image and displays the time taken by the parallel version of the algorithm.

The code then repeats the same steps for the sequential version of the image processing algorithm. It calls the `SequentialProcess()` function which processes the entire image in a single thread. The code then saves the processed image and displays the time taken by the sequential version of the algorithm.

# installation (debian/ubuntu)

To install the required libraries on Ubuntu, you can use the following commands:

```bash
sudo apt-get update
sudo apt-get install libopencv-dev build-essential
```
These commands will install the OpenCV library, which is used for image processing in the code.

To compile the code, you can use the following command:

```bash
g++ -std=c++11 -pthread -o main main.cpp  -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgcodecs -O4
```

Once the code has been compiled, you can run it using the following command:

```bash
./main
```
This will execute the main program, which will process the image in parallel and sequential modes and display the time taken by each mode.

## Troubleshooting
If that doesn't work, try this:
```bash
LD_LIBRARY_PATH=/usr/local/lib ./main
```
This command will set the `LD_LIBRARY_PATH` environment variable to the location of the OpenCV library and then execute the main program, if this variable is not set. You can also set this variable permanently.