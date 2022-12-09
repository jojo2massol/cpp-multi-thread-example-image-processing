/*
Georges de Massol - 2022-12-09

The goal of this is to learn how to do multiprocessing. 
The Idea is to process an image (png) to compute the average of the 9 neighbor pixels, in multiple threads.
The code have to :
- Load the image, and its dimensions
- Process the image in as much treads as there are cores,
- save the image in a file.
*/

#include <iostream>
#include <thread>
#include <vector>

// Include the necessary image processing libraries
#include <opencv2/opencv.hpp>

// lib for benchmarking time spent
#include <chrono>

using namespace std;
using namespace cv;

// Function that processes a part of the image in a separate thread
void processImage(const Mat &image, Mat &result, int startRow, int endRow)
{
    for (int row = startRow; row < endRow; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {
            // Compute the average of the 9 neighbor pixels
            Vec3i sum = Vec3i(0, 0, 0);
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -1; j <= 1; ++j)
                {
                    if (row + i >= 0 && row + i < image.rows && col + j >= 0 && col + j < image.cols)
                    {
                        sum += image.at<Vec3b>(row + i, col + j);
                    }
                }
            }
            Vec3b average = sum / 9;
            result.at<Vec3b>(row, col) = average;
        }
    }
}

void ParallelProcess(const Mat &image, Mat &result){

    // Get the number of cores in the system
    int numCores = thread::hardware_concurrency();

    // Create a vector of threads
    vector<thread> threads;

    // Divide the image into equal parts and create a separate thread for each part
    int rowsPerCore = image.rows / numCores;
    int startRow = 0;
    int endRow = rowsPerCore;
    for (int i = 0; i < numCores; ++i)
    {
        threads.push_back(thread(processImage, ref(image), ref(result), startRow, endRow));
        startRow = endRow;
        endRow += rowsPerCore;
    }

    // Wait for all threads to finish
    for (auto &t : threads)
    {
        t.join();
    }

}

void SequentialProcess(const Mat &image, Mat &result){
    processImage(image, result, 0, image.rows);
}

int main()
{
    // Load the image
    Mat image = imread("Pillars of Creation_full_res.png", IMREAD_COLOR);
    // Check if the image has been correctly loaded
    if (image.empty())
    {
        cerr << "Error: could not load the image" << endl;
        return 1;
    }
    // Create a matrix to store the result of the image processing
    Mat result(image.rows, image.cols, CV_8UC3);

    // Start the timer
    std::cout << "Starting the timer (parallel)" << std::endl;
    auto start = chrono::high_resolution_clock::now();

    // Process the image in parallel
    ParallelProcess(image, result);

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    // Compute the duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken by parallel: " << duration.count() << " milliseconds" << endl;

    // Save the processed image
    imwrite("parallel.png", result);

    // Start the timer
    std::cout << "Starting the timer (sequential)" << std::endl;
    start = chrono::high_resolution_clock::now();

    // Process the image in sequential
    SequentialProcess(image, result);

    // Stop the timer
    stop = chrono::high_resolution_clock::now();
    // Compute the duration
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken by sequential: " << duration.count() << " milliseconds" << endl;

    // Save the processed image
    imwrite("sequential.png", result);

    return 0;
}