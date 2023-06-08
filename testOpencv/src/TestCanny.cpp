#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    int edgeThresh = 1;
    int lowThreshold;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;

    Mat A = imread(argv[1], IMREAD_COLOR);
    Mat B, A_gray, detected_edges;
    Mat dst;
    imshow("Origin", A);
    waitKey(0);

    if ( !A.data ) { return -1; }

    cvtColor( A, A_gray, COLOR_BGR2GRAY);

    blur( A_gray, detected_edges, Size(3, 3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

    dst = Scalar::all(0);
    A.copyTo( dst, detected_edges );
    imshow("Canny", dst);
    waitKey(0);
    return 0;
}