#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * @brief 学习一些Opencv的基本操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */

int A_simple_example(int argc, char** argv )
{
    std::string image_path = samples::findFile(argv[1]);
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
}

int displayImage(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}

int testMat(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

	Mat A = image;

	Mat D (A, Rect(10, 10, 100, 100) ); // using a rectangle
	Mat E = A(Range::all(), Range(1,3)); // using row and column boundaries（展示全部行和部分列）
	Mat F = A.clone();
	Mat G;
	A.copyTo(G);
	Mat M(2,2, CV_8UC3, Scalar(0,0,255)); // 8uc3中的3代表三通道，scalar是一个像素点三个通道的值
    std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;
    
    int sz[3] = {2,2,2};
    Mat L(3,sz, CV_8UC(1), Scalar::all(0));
    std::cout << "L = " << std::endl << " " << L.size << std::endl << std::endl;

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", D);
    waitKey(0);
    return 0;
}

int testTime()
{
    double t = (double)getTickCount();
    for (int i = 0; i < 9999999; i ++)
    {
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "Times passed in seconds: " << t << std::endl;

    return 0;
}

/**
 * @brief 效率最高的查找表赋值方法，但使用OpenCV库内置的函数速度更快
 * 
 * @param I 
 * @param table 
 * @return Mat& 
 */
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));     

    int channels = I.channels();

    int nRows = I.rows * channels; 
    int nCols = I.cols;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
        std::cout << "Mat in memory is continuous." << std::endl;         
    }

    int i,j;
    uchar* p; 
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];             
        }
    }
    return I; 
}

int testScanImage(int argc, char** argv )
{
    int divideWith; // convert our input string to number - C++ style
    std::stringstream s;
    s << argv[2];
    s >> divideWith;
    if (!s)
    {
        std::cout << "Invalid number entered for dividing. " << std::endl; 
        return -1;
    }
    
    uchar table[256]; 
    for (int i = 0; i < 256; ++i)
       table[i] = divideWith* (i/divideWith);
    
    Mat img = imread(argv[1], IMREAD_COLOR);
    Mat output;

    imshow("origin", img);

    // ScanImageAndReduceC(img, table); 
    
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.data; // .data会返回矩阵第一行第一列的指针
    for ( int i = 0; i < 256; ++i)
    {
        p[i] = table[i];
    }
    LUT(img, lookUpTable, output);
    imshow("testScan" ,output);
    waitKey(0);

    return 0;
}

void Sharpen(const Mat& myImage,Mat& Result)
{
    // 若括号内表达式为False，则抛出一个错误
    CV_Assert(myImage.depth() == CV_8U);  // 仅接受uchar图像

    Result.create(myImage.size(),myImage.type());
    const int nChannels = myImage.channels();

    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);

        uchar* output = Result.ptr<uchar>(j);

        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }

    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}

void sharpen_With_filter2D(char** argv)
{
    double time_ = (double)getTickCount();
    Mat myImage = imread(argv[1], IMREAD_COLOR);
    Mat output;
    Mat kern = (Mat_<char>(3, 3) << 0, -1, 0, 
                                    -1, 5, -1, 
                                    0, -1, 0);
    filter2D(myImage, output, myImage.depth(), kern);
    time_ = ((double)getTickCount() - time_) / getTickFrequency();
    std::cout << "Times passed in seconds: " << time_ << std::endl;
    imshow("sharpen_with_filter2D", output);
    waitKey(0);
}

int add_Two_imgs(char** argv)
{
    double alpha = 0.5; double beta; double input;

    Mat src1, src2, dst;

    /// Ask the user enter alpha
    std::cout<<" Simple Linear Blender "<<std::endl;
    std::cout<<"-----------------------"<<std::endl;
    std::cout<<"* Enter alpha [0-1]: ";
    std::cin>>input;

    /// We use the alpha provided by the user iff it is between 0 and 1
    if( alpha >= 0 && alpha <= 1 )
    { alpha = input; }

    /// Read image ( same size, same type )
    src1 = imread(argv[1]);
    src2 = imread(argv[2]);

    if( !src1.data ) { printf("Error loading src1 \n"); return -1; }
    if( !src2.data ) { printf("Error loading src2 \n"); return -1; }

    /// Create Windows
    namedWindow("Linear Blend", 1);

    beta = ( 1.0 - alpha );
    addWeighted( src1, alpha, src2, beta, 0.0, dst);

    imshow( "Linear Blend", dst );

    waitKey(0);
    return 0;
}

int increase_Light(char** argv)
{
    double alpha;
    int beta;
    /// 读入用户提供的图像
    Mat image = imread( argv[1] );
    Mat new_image = Mat::zeros( image.size(), image.type() );

    /// 初始化
    std::cout << " Basic Linear Transforms " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "* Enter the alpha value [1.0-3.0]: ";
    std::cin >> alpha;
    std::cout << "* Enter the beta value [0-100]: ";
    std::cin >> beta;

    /// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
    // for( int y = 0; y < image.rows; y++ )
    // {
    //     for( int x = 0; x < image.cols; x++ )
    //     {
    //         for( int c = 0; c < 3; c++ )
    //         {
    //             new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
    //         }
    //     }
    // }
    image.convertTo(new_image, -1, alpha, beta);

    /// 创建窗口
    namedWindow("Original Image", 1);
    namedWindow("New Image", 1);

    /// 显示图像
    imshow("Original Image", image);
    imshow("New Image", new_image);

    /// 等待用户按键
    waitKey();
    return 0;
}

int main(int argc, char** argv )
{
    
	// A_simple_example(argc, argv);
	// displayImage(argc, argv);
	// testMat(argc, argv);
    // testTime();
    // testScanImage(argc, argv);
    // sharpen_With_filter2D(argv);
    // add_Two_imgs(argv);
    increase_Light(argv);
}