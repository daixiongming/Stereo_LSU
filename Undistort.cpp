// Undistort.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"



#include<opencv2/opencv.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


//#include "opencv.hpp"

#include<vector>
#include<string>
#include<algorithm>
#include<iostream>
#include<iterator>
#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>


using namespace std;
using namespace cv;

/*
void ReadIntrinsics(Mat &cameraMatrix, Mat &distCoeffs, Size &imageSize, char *IntrinsicsPath)
{
	bool FSflag = false;
	FileStorage readfs;

	FSflag = readfs.open(IntrinsicsPath, FileStorage::READ);
	if (FSflag == false) cout << "Cannot open the file" << endl;
	readfs["Camera_Matrix"] >> cameraMatrix;
	readfs["Distortion_Coefficients"] >> distCoeffs;
	readfs["image_Width"] >> imageSize.width;
	readfs["image_Height"] >> imageSize.height;

	cout << cameraMatrix << endl << distCoeffs << endl << imageSize << endl;

	readfs.release();
}

void Undistort_img(Mat map1, Mat map2, char *path)
{
	Mat img1, img2;
	img1 = imread(path);
	if (img1.empty()) cout << "Cannot open the image" << endl;
	remap(img1, img2, map1, map2, INTER_LINEAR);
	// 	imwrite(path, img2);
	imshow("src img", img1);
	imshow("dst img", img2);
	waitKey();
}

*/

int amain(int argc, char* argv[])
{
	Mat	cameraMatrix, distCoeffs;
	//Size imageSize;
	// char * IntrinsicsPath = "Intrinsics.xml";

	 cameraMatrix = Mat::eye(3, 3, CV_64F);
	//内参矩阵, ...
	cameraMatrix.at<double>(0, 0) = 3.00576392e+003;
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(0, 2) = 6.45924683e+002;
	cameraMatrix.at<double>(1, 1) = 1.51140420e+004;
	cameraMatrix.at<double>(1, 2) = 5.01527222e+002;
	//畸变参数，
	 distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = -2.31156492;
	distCoeffs.at<double>(1, 0) = 5.42513733e+001;
	distCoeffs.at<double>(2, 0) = 5.73278442e-002;
//	distCoeffs.at<double>(3, 0) = -9.8158e-001;
//	distCoeffs.at<double>(2, 0) = 0;
		distCoeffs.at<double>(3, 0) = 0;
	distCoeffs.at<double>(4, 0) = 0;
	Mat view, rview, map1, map2;
	Size imageSize;


	Mat image;

	string imgpath = argv[1];

	image = imread(imgpath, 1);
	imshow("img", image);


	

	/* VideoCapture capture(argv[1]);
	if (!capture.isOpened()) {
		return 0;
	}

	while (1) {

		if (!capture.read(frame)) {
			break;
		}





	}

	*/




	imageSize = image.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	remap(image, image, map1, map2, INTER_LINEAR);

	imshow("after undistorted", image);






	/*

	ReadIntrinsics(cameraMatrix, distCoeffs, imageSize, IntrinsicsPath);

	// 去畸变并保留最大图
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);

	Undistort_img(map1, map2, "E:/VS13/undistort/undistort/1.bmp");

	// 去畸变至全图
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(),
		imageSize, CV_16SC2, map1, map2);

	Undistort_img(map1, map2, "E:/VS13/undistort/undistort/1.bmp");

	*/

	waitKey(0);

    return 0;
}

