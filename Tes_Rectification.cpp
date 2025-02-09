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

int swsmain(int argc, char* argv[])
{
	//Mat	cameraMatrix, distCoeffs;

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);



	//Size imageSize;
	// char * IntrinsicsPath = "Intrinsics.xml";

//	 cameraMatrix = Mat::eye(3, 3, CV_64F);
	//内参矩阵, ...
/*	cameraMatrix[0].at<double>(0, 0) = 5.3471330014574585e+002;
	cameraMatrix[0].at<double>(0, 1) = 0;
	cameraMatrix[0].at<double>(0, 2) = 3.3513862534129083e+002;
	cameraMatrix[0].at<double>(1, 1) = 5.3471330014574585e+002;
	cameraMatrix[0].at<double>(1, 2) = 2.4020611211620977e+002;

	cameraMatrix[1].at<double>(0, 0) = 5.3471330014574585e+002;
	cameraMatrix[1].at<double>(0, 1) = 0;
	cameraMatrix[1].at<double>(0, 2) = 3.3401539789486708e+002;
	cameraMatrix[1].at<double>(1, 1) = 5.3471330014574585e+002;
	cameraMatrix[1].at<double>(1, 2) = 2.4159046721725986e+002;
	*/
	cameraMatrix[0].at<double>(0, 0) = 2.3698400732915161e+003;
	cameraMatrix[0].at<double>(0, 1) = 0;
	cameraMatrix[0].at<double>(0, 2) = 6.4816679955865754e+002;
	cameraMatrix[0].at<double>(1, 1) = 2.3698400732915161e+003;
	cameraMatrix[0].at<double>(1, 2) = 5.1036769274228334e+002;

	cameraMatrix[1].at<double>(0, 0) = 2.3698400732915161e+003;
	cameraMatrix[1].at<double>(0, 1) = 0;
	cameraMatrix[1].at<double>(0, 2) = 6.5167189937191222e+002;
	cameraMatrix[1].at<double>(1, 1) = 2.3698400732915161e+003;
	cameraMatrix[1].at<double>(1, 2) = 5.0875974034019333e+002;




	//畸变参数，
//	 distCoeffs = Mat::zeros(5, 1, CV_64F);

	distCoeffs[0] = distCoeffs[1] = Mat::zeros(1, 8, CV_64F);

/*	distCoeffs[0].at<double>(0, 0) = -2.7456948645081791e-001;
	distCoeffs[0].at<double>(0, 1) = -1.8313659520295664e-002;

	distCoeffs[0].at<double>(0, 7) = -2.4476896009779306e-001;

	distCoeffs[1].at<double>(0, 0) = -2.8073637369061927e-001;
	distCoeffs[1].at<double>(0, 1) = 9.3010333969653317e-002;

	distCoeffs[1].at<double>(0, 7) = 1.6329629645783817e-002; */

	distCoeffs[0].at<double>(0, 0) = -1.5971630575367365e-001;
	distCoeffs[0].at<double>(0, 1) = 5.6881683989374598e-001;

	distCoeffs[0].at<double>(0, 7) = -1.2951368551469130e+001;
//	distCoeffs[0].at<double>(0, 7) = 0;

	distCoeffs[1].at<double>(0, 0) = -1.6542987310342402e-001;
	distCoeffs[1].at<double>(0, 1) = 3.8436269899260509e+000;

	distCoeffs[1].at<double>(0, 7) = 3.3271122648490248e+001;
//	distCoeffs[1].at<double>(0, 7) = 0;

	Mat R =  Mat::zeros(3, 3, CV_64F);

/*	R.at<double>(0, 0) = 9.9975845099238192e-001;
	R.at<double>(0, 1) = 5.2938875802109996e-003;
	R.at<double>(0, 2) = -2.1331067099445444e-002;

	R.at<double>(1, 0) = -4.9129099556325592e-003;
	R.at<double>(1, 1) = 9.9982819095221864e-001;
	R.at<double>(1, 2) = 1.7873217197295795e-002;

	R.at<double>(2, 0) = 2.1422021031658117e-002;
	R.at<double>(2, 1) = -1.7764102327501703e-002;
	R.at<double>(2, 2) = 9.9961269183790236e-001;  */

	R.at<double>(0, 0) = 7.7298187879727964e-001;
	R.at<double>(0, 1) = -3.2866237330378151e-004;
	R.at<double>(0, 2) = -6.3442801564249351e-001;

	R.at<double>(1, 0) = 3.0209147349222970e-003;
	R.at<double>(1, 1) = 9.9999043591675640e-001;
	R.at<double>(1, 2) = 3.1626174570887726e-003;

	R.at<double>(2, 0) = 6.3442090848678045e-001;
	R.at<double>(2, 1) = -4.3611989245994761e-003;
	R.at<double>(2, 2) = 7.7297547879525130e-001;


	Mat T = Mat::zeros(3, 1, CV_64F);

/*	T.at<double>(0, 0) = -3.3385294685588063e+000;
	T.at<double>(1, 0) = 4.8752222150627342e-002;
	T.at<double>(2, 0) = -1.0621612774501774e-001; */

	T.at<double>(0, 0) = 5.4733719439647672e+001;
	T.at<double>(1, 0) = -3.7156829904808231e-001;
	T.at<double>(2, 0) = 1.6669033657083485e+001;


	Mat image;

	string imgpath = argv[1];

	image = imread(imgpath, 1);
	imshow("image left", image);

	Size imageSize = image.size();

	Rect validRoi[2];
	Mat R1, R2, P1, P2, Q;
	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	Mat rmap[2][2];
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	//图像是否倾斜到竖直或者水平：
	//bool isVerticalStereo = 1; //for temporary test;
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	Mat map[2][2];
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], Mat(), getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map[0][0], map[0][1]);

	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], Mat(), getOptimalNewCameraMatrix(cameraMatrix[1], distCoeffs[1], imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map[1][0], map[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}
	Mat image2[2];
	image2[0] = image;

	string imgpath2 = argv[2];

	image2[1] = imread(imgpath2, 1);
	imshow("image right", image2[1]);

	Mat rimg[2], cimg[2],rectifyimg[2];
	for (int k = 0; k < 2; k++)
	{
		Mat img = image2[k];
		remap(img, rimg[k], rmap[k][0], rmap[k][1], CV_INTER_LINEAR);

		remap(img, rectifyimg[k], map[k][0], map[k][1], INTER_LINEAR);


		imshow("rimg_k", rimg[k]);
		imshow("rectify image", rectifyimg[k]);
		if (k == 0)
			imwrite("output2rectified_right.jpeg", rectifyimg[0]);
		else
			imwrite("output2rectified_left.jpeg", rectifyimg[1]);
		cout <<"矩形参数："<< validRoi[0].x<<","<< validRoi[0].y<<","<< validRoi[0].width<<","<< validRoi[0].height << endl;
		Rect chang(validRoi[0].x, validRoi[0].y, validRoi[0].width, validRoi[0].height);
		rectangle(rectifyimg[0], chang, Scalar(0, 0, 225), 3, 8);
		imshow("recfified with rectangle", rectifyimg[0]);
	//	cvtColor(rimg[k], cimg[k], COLOR_GRAY2BGR);
		Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
		resize(rimg[k], canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
		bool useCalibrated = 1;
		if (useCalibrated)
		{
			sf = 0.02;
			Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
				cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
			rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
			imshow("rectangle", canvasPart);
		}

	}


	if (!isVerticalStereo)
		for (int j = 0; j < canvas.rows; j += 16)
			line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
	else
		for (int j = 0; j < canvas.cols; j += 16)
			line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);










//	distCoeffs.at<double>(0, 0) = -2.31156492;
//	distCoeffs.at<double>(1, 0) = 5.42513733e+001;
//	distCoeffs.at<double>(2, 0) = 5.73278442e-002;
//	distCoeffs.at<double>(3, 0) = -9.8158e-001;
//	distCoeffs.at<double>(2, 0) = 0;
//	distCoeffs.at<double>(3, 0) = 0;
//	distCoeffs.at<double>(4, 0) = 0;
//	Mat view, rview, map1, map2;
//	Size imageSize;


	


	

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


	/*

	imageSize = image.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	remap(image, image, map1, map2, INTER_LINEAR);

	imshow("after undistorted", image);


	*/



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

