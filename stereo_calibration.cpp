
#include "stdafx.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

static int print_help()
{
	cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
		" on the chessboards, and a flag: useCalibrated for \n"
		"   calibrated (0) or\n"
		"   uncalibrated \n"
		"     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
		"         matrix separately) stereo. \n"
		" Calibrate the cameras and display the\n"
		" rectified results along with the computed disparity images.   \n" << endl;
	cout << "Usage:\n ./stereo_calib -w board_width -h board_height [-nr /*dot not view results*/] <image list XML/YML file>\n" << endl;
	return 0;
}

//Block matching using Sum of Absolute Differencse(sad)
static Mat funcSADR2L(Mat leftImage, Mat rightImage, int windowSize, int dispMin, int dispMax, int STEP = 1)
{
	CV_Assert(leftImage.rows == rightImage.rows && leftImage.cols == rightImage.cols);
	CV_Assert(windowSize % 2 == 1);
	CV_Assert(dispMax > dispMin);
	int nrLeft = leftImage.rows - 1;
	int ncLeft = leftImage.cols - 1;
	int win = (windowSize - 1) / 2;
	Mat dispMap = Mat::zeros(nrLeft, ncLeft, CV_8UC1);
	for (int i = 0 + win; i <= nrLeft - win; i = i + STEP)
	{
		for (int j = 0 + win; j <= ncLeft - win - dispMax; j = j + STEP)
		{
			double prevSAD = 65532;
			double temp = 0.0;
			int bestMatchSoFar = dispMin;
			for (int dispRange = dispMin; dispRange <= dispMax; dispRange++)
			{
				double sad = 0.0;
				for (int a = -win; a <= win; a++)
				{
					for (int b = -win; b <= win; b++)
					{
						if (j + b + dispRange <= ncLeft)
						{
							temp = (double)rightImage.at<uchar>(i + a, j + b) - (double)leftImage.at<uchar>(i + a, j + b + dispRange);
							if (temp < 0.0)
							{
								temp = temp*(-1.0);
							}
							sad = sad + temp;
						}
					}
				}
				if (prevSAD > sad)
				{
					prevSAD = sad;
					bestMatchSoFar = dispRange;
				}
			}
			dispMap.at<uchar>(i, j) = bestMatchSoFar;
		}
	}
	normalize(dispMap, dispMap, 0, 255, CV_MINMAX);
	return dispMap;
}

//using sum of squared differences(ssd)
Mat funcSSDR2L(Mat leftImage, Mat rightImage, int windowSize, int dispMin, int dispMax, int STEP = 1)
{
	CV_Assert(leftImage.rows == rightImage.rows && leftImage.cols == rightImage.cols);
	CV_Assert(windowSize % 2 == 1);
	CV_Assert(dispMax > dispMin);
	int nrLeft = leftImage.rows - 1;
	int ncLeft = leftImage.cols - 1;
	int win = (windowSize - 1) / 2;
	Mat dispMap = Mat::zeros(nrLeft, ncLeft, CV_8UC1);
	for (int i = 0 + win; i <= nrLeft - win; i = i + STEP)
	{
		for (int j = 0 + win; j <= ncLeft - win - dispMax; j = j + STEP)
		{
			double prevSDD = 65532;
			double temp = 0.0;
			int bestMatchSoFar = dispMin;
			for (int dispRange = dispMin; dispRange <= dispMax; dispRange++)
			{
				double ssd = 0.0;
				for (int a = -win; a <= win; a++)
				{
					for (int b = -win; b <= win; b++)
					{
						if (j + b + dispRange <= ncLeft)
						{
							temp = (double)rightImage.at<uchar>(i + a, j + b) - (double)leftImage.at<uchar>(i + a, j + b + dispRange);
							temp = temp*temp;
							ssd = ssd + temp;
						}
					}
				}
				if (prevSDD > ssd)
				{
					prevSDD = ssd;
					bestMatchSoFar = dispRange;
				}
			}
			dispMap.at<uchar>(i, j) = bestMatchSoFar;
		}
	}
	normalize(dispMap, dispMap, 0, 255, CV_MINMAX);
	return dispMap;
}


//Using Normalized Cross Correlation(ncc)
Mat funcNCCR2L(Mat leftImage, Mat rightImage, int windowSize, int dispMin, int dispMax, int STEP = 1)
{
	CV_Assert(leftImage.rows == rightImage.rows && leftImage.cols == rightImage.cols);
	CV_Assert(windowSize % 2 == 1);
	CV_Assert(dispMax > dispMin);
	int nrLeft = leftImage.rows - 1;
	int ncLeft = leftImage.cols - 1;
	int win = (windowSize - 1) / 2;
	Mat dispMap = Mat::zeros(nrLeft, ncLeft, CV_8UC1);

	for (int i = 0 + win; i <= nrLeft - win; i = i + STEP)
	{
		for (int j = 0 + win; j <= ncLeft - win - dispMax; j = j + STEP)
		{
			double prevNCC = 0.0;
			int bestMatchSoFar = dispMin;
			for (int dispRange = dispMin; dispRange <= dispMax; dispRange++)
			{
				double ncc = 0.0;
				double nccNumberator = 0.0;
				double nccDenominator = 0.0;
				double nccDenominatorRightWindow = 0.0;
				double nccDenominatorLeftWindow = 0.0;
				for (int a = -win; a <= win; a++)
				{
					for (int b = -win; b <= win; b++)
					{
						nccNumberator += rightImage.at<uchar>(i + a, j + b)*leftImage.at<uchar>(i + a, j + b + dispRange);
						nccDenominatorRightWindow += rightImage.at<uchar>(i + a, j + b)*rightImage.at<uchar>(i + a, j + b);
						nccDenominatorLeftWindow += leftImage.at<uchar>(i + a, j + b + dispRange)*leftImage.at<uchar>(i + a, j + b + dispRange);
					}
				}
				nccDenominator = sqrt(nccDenominatorRightWindow*nccDenominatorLeftWindow);
				ncc = nccNumberator / nccDenominator;
				if (prevNCC < ncc)
				{
					prevNCC = ncc;
					bestMatchSoFar = dispRange;
				}
			}
			dispMap.at<uchar>(i, j) = bestMatchSoFar;
		}
	}
	normalize(dispMap, dispMap, 0, 255, CV_MINMAX);
	return dispMap;
}

static void
StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	bool displayCorners =true;//true;
	const int maxScale = 3;
//	const int maxScale = 2;
//	const float squareSize = 1.f;  // Set this to your actual square size
	const float squareSize = 2.8f;
	// ARRAY AND VECTOR STORAGE:

	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, 0);
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found = findChessboardCorners(timg, boardSize, corners,
					CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				imshow("no resize corner show", cimg);
			//	double sf = 1280. / MAX(img.rows, img.cols);
				double sf = 640. / MAX(img.rows, img.cols); //缩放显示图的效果
				resize(cimg, cimg1, Size(), sf, sf);
				imshow("corners", cimg1);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,
				30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_FIX_ASPECT_RATIO +
		CV_CALIB_ZERO_TANGENT_DIST +
		CV_CALIB_SAME_FOCAL_LENGTH +
		CV_CALIB_RATIONAL_MODEL +
		CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
	cout << "done with RMS error=" << rms << endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0， co-planear between 3 point
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average reprojection err = " << err / npoints << endl;

	// save intrinsic parameters into the file, you can open with sublime text 3, a great software reading files
	FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinsics.yml", CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!showRectified)
		return;

	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useCalibrated)
	{
		// we already computed everything.
	}
	// OR ELSE HARTLEY'S METHOD
	else
		// use intrinsic parameters of each camera, but
		// compute the rectification transformation directly
		// from the fundamental matrix
	{
		vector<Point2f> allimgpt[2];
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < nimages; i++)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		Mat H1, H2;
		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}
	cout << endl << endl << "*****************************************************" << endl;
	cout << "The first camera's intrinsics:" << cameraMatrix[0] << endl;
	cout << "The first camera's distortion:" << distCoeffs[0] << endl;
	cout << endl << endl << "*****************************************************" << endl;
	cout << "The second camera's intrinsics:" << cameraMatrix[1] << endl;

	cout << "The second camera's distortion:" << distCoeffs[1] << endl;

	cout << endl << endl << "*****************************************************" << endl;

	cout << distCoeffs[0].at<double>(0,0)<< endl;

	cout << distCoeffs[0].at<double>(0, 1) << endl;

	cout << distCoeffs[0].at<double>(0, 2) << endl;

	distCoeffs[0].at<double>(0, 7) = distCoeffs[1].at<double>(0, 7) = 0;
	distCoeffs[1].at<double>(0, 1) = distCoeffs[0].at<double>(0, 1);

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	//simple camera's principle
//	Mat map1, map2;
//	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], Mat(), getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);

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
	int imagecount = 0;
	char saveName[256];

	

	for (i = 0; i < nimages; i++)
//	for (i = 19; i >= 0;i--)
	{   
		for (k = 0; k < 2; k++)
		//for (k = 2; k > 0;k--)
		{
			Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg,rectifyimg;

	//		Mat destin(img.rows *5/2, img.cols / 2, img.type());
	//		resize(img, img, destin.size(), INTER_CUBIC);
			imshow("good image", img);
		//	img = image[k];
	//		cout << "good image size:" << img.cols << endl << "horizontal:" << img.rows << endl;
	//		cout << rmap[k][0] << endl << rmap[k][1] << endl;
			remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);

			remap(img, rectifyimg, map[k][0], map[k][1], INTER_LINEAR);
	//		resize(rimg, rimg, Size(30,30), INTER_CUBIC);
			imshow("r image", rimg);
			imshow("rectifying image from single camera", rectifyimg);
			rimg = rectifyimg;
	//		sprintf_s(saveName, "E:\\Dr_Ye\\dai%03d.jpeg", imagecount++);
	//		imwrite(saveName, rimg);
		//	rimg = img;
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
			if (useCalibrated)
			{
				sf = 0.5;
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
				if (k == 0)
				{
			//		imshow("rectified canvasPart1_right", canvasPart(vroi));
					imshow("rectified canvasPart1_left", canvasPart);
					sprintf_s(saveName, "E:\\Dr_Ye\\%03d.jpg", imagecount++);
					imwrite(saveName, canvasPart(vroi));
					imwrite(saveName, canvasPart);
				}
				else
				{
			//		imshow("rectified canvasPart1_left", canvasPart(vroi));
					sprintf_s(saveName, "E:\\Dr_Ye\\%03d.jpg", imagecount++);
			//		imwrite(saveName, canvasPart(vroi));
					imwrite(saveName, canvasPart);
				}

			}
			imshow("rectified canvasPart2", canvasPart);
			imshow("cimg", cimg);

		}


		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", canvas);
		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}


static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main(int argc, char** argv)
{
	Size boardSize;      //标定板的尺寸6X7
	string imagelistfn;  //采集到的图片帧数
	bool showRectified = true;

	for (int i = 1; i < argc; i++)
	{
		if (string(argv[i]) == "-w")         //命令行方式输入参数，在debugger中，输入宽；
		{
			if (sscanf_s(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
			{
				cout << "invalid board width" << endl;
				return print_help();
			}
		}
		else if (string(argv[i]) == "-h")      //输入高
		{
			if (sscanf_s(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
			{
				cout << "invalid board height" << endl;
				return print_help();
			}
		}
		else if (string(argv[i]) == "-nr")   //是否显示修饰以后的
			showRectified = false;
		else if (string(argv[i]) == "--help")
			return print_help();
		else if (argv[i][0] == '-')
		{
			cout << "invalid option " << argv[i] << endl;
			return 0;
		}
		else
			imagelistfn = argv[i];
	}

	if (imagelistfn == "")
	{
	//	imagelistfn = "7_6_chessboard.xml";
	//	imagelistfn = "stereo_calib.xml";     //从文件中输入；
		imagelistfn = "stereo_calib8_1.xml";     //从文件中输入；
	//	imagelistfn = "transfer_image.xml";
	//	boardSize = Size(9, 6);
	//	boardSize = Size(6, 7);
		boardSize = Size(7, 6);
	}
	else if (boardSize.width <= 0 || boardSize.height <= 0)
	{
		cout << "if you specified XML file with chessboards, you should also specify the board width and height (-w and -h options)" << endl;
		return 0;
	}

	vector<string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
		return print_help();
	}
	/*
	Mat image[2];
	string imgpath1 = argv[1];  //right image;
	string imgpath2 = argv[2];   //left image;

	image[0] = imread(imgpath1, 1);
	image[1] = imread(imgpath2, 1);

	*/
	StereoCalib(imagelist, boardSize, true, showRectified);   //直接标定；
	return 0;
}
