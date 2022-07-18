
#include "stdafx.h"


#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <stdio.h>

using namespace cv;

void F_Gray2Color(CvMat* gray_mat, CvMat* color_mat)
{
	if (color_mat)
		cvZero(color_mat);

	int stype = CV_MAT_TYPE(gray_mat->type), dtype = CV_MAT_TYPE(color_mat->type);
	int rows = gray_mat->rows, cols = gray_mat->cols;

	// 判断输入的灰度图和输出的伪彩色图是否大小相同、格式是否符合要求
	if (CV_ARE_SIZES_EQ(gray_mat, color_mat) && stype == CV_8UC1 && dtype == CV_8UC3)
	{
		CvMat* red = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
		CvMat* green = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
		CvMat* blue = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
		CvMat* mask = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);

		// 计算各彩色通道的像素值
		cvSubRS(gray_mat, cvScalar(255), blue);	// blue(I) = 255 - gray(I)
		cvCopy(gray_mat, red);			// red(I) = gray(I)
		cvCopy(gray_mat, green);			// green(I) = gray(I),if gray(I) < 128
		cvCmpS(green, 128, mask, CV_CMP_GE);	// green(I) = 255 - gray(I), if gray(I) >= 128
		cvSubRS(green, cvScalar(255), green, mask);
		cvConvertScale(green, green, 2.0, 0.0);

		// 合成伪彩色图
		cvMerge(blue, green, red, NULL, color_mat);

		cvReleaseMat(&red);
		cvReleaseMat(&green);
		cvReleaseMat(&blue);
		cvReleaseMat(&mask);
	}
}

/*

void saveDisp(const char* filename, const Mat& mat)
{
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "%02d/n", mat.rows);
	fprintf(fp, "%02d/n", mat.cols);
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			short disp = mat.at<short>(y, x); // 这里视差矩阵是CV_16S 格式的，故用 short 类型读取
			fprintf(fp, "%d/n", disp); // 若视差矩阵是 CV_32F 格式，则用 float 类型读取
		}
	}
	fclose(fp);
}     */

/*************************************Matlab 可以显示三维信息***********************************************
function img = txt2img(filename)
data = importdata(filename);
r = data(1);    % 行数
c = data(2);    % 列数
disp = data(3:end); % 视差
vmin = min(disp);
vmax = max(disp);
disp = reshape(disp, [c, r])'; % 将列向量形式的 disp 重构为 矩阵形式
%  OpenCV 是行扫描存储图像，Matlab 是列扫描存储图像
%  故对 disp 的重新排列是首先变成 c 行 r 列的矩阵，然后再转置回 r 行 c 列
img = uint8(255 * (disp - vmin) / (vmax - vmin));
mesh(disp);
set(gca, 'YDir', 'reverse');  % 通过 mesh 方式绘图时，需倒置 Y 轴方向
axis tight; % 使坐标轴显示范围与数据范围相贴合，去除空白显示区

*******************************************************************************************/

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
		"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
//	FILE* fp = fopen(filename, "wt");
	//FILE* fp=new FILE;
	FILE* fp = NULL;

	fopen_s(&fp,filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	//delete fp;
	fclose(fp);
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

int main(int argc, char** argv)
{
	const char* algorithm_opt = "--algorithm=";
	const char* maxdisp_opt = "--max-disparity=";
	const char* blocksize_opt = "--blocksize=";
	const char* nodisplay_opt = "--no-display";
	const char* scale_opt = "--scale=";

	if (argc < 3)
	{
		print_help();
		return 0;
	}
	const char* img1_filename = 0;
	const char* img2_filename = 0;
	const char* intrinsic_filename = 0;
	const char* extrinsic_filename = 0;
	const char* disparity_filename = 0;
	const char* point_cloud_filename = 0;

	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3 };
	int alg = STEREO_SGBM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	bool no_display = false;
	float scale = 1.f;

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	for (int i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-')
		{
			if (!img1_filename)
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if (strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0)
		{
			char* _alg = argv[i] + strlen(algorithm_opt);
			alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
				strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
				strcmp(_alg, "hh") == 0 ? STEREO_HH :
				strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
			if (alg < 0)
			{
				printf("Command-line parameter error: Unknown stereo algorithm\n\n");
				print_help();
				return -1;
			}
		}
		else if (strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0)
		{
			if (sscanf_s(argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities) != 1 ||
				numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
			{
				printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
				print_help();
				return -1;
			}
		}
		else if (strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0)
		{
			if (sscanf_s(argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize) != 1 ||
				SADWindowSize < 1 || SADWindowSize % 2 != 1)
			{
				printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
				return -1;
			}
		}
		else if (strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0)
		{
			if (sscanf_s(argv[i] + strlen(scale_opt), "%f", &scale) != 1 || scale < 0)
			{
				printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
				return -1;
			}
		}
		else if (strcmp(argv[i], nodisplay_opt) == 0)
			no_display = true;
		else if (strcmp(argv[i], "-i") == 0)
			intrinsic_filename = argv[++i];
		else if (strcmp(argv[i], "-e") == 0)
			extrinsic_filename = argv[++i];
		else if (strcmp(argv[i], "-o") == 0)
			disparity_filename = argv[++i];
		else if (strcmp(argv[i], "-p") == 0)
			point_cloud_filename = argv[++i];
		else
		{
			printf("Command-line parameter error: unknown option %s\n", argv[i]);
			return -1;
		}
	}

	if (!img1_filename || !img2_filename)
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}

	if ((intrinsic_filename != 0) ^ (extrinsic_filename != 0))
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		return -1;
	}

	if (extrinsic_filename == 0 && point_cloud_filename)
	{
		printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
		return -1;
	}

	int color_mode = alg == STEREO_BM ? 0 : -1;
	Mat img1 = imread(img1_filename, color_mode);
	Mat img2 = imread(img2_filename, color_mode);

	if (scale != 1.f)
	{
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}

	Size img_size = img1.size();

//	printf("The size of the image: %f pixels\n", img_size);

	Rect roi1, roi2;
	Mat Q;

	if (intrinsic_filename)
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename);
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, CV_STORAGE_READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename);
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;
	}

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = img1.channels();

	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = alg == STEREO_HH;

	var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
	var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
	var.nIt = 25;
	var.minDisp = -numberOfDisparities;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
	var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING;

	Mat disp, disp8;
	//Mat img1p, img2p, dispp;
	//copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	int64 t = getTickCount();
	if (alg == STEREO_BM)
		bm(img1, img2, disp);
	else if (alg == STEREO_VAR) {
		var(img1, img2, disp);
	}
	else if (alg == STEREO_SGBM || alg == STEREO_HH)
		sgbm(img1, img2, disp);
	t = getTickCount() - t;
	printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	if (alg != STEREO_VAR)
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	else
		disp.convertTo(disp8, CV_8U);
	if (!no_display)
	{
		namedWindow("left", 1);
		imshow("left", img1);
		namedWindow("right", 1);
		imshow("right", img2);
		namedWindow("disparity", 0);
		imshow("disparity", disp8);
		printf("press any key to continue...");
		fflush(stdout);
		waitKey();
		printf("\n");
	}

	if (disparity_filename)
		imwrite(disparity_filename, disp8);

	if (point_cloud_filename)
	{
		printf("storing the point cloud...");
		fflush(stdout);
		Mat xyz;
		reprojectImageTo3D(disp, xyz, Q, true);
		saveXYZ(point_cloud_filename, xyz);
		printf("\n");
	}

	return 0;
}

