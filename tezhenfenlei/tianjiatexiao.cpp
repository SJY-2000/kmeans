#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void segColor();
int createMaskByKmeans(cv::Mat src, cv::Mat & mask, cv::Mat &skinMat);

int main()
{
	double start = static_cast<double>(getTickCount());

	segColor();

	double time = ((double)getTickCount() - start) / getTickFrequency();

	cout << "processing time:" << time / 1000 << "ms" << endl;

	system("pause");

	return 0;
}
void segColor()
{

	Mat src = imread("G:\\picture\\t2.png");
	Mat dst = imread("G:\\picture\\t1.png");
	Mat skinMat;

	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	createMaskByKmeans(src, mask, skinMat);
	resize(skinMat, skinMat, dst.size(), 0, 0, INTER_LINEAR);

	addWeighted(dst, 0.3, skinMat, 0.7, 0, dst);

	imshow("skinMat", skinMat);
	imshow("src", src);
	imshow("mask", mask);
	imshow("dst", dst);
	waitKey(0);
}

int createMaskByKmeans(cv::Mat src, cv::Mat & mask, cv::Mat &skinMat)
{
	if ((mask.type() != CV_8UC1)
		|| (src.size() != mask.size())
		) {
		return 0;
	}

	int width = src.cols;
	int height = src.rows;

	int pixNum = width * height;
	int clusterCount = 2;
	Mat labels;
	Mat centers;
	Mat otherMat;

	Mat sampleData = src.reshape(0, pixNum);
	Mat km_data;
	sampleData.convertTo(km_data, CV_32F);

	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(km_data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	uchar fg[2] = { 0,255 };
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			mask.at<uchar>(row, col) = fg[labels.at<int>(row*width + col)];
		}
	}
	src.copyTo(skinMat, mask);              

	return 0;
}