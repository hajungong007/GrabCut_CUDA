#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include "grabcut.h"
#include "funcs.h"

using namespace std;
using namespace cv;

void maskShow(const Mat& mask, Mat& mask4show)
{
	mask4show.create( mask.size(), CV_8UC1);
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{			
			if (mask.at<uchar>(j,i) == GC_BGD)
				mask4show.at<uchar>(j,i) = 0;
			if (mask.at<uchar>(j,i) == GC_PR_BGD)
				mask4show.at<uchar>(j,i) = 1;
			if (mask.at<uchar>(j,i) == GC_PR_FGD)
				mask4show.at<uchar>(j,i) = 2;
			if (mask.at<uchar>(j,i) == GC_FGD)
				mask4show.at<uchar>(j,i) = 3;
		}
	}
}

void segResultShow(const Mat& img, const Mat& mask, Mat& segResult)
{
	img.copyTo(segResult);
	for (int j = 0; j < img.rows; j++)
	{
		for (int i = 0; i < img.cols; i++)
		{			
			if (mask.at<uchar>(j,i) == GC_BGD || mask.at<uchar>(j,i) == GC_PR_BGD  )
				segResult.at<Vec3b>(j,i) = segResult.at<Vec3b>(j,i) / 4;				
		}
	}
}

void maskBinary(const Mat& mask, Mat& maskResult)
{
	maskResult.create( mask.size(), CV_8UC1);
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{			
			if (mask.at<uchar>(j,i) == GC_FGD || mask.at<uchar>(j,i) == GC_PR_FGD  )
				maskResult.at<uchar>(j,i) = 255;
			else
				maskResult.at<uchar>(j,i) = 0;
		}
	}
}

string convertInt(int number)
{
	stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
}

string convertInt2(int number)
{
	string ss;
	if (number < 10)
		ss = "0"+convertInt(number);
	else
		ss = convertInt(number);
	return ss;//return a string with the contents of the stream
}

void segByimgDiff_color(const Mat& colorImg, const Mat& bgColorImg, Mat& maskC, 
	const Mat& bgdModelC, const Mat& fgdModelC, const Mat& bgdModelDiff, const Mat& fgdModelDiff,
	double alphaC, double alphadiff, double betaC, double betadiff)
{
	// important parameters!!!
	int rect1x = 0, rect1y = 0, rect2x = 640, rect2y = 1080/3; //NOTICE: the setting of rect1x/rect2x need to be changed
	int threshold1 = 10, threshold2 = 30;

	// create rect area
	Rect rect;
	rect.x = rect1x;
	rect.y = rect1y;
	rect.width = rect2x - rect1x;
	rect.height = rect2y - rect1y;

	Mat imgDiff = abs(colorImg - bgColorImg);
	// make the mask from imgDiff
	Mat maskFG, maskBG;
	maskFG.create(colorImg.size(), CV_8UC1); maskFG.setTo(0);
	maskBG.create(colorImg.size(), CV_8UC1); maskBG.setTo(0);
	for (int j = 0; j < colorImg.rows; j++)
		for (int i = 0; i < colorImg.cols; i++)
		{
			if (imgDiff.at<Vec3b>(j,i)[0] > threshold2 ||
				imgDiff.at<Vec3b>(j,i)[1] > threshold2 ||
				imgDiff.at<Vec3b>(j,i)[2] > threshold2)
				maskFG.at<uchar>(j,i) = 1;
			if (imgDiff.at<Vec3b>(j,i)[0] < threshold1 &&
				imgDiff.at<Vec3b>(j,i)[1] < threshold1 &&
				imgDiff.at<Vec3b>(j,i)[2] < threshold1)
				maskBG.at<uchar>(j,i) = 1;
		}
	Mat element = getStructuringElement( MORPH_RECT, Size( 3, 3 ), Point( 1, 1 ) );
	erode( maskFG, maskFG, element );
	erode( maskBG, maskBG, element );
	maskC.create( colorImg.size(), CV_8UC1);
	maskC.setTo( GC_BGD ); 
	(maskC(rect)).setTo( Scalar(GC_PR_FGD) );
	for (int j = rect1y; j < rect2y; j++)
		for (int i = rect1x; i < rect2x; i++)
		{			
			if (maskFG.at<uchar>(j,i) == 1)
				maskC.at<uchar>(j,i) = GC_FGD;
			if (maskBG.at<uchar>(j,i) == 1)
				maskC.at<uchar>(j,i) = GC_BGD;
		}

	// segmentation from sample
	grabCut_lockFGBGmodel_linearCombine( colorImg, imgDiff, maskC, 
		bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff, 
		alphaC, alphadiff, betaC, betadiff);
	//grabCut_lockFGBGmodel_multiCombine( colorImg, imgDiff, maskC, 
	//	bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff, 
	//	alphaC, alphadiff, betaC, betadiff);


}
