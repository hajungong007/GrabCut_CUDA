#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include "grabcut.h"
#include "funcs.h"
#include "fastcode.h"

using namespace std;
using namespace cv;

void maskShow(const Mat& mask, Mat& mask4show)
{
    fastcode::maskShowCaller(mask, mask4show);
}

void segResultShow(const Mat& img, const Mat& mask, Mat& segResult)
{
    fastcode::segResultShowCaller(img, mask, segResult);
}

void maskBinary(const Mat& mask, Mat& maskResult)
{
    fastcode::maskBinaryCaller(mask, maskResult);
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
	const GMM& bgdModelC, const GMM& fgdModelC, const GMM& bgdModelDiff, const GMM& fgdModelDiff, double * GMMonGPU,
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
    Mat imgDiffs[3];
    split(imgDiff, imgDiffs);
	// make the mask from imgDiff
	Mat maskFG, maskBG;
	maskFG.create(colorImg.size(), CV_8UC1); maskFG.setTo(0);
	maskBG.create(colorImg.size(), CV_8UC1); maskBG.setTo(0);
    fastcode::thresholdCaller(imgDiffs[0], imgDiffs[1], imgDiffs[2], maskFG, maskBG);
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
		bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff, GMMonGPU,
		alphaC, alphadiff, betaC, betadiff);


}
