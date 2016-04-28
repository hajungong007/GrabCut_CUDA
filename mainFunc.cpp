#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "grabcut.h"
#include "funcs.h"
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int main()
{
    time_t starttime;
    time(&starttime);
	// sample of  color img and imgDiff
	string colorImgSample_filename = "samples/color.png";
	Mat colorImgSample = imread(colorImgSample_filename, CV_LOAD_IMAGE_COLOR);
	string imgDiffSample_filename = "samples/diff.png";
	Mat imgDiffSample = imread(imgDiffSample_filename, CV_LOAD_IMAGE_COLOR);
	string maskImgSample_filename =  "samples/mask.png";
	Mat maskSample = imread(maskImgSample_filename, CV_LOAD_IMAGE_GRAYSCALE );

	// learn bgdGMM & fgdGMM
	Mat bgdModelC, fgdModelC;
	Mat bgdModelDiff, fgdModelDiff;
	learnGMMsFromSample( colorImgSample, maskSample, bgdModelC, fgdModelC );
	learnGMMsFromSample( imgDiffSample, maskSample, bgdModelDiff, fgdModelDiff );

	string bgColorImg_filename = "data/" + convertInt(101) + ".png"; // NOTICE: This is background image!
	Mat bgColorImg = imread(bgColorImg_filename, CV_LOAD_IMAGE_COLOR);
	for (int i = 201; i < 300; i = i+1 ) // NOTICE: These are the frames which need to be segmented
	{
		string colorImg_filename = "data/" + convertInt(i) + ".png";
		Mat colorImg = imread(colorImg_filename, CV_LOAD_IMAGE_COLOR);

		Mat mask;
		// begin segmentation
		// Sample 1:
		segByimgDiff_color(colorImg, bgColorImg, mask, bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff);
		// show result
		Mat resultC;
		segResultShow(colorImg, mask, resultC);
		imwrite("results/"+convertInt(i)+" 0.5_0.5_0.5_0.5"+".png", resultC);
		Mat mask4show;
		maskBinary(mask, mask4show);
		imwrite("results/"+convertInt(i)+" 0.5_0.5_0.5_0.5_mask"+".png", mask4show);


//		// Sample 2:
//		segByimgDiff_color(colorImg, bgColorImg, mask, bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff,1,0,1,0);
//		// show result
//		segResultShow(colorImg, mask, resultC);
//		imwrite("results/"+convertInt(i)+" 1_0_1_0"+".png", resultC);
//		maskBinary(mask, mask4show);
//		imwrite("results/"+convertInt(i)+" 1_0_1_0_mask"+".png", mask4show);

//		// Sample 3:
//		segByimgDiff_color(colorImg, bgColorImg, mask, bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff,0,1,0,1);
//		// show result
//		segResultShow(colorImg, mask, resultC);
//		imwrite("results/"+convertInt(i)+" 0_1_0_1"+".png", resultC);
//		maskBinary(mask, mask4show);
//		imwrite("results/"+convertInt(i)+" 0_1_0_1_mask"+".png", mask4show);
	}
    time_t endtime;
    time(&endtime);

    cout<< "Time used: "<< difftime(starttime, endtime) << " seconds\n";

	waitKey();
	return 0;
}
