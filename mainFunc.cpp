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
    clock_t starttime = clock();
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
	Mat bgColorImgs[3];
	split(bgColorImg, bgColorImgs);
	Mat colorImgs[3];
	Mat resultCs[3];

    GMM bgdGMM_C(bgdModelC), fgdGMM_C(fgdModelC);
    GMM bgdGMM_diff(bgdModelDiff), fgdGMM_diff(fgdModelDiff);

    int l = 4/*models*/*(5/*components*/*(1+3+9/*weight,mean,cov*/+9/*inverseconv*/)+5/*convdeterm*/);

    double * gmmmodels = new double[l];

    int cursor = 0;

    int i,j,k;

    for(i = 0; i < 5; i++) gmmmodels[cursor++] = bgdGMM_C.coefs[i];
    for(i = 0; i < 15; i++) gmmmodels[cursor++] = bgdGMM_C.mean[i];
    for(i = 0; i < 45; i++) gmmmodels[cursor++] = bgdGMM_C.cov[i];
    for(i = 0; i < 5; i++)
        for(j = 0; j < 3; j++)
            for(k = 0; k < 3; k++)
                gmmmodels[cursor++] = bgdGMM_C.inverseCovs[i][j][k];
    for(i = 0; i < 5; i++) gmmmodels[cursor++] = bgdGMM_C.covDeterms[i];


    for(i = 0; i < 5; i++) gmmmodels[cursor++] = fgdGMM_C.coefs[i];
    for(i = 0; i < 15; i++) gmmmodels[cursor++] = fgdGMM_C.mean[i];
    for(i = 0; i < 45; i++) gmmmodels[cursor++] = fgdGMM_C.cov[i];
    for(i = 0; i < 5; i++)
        for(j = 0; j < 3; j++)
            for(k = 0; k < 3; k++)
                gmmmodels[cursor++] = fgdGMM_C.inverseCovs[i][j][k];
    for(i = 0; i < 5; i++) gmmmodels[cursor++] = fgdGMM_C.covDeterms[i];

    for(i = 0; i < 5; i++) gmmmodels[cursor++] = bgdGMM_diff.coefs[i];
    for(i = 0; i < 15; i++) gmmmodels[cursor++] = bgdGMM_diff.mean[i];
    for(i = 0; i < 45; i++) gmmmodels[cursor++] = bgdGMM_diff.cov[i];
    for(i = 0; i < 5; i++)
        for(j = 0; j < 3; j++)
            for(k = 0; k < 3; k++)
                gmmmodels[cursor++] = bgdGMM_diff.inverseCovs[i][j][k];
    for(i = 0; i < 5; i++) gmmmodels[cursor++] = bgdGMM_diff.covDeterms[i];

    for(i = 0; i < 5; i++) gmmmodels[cursor++] = fgdGMM_diff.coefs[i];
    for(i = 0; i < 15; i++) gmmmodels[cursor++] = fgdGMM_diff.mean[i];
    for(i = 0; i < 45; i++) gmmmodels[cursor++] = fgdGMM_diff.cov[i];
    for(i = 0; i < 5; i++)
        for(j = 0; j < 3; j++)
            for(k = 0; k < 3; k++)
                gmmmodels[cursor++] = fgdGMM_diff.inverseCovs[i][j][k];
    for(i = 0; i < 5; i++) gmmmodels[cursor++] = fgdGMM_diff.covDeterms[i];

    double * GMMonGPU;

    fastcode::copyGMMtoGPU(gmmmodels, GMMonGPU, l);

	for (i = 201; i < 250; i = i+1 ) // NOTICE: These are the frames which need to be segmented
	{
		string colorImg_filename = "data/" + convertInt(i) + ".png";
		Mat colorImg = imread(colorImg_filename, CV_LOAD_IMAGE_COLOR);
		split(colorImg, colorImgs);
		Mat mask;
		// begin segmentation
		// Sample 1:
		segByimgDiff_color(colorImg, bgColorImg, mask, bgdGMM_C, fgdGMM_C, bgdGMM_diff, fgdGMM_diff);
		// show result
		Mat resultC;
		segResultShow(colorImgs[0], mask, resultCs[0]);
		segResultShow(colorImgs[1], mask, resultCs[1]);
		segResultShow(colorImgs[2], mask, resultCs[2]);
		merge(resultCs, 3, resultC);
		imwrite("results/"+convertInt(i)+" 0.5_0.5_0.5_0.5"+".png", resultC);
		Mat mask4show;
		maskBinary(mask, mask4show);
		imwrite("results/"+convertInt(i)+" 0.5_0.5_0.5_0.5_mask"+".png", mask4show);


		// Sample 2:
//		segByimgDiff_color(colorImg, bgColorImg, mask, bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff,1,0,1,0);
//		// show result
//		segResultShow(colorImg, mask, resultC);
//		imwrite("results/"+convertInt(i)+" 1_0_1_0"+".png", resultC);
//		maskBinary(mask, mask4show);
//		imwrite("results/"+convertInt(i)+" 1_0_1_0_mask"+".png", mask4show);

		// Sample 3:
//		segByimgDiff_color(colorImg, bgColorImg, mask, bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff,0,1,0,1);
//		// show result
//		segResultShow(colorImg, mask, resultC);
//		imwrite("results/"+convertInt(i)+" 0_1_0_1"+".png", resultC);
//		maskBinary(mask, mask4show);
//		imwrite("results/"+convertInt(i)+" 0_1_0_1_mask"+".png", mask4show);
	}
    clock_t endtime = clock();

    cout<<"Time used :"<<(double)(endtime- starttime)/CLOCKS_PER_SEC*1000.0<<" milliseconds\n";

    delete [] gmmmodels;
    fastcode::freeGMMonGPU(GMMonGPU);

	waitKey();
	return 0;
}
