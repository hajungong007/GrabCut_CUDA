#ifndef GRABCUT_H
#define GRABCUT_H

#ifdef __cplusplus
#if __cplusplus
//extern "C"{
#endif
#endif
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};
void learnGMMsFromSample( const Mat& img, const Mat& mask, Mat& bgdGMMPara, Mat& fgdGMMPara );
/* learn GMM model parameters from samples which are given by the user */


void grabCut_lockFGBGmodel_linearCombine( InputArray _colorImg, InputArray _imgDiff, InputOutputArray _maskC,
										 const GMM & bgdGMM_C, const GMM & fgdGMM_C, const GMM & bgdGMM_diff, const GMM & fgdGMM_diff,
										 double alphaC, double alphadiff, double betaC, double betadiff);
/* linearly combine two cues (imgDiff + color) by the form: P(to source, to sink) = alphaC * -log(P(color)) + alphadiff * -log(P(diff)),
                                                            W(pixel, pixel) = betaC * W_based_on_color + betadiff * W_based_on_diff */

#ifdef __cplusplus
#if __cplusplus
//}
#endif
#endif

#endif
