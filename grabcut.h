#ifdef __cplusplus
#if __cplusplus
//extern "C"{
#endif
#endif
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

void learnGMMsFromSample( const Mat& img, const Mat& mask, Mat& bgdGMMPara, Mat& fgdGMMPara );
/* learn GMM model parameters from samples which are given by the user */


void grabCut_lockFGBGmodel_linearCombine( InputArray _colorImg, InputArray _imgDiff, InputOutputArray _maskC,
										 GMM & bgdGMM_C, GMM & fgdGMM_C, GMM & bgdGMM_diff, GMM & fgdGMM_diff,
										 double alphaC, double alphadiff, double betaC, double betadiff);
/* linearly combine two cues (imgDiff + color) by the form: P(to source, to sink) = alphaC * -log(P(color)) + alphadiff * -log(P(diff)),
                                                            W(pixel, pixel) = betaC * W_based_on_color + betadiff * W_based_on_diff */

#ifdef __cplusplus
#if __cplusplus
//}
#endif
#endif
