#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

void learnGMMsFromSample( const Mat& img, const Mat& mask, Mat& bgdGMMPara, Mat& fgdGMMPara );
/* learn GMM model parameters from samples which are given by the user */

void grabCut_origin( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode );
/* original grabcut() function provided by OpenCV */

void grabCut_lockFGBGmodel_linearCombine( InputArray _colorImg, InputArray _imgDiff, InputOutputArray _maskC, 
										 InputArray _bgdModelC, InputArray _fgdModelC, InputArray _bgdModelDiff, InputArray _fgdModelDiff, 
										 double alphaC, double alphadiff, double betaC, double betadiff);
/* linearly combine two cues (imgDiff + color) by the form: P(to source, to sink) = alphaC * -log(P(color)) + alphadiff * -log(P(diff)),
                                                            W(pixel, pixel) = betaC * W_based_on_color + betadiff * W_based_on_diff */

void grabCut_lockFGBGmodel_multiCombine( InputArray _colorImg, InputArray _imgDiff, InputOutputArray _maskC, 
										 InputArray _bgdModelC, InputArray _fgdModelC, InputArray _bgdModelDiff, InputArray _fgdModelDiff, 
										 double alphaC, double alphadiff, double betaC, double betadiff);
/* linearly combine two cues (imgDiff + color) by the form: P(to source, to sink) = alphaC * -log(P(color)) + alphadiff * -log(P(diff)),
                                                            W(pixel, pixel) = W_based_on_color^betaC + W_based_on_diff^betadiff */
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif 
