#ifdef __cplusplus
#if __cpluscplus
extern "C"{
#endif
#endif
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

void maskShow(const Mat& mask, Mat& mask4show);
/* sure_BG --- 0
   PR_BG --- 1
   PR_FG --- 2
   sure_FG --- 3 */

void segResultShow(const Mat& img, const Mat& mask, Mat& segResult);
/* FG --- keep the same;
   BG --- intensity / 4 */

void maskBinary(const Mat& mask, Mat& maskResult);
/* sure_FG & PR_FG ---> 255
   sure_BG & PR_BG ---> 0 */

void segByimgDiff_color(const Mat& colorImg, const Mat& bgColorImg, Mat& maskC,
	const GMM& bgdModelC, const GMM& fgdModelC, const GMM& bgdModelDiff, const GMM& fgdModelDiff,
	double alphaC=0.5, double alphadiff=0.5, double betaC=0.5, double betadiff=0.5);
/* segment img by combination of imgDiff cue and color cue
   input need learnt GMM models (bgdModelC, fgdModelC, bgdModelDiff, fgdModelDiff)
   alphaC, alphadiff, betaC, betadiff are weights of two cues; they are used in the construction of the graph
			P(to source, to sink) = alphaC * -log(P(color)) + alphadiff * -log(P(diff)),
            W(pixel, pixel) = betaC * W_based_on_color + betadiff * W_based_on_diff
	   or   W(pixel, pixel) = W_based_on_color^betaC + W_based_on_diff^betadiff
*/

string convertInt(int number);
string convertInt2(int number);
#ifdef __cplusplus
#if __cplusplus
#endif
#endif
