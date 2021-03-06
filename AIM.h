/*
 * AIM.h
 *
 *  Created on: Oct 28, 2012
 *      Author: siddharthadvani
 */

#ifndef AIM_H_
#define AIM_H_

//#include "opencv2/opencv.hpp"
#include "opencv/cv.h" 
#include "opencv/cvaux.h" 
#include "opencv/highgui.h" 
#include <iostream>
#include "pthread.h"

using namespace cv;
using namespace std;

class AIM {
public:
	AIM();
	AIM(string BasisFilename, int NumFixedPointFractionalBits);
	virtual ~AIM();

private:
//	string Filename;
	Mat basis;
	bool Debug;

public:

	bool UseFixedPoint;
	double FixedPointScaleFactor;
	struct IST {
		Mat InfoImage;
		Mat SaliencyImage;
		Mat ThreshImage;
	} ist;

public:
	void LoadBasis(string Filename);

	Mat GetDensity(Mat InputImage);

	IST GetSaliency(Mat InputImage);
// added by nandhini //
//	static	void *basisthreadfunc(void *targ);
	IST GetSaliencyMT(Mat InputImage);

	IST GetGaborSaliency(Mat InputImage);

	vector <Mat> GetGaborConvolutionBasis(Mat InputImage);

	Mat GetInfoMapfromConvolution(vector <Mat> BasisImages, Size InfoMapSize);

	Mat GetThreshMap (Mat InfoMap, double dispThresh);

	Mat DoPixelMasking(Mat InputImage, Mat ThreshMap);

	Mat GetConvolution2D(Mat InputImage, Mat Kernel);

	double CalcPercentile(Mat sequence, double percentile);
};

#endif /* AIM_HPP_ */
