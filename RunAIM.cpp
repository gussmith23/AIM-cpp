/*
 * RunAIM.cpp
 *
 *  Created on: Oct 28, 2012
 *      Author: siddharthadvani
 */

#include "opencv2/opencv.hpp"
#include "AIM.h"

class AIM;

using namespace cv;
using namespace std;

int main (int argc, char* argv[])
{
	///////////////////////////////
	//{ // Initialize variables
	//////////////////////////////

	Mat OriginalInputImage, Basis;

	// Outputs
	Mat InfoImage, SaliencyImage, ThreshImage;

	int resize_percentage;
	string run_mode;

	string basis_file_name;

	///////////////////////////////
	//} // Initialize variables
	//////////////////////////////

	if (argc < 2 )
	{
		cerr << "Usage: ./cvAimLPT file_name <b> <100> <basis.yml>" << endl;
		return -1;
	}
	else
		OriginalInputImage = imread(argv[1], CV_LOAD_IMAGE_COLOR+CV_LOAD_IMAGE_ANYDEPTH);

	///////////////////////////////
	//{ // Set defaults for arguments
	//////////////////////////////

	if(argc < 3) run_mode = "b";    	// regular AIM
	else run_mode = argv[2];

	if(argc < 4) resize_percentage = 100;    	// no resize
	else resize_percentage = atoi (argv[3]);

	if(argc < 5) basis_file_name = "basis.yml";
	else basis_file_name = argv[4];
	///////////////////////////////
	//} // Set defaults for arguments
	//////////////////////////////


	///////////////////////////////
	//{ // Pre-Processing
	//////////////////////////////

	//Mat InputImage = cvCreateMat(256,512,OriginalInputImage.type());
	Mat InputImage = cvCreateMat(480,640,OriginalInputImage.type());
	//Mat InputImage = cvCreateMat(960,1280,OriginalInputImage.type());
	//Mat InputImage = cvCreateMat(768,1024,OriginalInputImage.type());

	//Mat InputImage = cvCreateMat((OriginalInputImage.rows*resize_percentage)/100, 								(OriginalInputImage.cols*resize_percentage)/100,OriginalInputImage.type());

	resize(OriginalInputImage, InputImage, InputImage.size(),0,0);

	//imshow( "please", InputImage);
	//waitKey(0);

	double minVal, maxVal;
	minMaxLoc(InputImage, &minVal, &maxVal, NULL, NULL);

	// Normalize image
	Mat NormInputImage = cvCreateMat(InputImage.rows, InputImage.cols, CV_32FC3); // AIM assumes RGB channels

	InputImage.convertTo(NormInputImage,CV_32FC3,1/maxVal,0);
	//cout<<"test"<<NormInputImage.channels()<<endl;
	///////////////////////////////
	//} // Pre-Processing
	//////////////////////////////


	///////////////////////////////
	//{ // MAIN
	//////////////////////////////

	//AIM mkAIM;
	//AIM mkAIM("basis.yml", 19);

	if (run_mode == "b")
	{
		AIM mkAIM;

		mkAIM.LoadBasis(basis_file_name);

		mkAIM.ist = mkAIM.GetSaliency(NormInputImage);

		//mkAIM.ist = mkAIM.GetSaliencyMT(NormInputImage);
		//InfoImage = mkAIM.ist.InfoImage;
//		SaliencyImage = mkAIM.ist.SaliencyImage;
	//	ThreshImage = mkAIM.ist.ThreshImage;
	}
	else
	{
		AIM mkAIM7, mkAIM17, mkAIM27, mkAIM37;  // 4 scales

		mkAIM7.LoadBasis("specgaborkernels_scale7.yml");
		mkAIM17.LoadBasis("specgaborkernels_scale17.yml");
		mkAIM27.LoadBasis("specgaborkernels_scale27.yml");
		mkAIM37.LoadBasis("specgaborkernels_scale37.yml");

		// AIM - STEP 1 for Gabor 7x7
		vector <Mat> ConvolvedImageScale7 = mkAIM7.GetGaborConvolutionBasis(NormInputImage);

		// AIM - STEP 1 for Gabor 17x17
		vector <Mat> ConvolvedImageScale17 = mkAIM17.GetGaborConvolutionBasis(NormInputImage);

		// AIM - STEP 1 for Gabor 27x27
		vector <Mat> ConvolvedImageScale27 = mkAIM27.GetGaborConvolutionBasis(NormInputImage);

		// AIM - STEP 1 for Gabor 37x37
		vector <Mat> ConvolvedImageScale37 = mkAIM37.GetGaborConvolutionBasis(NormInputImage);

		vector <Mat> ConvolvedImageAllScales(
											ConvolvedImageScale7.capacity()+
											ConvolvedImageScale17.capacity()+
											ConvolvedImageScale27.capacity()+
											ConvolvedImageScale37.capacity()
											);

		// There may be a better way to stack up all convolved results into a single array of arrays. For now just doing it this way
		for (unsigned int i = 0; i < ConvolvedImageScale7.capacity(); i++)
		{
			ConvolvedImageAllScales[i] = ConvolvedImageScale7[i];
		}

		for (unsigned int j = 0; j < ConvolvedImageScale17.capacity(); j++)
		{
			ConvolvedImageAllScales[ConvolvedImageScale7.capacity() + j] = ConvolvedImageScale17[j];
		}

		for (unsigned int k = 0; k < ConvolvedImageScale27.capacity(); k++)
		{
			ConvolvedImageAllScales[ConvolvedImageScale7.capacity() + ConvolvedImageScale17.capacity()+k] = ConvolvedImageScale27[k];
		}

		for (unsigned int l = 0; l < ConvolvedImageScale37.capacity(); l++)
		{
			ConvolvedImageAllScales[ConvolvedImageScale7.capacity() + ConvolvedImageScale17.capacity() + ConvolvedImageScale27.capacity()+l] = ConvolvedImageScale37[l];
		}

		Size InfoMapSize(InputImage.cols, InputImage.rows);

		// STEP 2 & 3 : Get Density and Convert to Log Likelihood (Can use any of the AIM object)
		InfoImage = mkAIM7.GetInfoMapfromConvolution(ConvolvedImageAllScales, InfoMapSize);

		// Post Processing
		ThreshImage = mkAIM7.GetThreshMap(InfoImage, 0.8);

		SaliencyImage = mkAIM7.DoPixelMasking(NormInputImage, ThreshImage);
	}


	///////////////////////////////
	//} // MAIN
	//////////////////////////////


	///////////////////////////////
	//{ // Display I, S, T
	//////////////////////////////

//	imshow("Threshmap", ThreshImage);
//	imshow("Saliency", SaliencyImage);
//	imshow("Infomap", InfoImage);

	////waitKey(0);

	///////////////////////////////
	//} // Display I, S, T
	//////////////////////////////

	return 0;
}

