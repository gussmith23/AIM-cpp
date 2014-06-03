/*
 * AIM.cpp
 *
 *  Created on: Oct 28, 2012
 *      Author: siddharthadvani
 *
 *  Histogram Bins = 1000
 *  Pixel Masking Threshold = 80%
 *
 *  References :
 *
 *   1. Bruce, N.D.B., Tsotsos, J.K., Saliency,
 *   Attention, and Visual Search: An Information Theoretic
 *   Approach, Journal of Vision 9:3, pp.1-24, 2009,
 *   http://journalofvision.org/9/3/5/, doi:10.1167/9.3.5.
 *
 *   2. Bruce, N.D.B., Tsotsos, J.K., Saliency based on
 *   Information Maximization. Advances in Neural
 *   Information Processing Systems, 18.

 * 	 3. Bruce, N. Features that draw visual attention: An
 *   information theoretic perspective. Neurocomputing,
 *   v. 65-66, pp. 125-133, May 2005.
 *
 *   Extentions :
 *
 *   12/25/2012 - Added Gabor Implementation of AIM
 *   			  Trained set of Basis Kernels is replaced
 *   			  with Gabor Kernels at 4 scales
 *   			  with 6 orientations each
 */

#include "opencv2/opencv.hpp"
#include <sys/time.h>

#include "AIM.h"

#define MaxBasis 25

#define NUMTHREADS 4

using namespace cv;
using namespace std;


// added by nandhini //

struct basis_thread_arg
{
	int tid;
	Mat bfilter;
	Mat output;
	Mat input[3];
};






AIM::AIM() {
	// TODO Auto-generated constructor stub
	UseFixedPoint = false;
	Debug = false;
//	Filename = "basis.yml";
}

AIM::AIM(string BasisFilename, int NumFixedPointFractionalBits) {
	UseFixedPoint = true;
	FixedPointScaleFactor = pow((double)2,NumFixedPointFractionalBits);
	Debug = false;
//	Filename = BasisFilename;
}

AIM::~AIM() {
	// TODO Auto-generated destructor stub
}


void AIM::LoadBasis(string Filename){

	FileStorage fs(Filename, FileStorage::READ);

	if (UseFixedPoint)
	{
		fs["B"] >> basis;
		//Mat Basis = cvCreateMat(basis.rows,basis.cols,CV_16UC1);
	 	basis = basis * FixedPointScaleFactor;
	}
	else
		fs["B"] >> basis;
		//Mat Basis = cvCreateMat(basis.rows,basis.cols,CV_32FC1);
		//Basis = basis;

	fs.release();

	return;
}


Mat AIM::GetConvolution2D(Mat source, Mat kernel) {

	Mat dest;

	Point anchor(-1,-1);  // Anchor is at Filter center

    int borderMode = BORDER_CONSTANT;

    filter2D(source, dest, source.depth(), kernel, anchor, 1, borderMode);

    /*
    dest = dest.colRange((kernel.cols-1)/2-1, dest.cols - (kernel.cols-1)/2-1)
               .rowRange((kernel.rows-1)/2-1, dest.rows - (kernel.rows-1)/2-1);
    */

    return dest;
}


/* Input: InputImage
 * Output: Set of ConvolvedImages
 */
vector <Mat> AIM::GetGaborConvolutionBasis(Mat InputImage) {

	int NumColorChannels = InputImage.channels();
	int NumBasis = basis.rows;
	int p = sqrt(basis.cols/3);

	cout<<"num basis functions :"<<NumBasis<<" p is "<<p<<endl; 


	Mat basis_patch;

	vector <Mat> BasisImages(NumBasis*NumColorChannels);

	Mat InputImageRGBColor = Mat::zeros(InputImage.rows, InputImage.cols, InputImage.type());
	Mat InputImageRGB[NumColorChannels];

	cvtColor(InputImage,InputImageRGBColor,CV_BGR2RGB);

	split(InputImageRGBColor, InputImageRGB);


	for (int i = 0; i < NumBasis; i++)
	{

	/**************************************************
	 * STEP 1 : Convolution of Basis Kernel with Input Image
	 ***************************************************/
		for (int j = 0; j < NumColorChannels; j++)
		{
			Mat basis_row = basis.operator ()(Range(i,i+1),Range(j*p*p,(j+1)*p*p));

			basis_patch = basis_row.reshape(1,p);

			if (UseFixedPoint)
				BasisImages[i] = Mat::zeros(InputImage.rows,InputImage.cols,CV_16UC1);
			else
				BasisImages[i] = Mat::zeros(InputImage.rows,InputImage.cols,CV_32FC1);

			// Not adding R, G, B - This is where it differs from regular AIM
			BasisImages[i*NumColorChannels + j] = GetConvolution2D(InputImageRGB[j], basis_patch);

			if (Debug == true)
				cout << "Convolution took some time" << endl;
		}
	}

	return BasisImages;
}


/* Input: Set of Convolved Images, SizeofInfoMap
 * Output: InfoMap
 */
Mat AIM::GetInfoMapfromConvolution(vector <Mat> BasisImages, Size InfoMapSize) {

	Mat InfoMap = Mat::zeros(InfoMapSize.height, InfoMapSize.width, CV_32FC1);
	Mat ConstImage = Mat::ones(InfoMapSize.height, InfoMapSize.width, CV_32FC1)*0.000001;

	// How many pixels correspond to 1 degree visual angle
	double sigval = 8;

	// What size of window is needed to contain the above
	int ksize = 31; // In the MATLAB version it is 30x30 window. But OpenCV likes odd size kernels


	for (unsigned int i = 0; i < BasisImages.capacity(); i++)
	{
		/**************************************************
		 * STEP 2 : Get Density
		 ***************************************************/
		Mat DenMap = GetDensity(BasisImages[i]);

		/**************************************************
		 * STEP 3 : Convert to Log Likelihood
		 ***************************************************/
    	// Don't know why we do this step but we do it
       	add(InfoMap, ConstImage, InfoMap);

    	subtract(InfoMap, DenMap, InfoMap);
	}

	/**************************************************
	 * Post-Processing of Information Map
	 ***************************************************/
	double minVal, maxVal;

	minMaxLoc(InfoMap, &minVal, &maxVal, NULL, NULL);

	// Min values scaled to 0
	InfoMap = InfoMap - minVal;

	// Normalize
	InfoMap.convertTo(InfoMap,-1,1/(maxVal-minVal),0);

	Mat gker = getGaussianKernel(ksize,sigval, CV_32F);

	sepFilter2D(InfoMap, InfoMap, -1, gker, gker);   // separable convolution (speeds up execution)

	return InfoMap;
}


/* Input: InfoMap, DisplayThreshold
 * Output: ThreshMap
 */
Mat AIM::GetThreshMap (Mat InfoMap, double dispThresh) {

	Mat InfomapLine = InfoMap.clone();
	Mat InfomapVector = InfomapLine.reshape(1,InfoMap.rows * InfoMap.cols);

	double disp_rank = CalcPercentile(InfomapVector, dispThresh);

	Mat ThreshMap;
	compare(InfoMap,disp_rank,ThreshMap,CMP_GT);

	ThreshMap.convertTo(ThreshMap, CV_32FC1, 1/double(255), 0);

	return ThreshMap;
}


/* Inputs: InputImage, ThreshMap
 * Output: SaliencyMap
 */
Mat AIM::DoPixelMasking (Mat InputImage, Mat ThreshMap) {

	int NumColorChannels = InputImage.channels();
	std::vector <Mat> SaliencyMap (NumColorChannels);

	Mat InputImageRGBColor = Mat::zeros(InputImage.rows, InputImage.cols, InputImage.type());
	cvtColor(InputImage,InputImageRGBColor,CV_BGR2RGB);

	Mat InputImageRGB[NumColorChannels];
	split(InputImageRGBColor, InputImageRGB);

	for (int i = 0; i < NumColorChannels; i++)
	{
		SaliencyMap[i] = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	}

	multiply(InputImageRGB[0],ThreshMap,SaliencyMap[2]);  // R-R channel
	multiply(InputImageRGB[1],ThreshMap,SaliencyMap[1]);  // G-G channel
	multiply(InputImageRGB[2],ThreshMap,SaliencyMap[0]);  // B-B channel

	Mat SalienceMap;
	merge(SaliencyMap,SalienceMap);

	return SalienceMap;
}


/* Input: InputImage
 * Output: DensityMap
 */
Mat AIM::GetDensity(Mat InputImage) {

	int bins = 1000;

	double minVal,  maxVal;

	int numPixels = InputImage.rows * InputImage.cols;

	minMaxLoc(InputImage, &minVal, &maxVal, NULL, NULL);

	Mat LogMap;

	int hbins [] = {bins};
	int plane [] = {0}; // channel no 0
	float range[] = { 0, 1.001 };  // All pixels were normalized while converting to double bit precision, so range is 0 to 1 (1.001 because OpenCV likes exclusiveness in the upper range)
	const float *hranges[] = {range};
	Mat HistImage;

	//Check that we have a well formed input range (i.e. the max-min value should not be zero)
	if (maxVal == minVal)
		return Mat::zeros(InputImage.rows, InputImage.cols, InputImage.type());

	// Normalize
	InputImage = InputImage - minVal;
	InputImage.convertTo(InputImage, -1, 1/(maxVal-minVal),0);

	// Calculate the Histogram
	calcHist(&InputImage, 1,  // one image
				 plane, Mat(), // do no use mask
				 HistImage, 1, hbins,
				 hranges);

	if (Debug == true)
	{
		cout << "value = " << HistImage.at<double>(105) << endl;
		cout << "Histogram calculated" << endl;
	}

	Mat ProbMap;

	calcBackProject(&InputImage,1,0,
					HistImage,ProbMap,
					hranges,1,true);

	if (Debug == true)
		cout << "Back projection calculated" << endl;

	// Get Density Estimate between 0 and 1
	ProbMap.convertTo(ProbMap,-1,1/(double)numPixels,0);

	if (Debug == true)
		cout << "Division done" << endl;

	// Take Logs
	log(ProbMap, LogMap);

	if (Debug == true)
		cout << "Log calculated" << endl;

	return LogMap;
}

void* basisthreadfunc(void *targ)
{

	AIM aimobj; 
	unsigned int NumColorChannels = 3;
	unsigned int p = 21;  // filter size
	struct basis_thread_arg *arg;
	arg = (struct basis_thread_arg *)targ;
	int id = arg->tid;
	Mat filter1d = arg->bfilter;
	
	Mat iimage[NumColorChannels];
	Mat tempimage[NumColorChannels];

	//cout<<"inside thread"<<id<<endl;

	Mat BasisImageOutput;

	for (int i=0; i<NumColorChannels; i++)
	{
		iimage[i] = arg->input[i];
	}

	for (int j = 0; j < NumColorChannels; j++)
	{
		Mat basis_row = filter1d.colRange(j*p*p,(j+1)*p*p);

		Mat basis_patch = basis_row.reshape(1,p);

		tempimage[j] = aimobj.GetConvolution2D(iimage[j], basis_patch);

	}

	BasisImageOutput = tempimage[0] + tempimage[1] + tempimage[2];	
	Mat DenMap = aimobj.GetDensity(BasisImageOutput);

	/*if (id==2)
	{
		//cout<<"Number of rows and cols "<<BasisImageOutput.rows<<"  "<<BasisImageOutput.cols<<" "<<BasisImageOutput.channels()<<endl;
		namedWindow("img_display", CV_WINDOW_AUTOSIZE);
		imshow("img_display", DenMap);
		waitKey(0);
	} */


	arg->output = DenMap;

}		



// Added by nandhini ///
AIM::IST AIM::GetSaliencyMT(Mat InputImage) 
{

	

	// How many pixels correspond to 1 degree visual angle
	double sigval = 8;

	// What size of window is needed to contain the above
	int ksize = 31; // In the MATLAB version it is 30x30 window. But OpenCV likes odd size kernels

	int NumColorChannels = InputImage.channels();
	int p = sqrt(basis.cols/3);
	
	Mat basis_patch;
	Mat InputImageRGBColor = Mat::zeros(InputImage.rows, InputImage.cols, InputImage.type());
	Mat InputImageRGB[NumColorChannels];
	Mat OutputImageRGB[NumColorChannels];
	cvtColor(InputImage,InputImageRGBColor,CV_BGR2RGB);

	if (Debug == true)
		cout << "Color converted" << endl;

	split(InputImageRGBColor, InputImageRGB);

	int NumBasis = basis.rows;
	Mat DenMap[NumBasis];

	for (int p=0; p<NumBasis; p++)
	{
		DenMap[p] = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	}
	pthread_t basisthreads[NumBasis];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

	struct basis_thread_arg bt[NumBasis];

	struct timeval s_time, f_time;
	unsigned long exec_time = 0;
	gettimeofday(&s_time, 0);

	for (int i = 0; i < NumBasis; i++)
	{
	
		//cout<<"creating thread "<<i<<endl; 

		Mat basis_row = basis.row(i);

		
		bt[i].tid = i;
		bt[i].bfilter = basis_row;

		bt[i].output = DenMap[i];

		for (int j=0; j<NumColorChannels; j++)
		{
			bt[i].input[j] = InputImageRGB[j];
		}
		

		int ret = pthread_create(&basisthreads[i], &attr, &basisthreadfunc, (void *)&bt[i]);	
	
		if (ret)
		cout<<"ERROR in creating threads"<<endl;
		
	 }


	for (int k=0; k<NumBasis; k++)
	{
		pthread_join(basisthreads[k], NULL);
	}	


	// Copy outputs

	for (int q=0; q<NumBasis; q++)
		DenMap[q]=bt[q].output;


	// All threads have calculated the density maps now. Add all log likelihood maps. 
	Mat InfoMap = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	Mat ConstImage = Mat::ones(InputImage.rows, InputImage.cols, CV_32FC1)*0.000001;

       	add(InfoMap, ConstImage, InfoMap);

	 
		
	for (int k1=0; k1<NumBasis; k1++)
	{
		subtract(InfoMap, DenMap[k1], InfoMap);
	}		


	gettimeofday(&f_time, 0);

	exec_time = ((f_time.tv_sec - s_time.tv_sec) * 1000000) + (f_time.tv_usec - s_time.tv_usec);

	
	cout << "Total Density Estimation Time"<< exec_time << endl;





	//  *************************************************
	 // Post-Processing of Information Map
	//  **************************************************
	double minVal, maxVal;

	minMaxLoc(InfoMap, &minVal, &maxVal, NULL, NULL);

	// Min values scaled to 0
	InfoMap = InfoMap - minVal;

	// Normalize
	InfoMap.convertTo(InfoMap,-1,1/(maxVal-minVal),0);

	Mat gker = getGaussianKernel(ksize,sigval, CV_32F);

	sepFilter2D(InfoMap, InfoMap, -1, gker, gker);   // separable convolution (speeds up execution)

	  //  *************************************************
	 // Pixel Masking
	 //   ***************************************************
	std::vector <Mat> SaliencyMap (NumColorChannels);

	for (int i = 0; i < NumColorChannels; i++)
	{
		SaliencyMap[i] = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	}

	Mat InfomapLine = InfoMap.clone();
	Mat InfomapVector = InfomapLine.reshape(1,InfoMap.rows * InfoMap.cols);
	double disp_rank = CalcPercentile(InfomapVector, 0.8);

	Mat ThreshMap;
	compare(InfoMap,disp_rank,ThreshMap,CMP_GT);

	ThreshMap.convertTo(ThreshMap, CV_32FC1, 1/double(255), 0);

	multiply(InputImageRGB[0],ThreshMap,SaliencyMap[2]);  // R-R channel
	multiply(InputImageRGB[1],ThreshMap,SaliencyMap[1]);  // G-G channel
	multiply(InputImageRGB[2],ThreshMap,SaliencyMap[0]);  // B-B channel

	Mat SalienceMap;
	merge(SaliencyMap,SalienceMap);

	//   **************************************************
	 //  Return Values
	//  ***************************************************
	ist.InfoImage = InfoMap;
	ist.SaliencyImage = SalienceMap;
	ist.ThreshImage = ThreshMap;

	return ist;
}


/*int AIM::ZonebasedDensityEstimation(Mat InputImage)
{
	// Support zone based computation 

	
	return 0;
} 
*/


/* Inputs: InputImage, BasisFile
 * Output: Structure containing InfoMap, SalienceMap and ThreshMap
 */
AIM::IST AIM::GetSaliency(Mat InputImage) {

	// How many pixels correspond to 1 degree visual angle
	struct timeval s_time, f_time, s_time1, f_time1;
	unsigned long exec_time = 0;
	gettimeofday(&s_time, 0);
	double sigval = 8;

	// What size of window is needed to contain the above
	int ksize = 31; // In the MATLAB version it is 30x30 window. But OpenCV likes odd size kernels

	int NumColorChannels = InputImage.channels();
	int p = sqrt(basis.cols/3);
	
//	if (Debug == true)
		//ocut << "test in ST p = " << p << endl;

	Mat basis_patch;
	Mat InputImageRGBColor = Mat::zeros(InputImage.rows, InputImage.cols, InputImage.type());
	Mat InputImageRGB[NumColorChannels];
	Mat OutputImageRGB[NumColorChannels];
	Mat InfoMap = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	Mat ConstImage = Mat::ones(InputImage.rows, InputImage.cols, CV_32FC1)*0.000001;

	cvtColor(InputImage,InputImageRGBColor,CV_BGR2RGB);

	if (Debug == true)
		cout << "Color converted" << endl;

	split(InputImageRGBColor, InputImageRGB);

	int NumBasis = basis.rows;
	//cout <<"test2 "<<NumBasis<<endl;
	Mat BasisImage[NumBasis];

	unsigned long conv_time =0;
	for (int i = 0; i < NumBasis; i++)
	{

	
	// STEP 1 : Convolution of Basis Kernel with Input Image
	 	gettimeofday(&s_time1,0);
		for (int j = 0; j < NumColorChannels; j++)
		{
			Mat basis_row = basis.operator ()(Range(i,i+1),Range(j*p*p,(j+1)*p*p));

			basis_patch = basis_row.reshape(1,p);

			if (UseFixedPoint)
				BasisImage[i] = Mat::zeros(InputImage.rows,InputImage.cols,CV_16UC1);
			else
				BasisImage[i] = Mat::zeros(InputImage.rows,InputImage.cols,CV_32FC1);

			OutputImageRGB[j] = GetConvolution2D(InputImageRGB[j], basis_patch);

			if (Debug == true)
				cout << "Convolution took some time" << endl;

			BasisImage[i] = BasisImage[i] + OutputImageRGB[j];
		} 

		
		gettimeofday(&f_time1,0);
		exec_time = ((f_time1.tv_sec - s_time1.tv_sec) * 1000000) + (f_time1.tv_usec - s_time1.tv_usec);
		conv_time+=exec_time;	
		cout << "Total Conv Time"<< conv_time << endl;

	// **************************************************
   	 // STEP 2 : Get Density
     // ***************************************************
		Mat DenMap = GetDensity(BasisImage[i]);

	//  **************************************************
   //    STEP 3 : Convert to Log Likelihood
   //  ***************************************************
    	// Don't know why we do this step but we do it
       	add(InfoMap, ConstImage, InfoMap);

    	subtract(InfoMap, DenMap, InfoMap);

    	if (Debug == true)
    		cout << "Step 2 and 3 done" << endl;
	}
	gettimeofday(&f_time, 0);

	exec_time = ((f_time.tv_sec - s_time.tv_sec) * 1000000) + (f_time.tv_usec - s_time.tv_usec);
	cout << "Total Density Estimation Time"<< exec_time << endl;


	//  *************************************************
	 // Post-Processing of Information Map
	//  **************************************************

	gettimeofday(&s_time,0);
	double minVal, maxVal;

	minMaxLoc(InfoMap, &minVal, &maxVal, NULL, NULL);

	// Min values scaled to 0
	InfoMap = InfoMap - minVal;

	// Normalize
	InfoMap.convertTo(InfoMap,-1,1/(maxVal-minVal),0);

	Mat gker = getGaussianKernel(ksize,sigval, CV_32F);

	sepFilter2D(InfoMap, InfoMap, -1, gker, gker);   // separable convolution (speeds up execution)

	  //  *************************************************
	 // Pixel Masking
	 //   ***************************************************
	std::vector <Mat> SaliencyMap (NumColorChannels);

	for (int i = 0; i < NumColorChannels; i++)
	{
		SaliencyMap[i] = Mat::zeros(InputImage.rows, InputImage.cols, CV_32FC1);
	}

	Mat InfomapLine = InfoMap.clone();
	Mat InfomapVector = InfomapLine.reshape(1,InfoMap.rows * InfoMap.cols);

	double disp_rank = CalcPercentile(InfomapVector, 0.8);

	Mat ThreshMap;
	compare(InfoMap,disp_rank,ThreshMap,CMP_GT);

	ThreshMap.convertTo(ThreshMap, CV_32FC1, 1/double(255), 0);

	multiply(InputImageRGB[0],ThreshMap,SaliencyMap[2]);  // R-R channel
	multiply(InputImageRGB[1],ThreshMap,SaliencyMap[1]);  // G-G channel
	multiply(InputImageRGB[2],ThreshMap,SaliencyMap[0]);  // B-B channel

	Mat SalienceMap;
	merge(SaliencyMap,SalienceMap);

	//   **************************************************
	 //  Return Values
	//  ***************************************************
	ist.InfoImage = InfoMap;
	ist.SaliencyImage = SalienceMap;
	ist.ThreshImage = ThreshMap;
	gettimeofday(&f_time, 0);

	exec_time = ((f_time.tv_sec - s_time.tv_sec) * 1000000) + (f_time.tv_usec - s_time.tv_usec);
	cout << "Post processing time"<< exec_time << endl;


	return ist;
}


/* Inputs: sequence (1D vector), percentile (0 <= percentile <= 1)
 * Output: Percentile rank
 * Example Use : double n = Percentile (infomap, 0.98);
 * Refer : http://en.wikipedia.org/wiki/Percentile
 */
double AIM::CalcPercentile(Mat sequence, double percentile) {

	// Get the length of infomap
    double N = sequence.rows;

	// Sort the sequence
	cv::sort(sequence, sequence, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

    // Calculate Rank (To check for extremes i.e. 0 and 1)
    double n1 = (N-1)*percentile + 1;

    // Calculate Rank (For linear interpolation)
    double n2 = (N*percentile) + 0.5;

    if (n1 == 1) return sequence.at<float>(0);

    else if (n1 == N)
    	{
    	return sequence.at<float>(N - 1);
    	}
    else  // Linear Interpolation
    {
         int k = (int)n1;  // round down to nearest integer

         double d = n2 - k;

         return sequence.at<float>(k - 1) + d * (sequence.at<float>(k) - sequence.at<float>(k - 1));
    }
}


