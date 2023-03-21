/*  Tracking Class 

	Written by Juhyun Lee
	Editted by Hyungwon Choi
	2013.07.26
	Track the object in the box 
*/

#define WINSIZE_HEIGHT	12
#define WINSIZE_WIDTH	15
#define LIMIT_FRAME		1000
#define PI				3.14159265

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <time.h>
#include <math.h>
#include <limits>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv::gpu;
using namespace cv;

class Tracker
{
	public:

	struct afnv_obj{
		cv::Mat R;
		cv::Mat afnv;
		cv::Mat size;
	};

	struct whiten_out{
		cv::Mat output;
		cv::Mat outmean;
		cv::Mat outstd;
	};

	struct crop{
		cv::Mat cropimg;
		cv::Mat cropnorm;
		cv::Mat cropmean;
		cv::Mat cropstd;
	};

	struct uv{
		cv::Mat vx;
		cv::Mat vy;
	};

	struct gly{
		cv::Mat crop;
		cv::Mat inrange;
	};

	struct normal{
		cv::Mat data;
		cv::Mat norms;
	};
    

	Mat sz_T;
    Mat init_pos;
    int numFrame;
    Mat firstframe;


    int prod;
    Mat lambda;
    Mat initlambda3;
    double angle_threshold;
    float Lip;
    int Maxit;
    int nT;
    Mat rel_std_afnv;
    int n_sample;
    int occlusionNf;
    float alpha;
    int numLevels;
    double pyrScale;
    bool fastPyramids;
    int winSize;
    int numIters;
    int polyN;
    double polySigma;
    bool resize_img;
    double rfactor;


    Mat firstFrame_gray;
    struct crop Tmp1;
    Mat norms;
    Mat A;
    Mat II;
    struct afnv_obj afnv_objThis;
    Mat map_aff;
    Mat aff_samples_n;
    Mat aff_samples;
    Mat TempReal;
    Mat Dict;
    Mat Temp_1;
    Mat Temp_1inv;
    Mat fixT;

 	Mat trackrect;
	Mat trackrect0;
   

	void Init(Mat frame, int x1, int y1, int x2, int y2);
	struct whiten_out whitening(Mat input);
	struct afnv_obj corner2affine(Mat corners, Mat tsize);
	struct crop corner2image(Mat img_gray, Mat initp, Mat tsize);
	struct crop initTemplate(Mat tsize, int numT, Mat img, Mat cpt);
	void drawMotionField(Mat &imgU, Mat &imgV, Mat &imgMotion,int xSpace, int ySpace, float cutoff, float multiplier, CvScalar color);
	Mat draw_sample(Mat mean_afnv, Mat std_afnv);
	struct gly crop_candidates(Mat img_frame, Mat curr_sample, Mat template_size);
	struct normal normalizeTemplate(Mat normal_in);
	Mat APGLASSO(Mat samples, Mat dict, Mat lambda, float lip, int maxiter, int nT);
	Mat softthres(Mat x, float lambda);
	Mat resample(Mat curr_sample, Mat prob, Mat afnv);
	double images_angle(Mat l1, Mat l2);
	Mat aff2image(Mat aff_maps, Mat T_sz);
	Mat drawAffine(Mat afnv, Mat tsize);
	Mat startTracking(Mat frame1, Mat &frame0, Mat sz_T, Mat &TempReal, Mat &Dict, Mat &Temp_1, Mat &Temp_1inv, Mat &A, Mat &map_aff, Mat &aff_samples, struct crop &Tmp1, Mat &norms, Mat fixT, int numframes, Mat &trackrect0);
	
	struct uv farneback_flow(Mat frame0_rgb_, Mat frame1_rgb_, int numLevels, double pyrScale, bool fastPyramids, int winSize, int numIters,	int polyN, double polySigma, bool resize_img, double rfactor);
	Mat findBlobs(Mat binary);
	Mat Our_round(Mat x);	
	Mat imgaffine(Mat img, Mat map_afnv, Mat tsize, int *L);
};

void Tracker::Init(Mat frame, int x1, int y1, int x2, int y2)
{

    frame.copyTo(firstframe);
	if(firstframe.channels()==3)
	{
		cvtColor(firstframe, firstFrame_gray, CV_RGB2GRAY);
	}
	else{
		firstframe.copyTo(firstFrame_gray);
	}    
	sz_T.create(1,2, CV_32FC1);
    init_pos.create(2,3, CV_32FC1);
    numFrame = 1;

	init_pos.at<float>(0,0) = (float)y1;
	init_pos.at<float>(1,0) = (float)x1;
	init_pos.at<float>(0,1) = (float)y2;
	init_pos.at<float>(1,1) = (float)x1;
	init_pos.at<float>(0,2) = (float)y1;
	init_pos.at<float>(1,2) = (float)x2;
	sz_T.at<float>(0,0) = (float)WINSIZE_HEIGHT;
	sz_T.at<float>(0,1) = (float)WINSIZE_WIDTH;

    prod = WINSIZE_HEIGHT*WINSIZE_WIDTH;
    lambda = Mat::zeros(1,3, CV_32FC1);
	initlambda3 = Mat::zeros(1,1, CV_32FC1);
	angle_threshold = 40.0;
    Lip  = 8.0;
    Maxit = 5;
    nT = 10;
    rel_std_afnv = Mat::zeros(1,6, CV_32FC1);
   	occlusionNf = 0;
    alpha = 50.0;
    numLevels = 4;
    pyrScale = 0.5;
    fastPyramids = true;
    winSize = 11;
    numIters = 10;
    polyN = 7;
    polySigma = 2.4;
    resize_img = false;
    rfactor = 1.0;


	lambda.at<float>(0,0) = 0.2;
    lambda.at<float>(0,1) = 0.001;
    lambda.at<float>(0,2) = 10.0;
	initlambda3.at<float>(0,0) = lambda.at<float>(0,2);    
	rel_std_afnv.at<float>(0,0) = 0.03;
    rel_std_afnv.at<float>(0,1) = 0.0005;
    rel_std_afnv.at<float>(0,2) = 0.0005;
    rel_std_afnv.at<float>(0,3) = 0.03;
    rel_std_afnv.at<float>(0,4) = 1.0;
    rel_std_afnv.at<float>(0,5) = 1.0;
    n_sample = 600;

    Tmp1 = {Mat::zeros(prod, nT, CV_32FC1), Mat::zeros(1, nT, CV_32FC1), Mat::zeros(1, nT, CV_32FC1), Mat::zeros(1, nT, CV_32FC1)};
    Tmp1 = initTemplate(sz_T, nT, firstFrame_gray, init_pos);

    norms = Mat::zeros(Tmp1.cropnorm.rows, Tmp1.cropnorm.cols, CV_32FC1);
    multiply(Tmp1.cropnorm, Tmp1.cropstd, norms);

    II = Mat::eye(prod, prod, CV_32FC1);
    hconcat(Tmp1.cropimg, II, A);
    afnv_objThis = {Mat::zeros(3,3, CV_32FC1), Mat::zeros(1,6, CV_32FC1), Mat::zeros(1,2, CV_32FC1)};
    afnv_objThis = corner2affine(init_pos, sz_T);

    map_aff = afnv_objThis.afnv;
    aff_samples_n = Mat::ones(n_sample, 1, CV_32FC1);
    aff_samples = aff_samples_n*map_aff;

    fixT = Tmp1.cropimg.col(0)/nT;
    hconcat(A, fixT, TempReal);
    Dict = TempReal.t()*TempReal;
    hconcat(Tmp1.cropimg, fixT, Temp_1);

    SVD svd(Temp_1);
    for(int ii=0; ii<svd.w.rows;ii++){
        if (svd.w.at<float>(ii,0) < 0.001){
            svd.w.at<float>(ii,0) = 0.0;
        }
    }

    Temp_1inv = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();
    Temp_1 = Temp_1*Temp_1inv;

	trackrect = Mat::zeros(2,4, CV_32FC1);
	
/*
    vy = Mat::zeros(firstFrame_gray.rows, firstFrame_gray.cols, CV_32FC1);
    vx = Mat::zeros(firstFrame_gray.rows, firstFrame_gray.cols, CV_32FC1);
    uv1 = {Mat::zeros(firstFrame_gray.rows, firstFrame_gray.cols, CV_32FC1), Mat::zeros(firstFrame_gray.rows, frame1.cols, CV_32FC1)};
    sc = Mat::zeros(1,1,CV_32FC1);
    sc2 = Mat::zeros(1,1, CV_32FC1);
    sc_temp = Mat::ones(1, 6, CV_32FC1);
    Mat std_aff;
    gly1 = {Mat::zeros(prod, n_sample, CV_32FC1), Mat::zeros(n_sample, 1, CV_32FC1)};
    Y_whiten = {Mat::zeros(gly1.crop.rows, gly1.crop.cols, CV_32FC1), Mat::zeros(1, gly1.crop.cols, CV_32FC1), Mat::zeros(1, gly1.crop.cols, CV_32FC1)};
    Y_normal = {Mat::zeros(Y_whiten.output.rows, Y_whiten.output.cols, CV_32FC1), Mat::zeros(1, Y_whiten.output.cols, CV_32FC1)};
    sumD_s = Mat::zeros(1,1, CV_32FC1);
    id_max = 0;
    maxarea = 0;
    normal2 = {Mat(Tmp1.cropimg.rows, Tmp1.cropimg.cols, CV_32FC1), Mat(1, Tmp1.cropimg.cols, CV_32FC1)};

    quit = NULL;
*/
}


Tracker::whiten_out Tracker::whitening(Mat input)
{
    /* how to use whitening*/
    //Mat input(3,3,CV_32FC1, Scalar(10));
    //struct whiten_out whiten_out1 = {Mat::zeros(input.rows,input.cols,CV_32FC1), Mat::zeros(1,input.cols,CV_32FC1), Mat::zeros(1,input.cols,CV_32FC1)};
    //whiten_out1 = whitening(input);

    int MN = input.rows;
    Mat inputF;
    input.copyTo(inputF);
    //input.convertTo(inputF, CV_32FC1);
    Mat mean_input(1,1, CV_32FC1);
    Mat std_input(1,1, CV_32FC1);
    Mat a(1, input.cols, CV_32FC1);
    Mat b(1, input.cols, CV_32FC1);
    Mat output(input.rows, input.cols, CV_32FC1);

    for(int i=0; i<inputF.cols;i++)
    {
        meanStdDev(inputF.col(i), mean_input, std_input,noArray());
        mean_input.copyTo(a.col(i));
        std_input.copyTo(b.col(i));
    }
    //b= b+ pow(10.0,-14.0);
    b= b+0.00000000000001;
    Mat cal = Mat::ones(MN, 1, CV_32FC1);

    divide(inputF-cal*a,cal*b, output);

    whiten_out whiten_out1;
    output.copyTo(whiten_out1.output);
    a.copyTo(whiten_out1.outmean);
    b.copyTo(whiten_out1.outstd);

    return whiten_out1;
}

Tracker::afnv_obj Tracker::corner2affine(Mat corners, Mat tsize)
{
    /* how to use corner2affine*/
    //  struct afnv_obj afnv_objThis = {Mat::zeros(3,3,CV_32FC1), Mat::zeros(1,6,CV_32FC1), Mat::zeros(1,2,CV_32FC1)};
    //afnv_objThis = corner2affine(init_pos, sz_T);

    Mat inp = Mat::ones(3,3, CV_32FC1);
    (corners.row(0)).copyTo(inp.row(0));
    (corners.row(1)).copyTo(inp.row(1));


    Mat outp = Mat::ones(3,3, CV_32FC1);
    outp.at<float>(0,1) = tsize.at<float>(0,0);
    outp.at<float>(1,2) = tsize.at<float>(0,1);

    afnv_obj afnv_obj1;

    Mat R = inp*(outp.inv());
    afnv_obj1.R = R;
    afnv_obj1.afnv = Mat::zeros(1,6, CV_32FC1);

    afnv_obj1.afnv.at<float>(0,0) = R.at<float>(0,0);
    afnv_obj1.afnv.at<float>(0,1) = R.at<float>(0,1);
    afnv_obj1.afnv.at<float>(0,2) = R.at<float>(1,0);
    afnv_obj1.afnv.at<float>(0,3) = R.at<float>(1,1);
    afnv_obj1.afnv.at<float>(0,4) = R.at<float>(0,2);
    afnv_obj1.afnv.at<float>(0,5) = R.at<float>(1,2);

    afnv_obj1.size = tsize;

    return afnv_obj1;

}


Tracker::crop Tracker::corner2image(Mat img_gray, Mat initp, Mat tsize)
{
    /*how to use corner2image*/
    //struct crop crop1 = {Mat::zeros((int)prod, 1, CV_32FC1),  Mat::zeros(1,1, CV_32FC1), Mat::zeros(1,1, CV_32FC1), Mat::zeros(1,1, CV_32FC1)};
    //crop1 = corner2image(firstFrame_gray, init_pos, sz_T);
    struct afnv_obj afnv_objThis = {Mat::zeros(3,3,CV_32FC1), Mat::zeros(1,6,CV_32FC1), Mat::zeros(1,2,CV_32FC1)};
    afnv_objThis = corner2affine(initp, tsize);
    Mat map_afnv = afnv_objThis.afnv;

    Mat img_map;
    Mat img_mapF;
    Mat img_map_shape;
    int dummy;
    int *L;
    L = &dummy;
    *L=0;

    int prod =((int) tsize.at<float>(0,0))*((int)tsize.at<float>(0,1));
    img_map = imgaffine(img_gray, map_afnv, tsize, L);
    img_map.convertTo(img_mapF, CV_32FC1);
    Mat img_mapFT;
    img_mapFT = img_mapF.t();
    img_map_shape=img_mapFT.reshape(0, prod);
    struct whiten_out whiten_outThis = {Mat::zeros(prod,1,CV_32FC1), Mat::zeros(1,1,CV_32FC1), Mat::zeros(1,1,CV_32FC1)};

    whiten_outThis = whitening(img_map_shape);
    float crop_norm = (float)norm(whiten_outThis.output, NORM_L2);
    whiten_outThis.output = whiten_outThis.output/crop_norm;

    crop cropThis;
    whiten_outThis.output.copyTo(cropThis.cropimg);
    whiten_outThis.outmean.copyTo(cropThis.cropmean);
    whiten_outThis.outstd.copyTo(cropThis.cropstd);
    cropThis.cropnorm = Mat::zeros(1,img_map_shape.cols, CV_32FC1);
    for(int i=0; i<img_map_shape.cols;i++)
    {
        cropThis.cropnorm.at<float>(0,i) = crop_norm;
    }

    return cropThis;

}

Tracker::crop Tracker::initTemplate(Mat tsize, int numT, Mat img, Mat cpt)
{
    
	float prod = tsize.at<float>(0,0)*tsize.at<float>(0,1);
    struct crop crop1 = {Mat::zeros((int)prod, 1, CV_32FC1), Mat::zeros(1,1, CV_32FC1), Mat::zeros(1,1, CV_32FC1), Mat::zeros(1,1, CV_32FC1)};
    struct crop Tmp = {Mat::zeros((int)prod, numT, CV_32FC1), Mat::zeros(1,numT, CV_32FC1), Mat::zeros(1,numT, CV_32FC1), Mat::zeros(1,numT, CV_32FC1)};
    Mat p = Mat::zeros(cpt.rows, cpt.cols, CV_32FC1);


    Mat forrnd = Mat::zeros(cpt.rows, cpt.cols, CV_32FC1);
    Mat rndmean = Mat::zeros(1,1, CV_32FC1);
    Mat rndstd(1,1, CV_32FC1, Scalar(0.6));


    Tmp.cropimg = Mat::zeros((int)prod, numT, CV_32FC1);
    Mat cropnorm = Mat::zeros(1,numT, CV_32FC1);
    Mat cropmean = Mat::zeros(1,numT, CV_32FC1);
    Mat cropstd = Mat::zeros(1,numT, CV_32FC1);



    for(int n=0;n<numT;n++)
    {

        if(n==0)
        { cpt.copyTo(p);
        }
        else
        {   randn(forrnd, rndmean, rndstd);
        p = cpt+forrnd;

        }

        crop1= corner2image(img, p, tsize);
        crop1.cropimg.copyTo(Tmp.cropimg.col(n));
        Tmp.cropnorm.at<float>(0,n) = crop1.cropnorm.at<float>(0,0);
        Tmp.cropmean.at<float>(0,n) = crop1.cropmean.at<float>(0,0);
        Tmp.cropstd.at<float>(0,n) = crop1.cropstd.at<float>(0,0);

    }

    return Tmp;
}

void Tracker::drawMotionField(Mat &imgU, Mat &imgV, Mat &imgMotion,
    int xSpace, int ySpace, float cutoff, float multiplier, CvScalar color)
{
    int x = 0, y = 0;
    float *ptri;
    float deltaX = 0.0, deltaY = 0.0, angle = 0.0, hyp = 0.0;
    Point p0, p1;

    for( y = ySpace; y < imgU.rows; y += ySpace )
    {
        for(x = xSpace; x < imgU.cols; x += xSpace )
        {
            p0.x = x;
            p0.y = y;

            ptri = imgU.ptr<float>(y);
            deltaX = ptri[x];

            ptri = imgV.ptr<float>(y);
            deltaY = ptri[x];

            angle = atan2(deltaY, deltaX);
            hyp = sqrt(deltaX*deltaX + deltaY*deltaY);

            if(hyp > cutoff)
            {
                p1.x = p0.x + cvRound(multiplier*hyp*cos(angle));
                p1.y = p0.y + cvRound(multiplier*hyp*sin(angle));

                cv::line(imgMotion,p0,p1,color,1,CV_AA,0);

                p0.x = p1.x + cvRound(2*cos(angle-M_PI + M_PI/4));
                p0.y = p1.y + cvRound(2*sin(angle-M_PI + M_PI/4));
                cv::line( imgMotion, p0, p1, color,1, CV_AA, 0);

                p0.x = p1.x + cvRound(2*cos(angle-M_PI - M_PI/4));
                p0.y = p1.y + cvRound(2*sin(angle-M_PI - M_PI/4));
                cv::line( imgMotion, p0, p1, color,1, CV_AA, 0);
            }
        }
    }
}


Mat Tracker::draw_sample(Mat mean_afnv, Mat std_afnv)
{
    int nsamples = mean_afnv.rows;
    int MV_LEN =6;
    log(mean_afnv.col(0), mean_afnv.col(0));
    log(mean_afnv.col(3), mean_afnv.col(3));

    Mat outs = Mat::zeros(nsamples, MV_LEN, CV_32FC1);
    Mat forrnd = Mat::zeros(nsamples, MV_LEN, CV_32FC1);
    Mat rndmean = Mat::zeros(1,1, CV_32FC1);
    Mat rndstd = Mat::ones(1,1, CV_32FC1);
    randn(forrnd, rndmean, rndstd);
    Mat std_afnv_diag = Mat::diag(std_afnv);
    outs = forrnd*std_afnv_diag +mean_afnv;

    exp(outs.col(0), outs.col(0));
    exp(outs.col(3), outs.col(3));

    return outs;
}


Tracker::gly Tracker::crop_candidates(Mat img_frame, Mat curr_sample, Mat template_size)
{
    Mat img_frame_now;
    img_frame.convertTo(img_frame_now, CV_32FC1);

    int nsamples = curr_sample.rows;
    int c = template_size.at<float>(0,0)*template_size.at<float>(0,1);

    Mat curr_afnv = Mat::zeros(1, curr_sample.cols, CV_32FC1);
    Mat img_cut,img_cut_sh, img_cut_T;
    int dummy;
    int *L;
    L = &dummy;
    *L=0;


    struct gly gly1 = {Mat::zeros(c, nsamples, CV_32FC1), Mat::zeros(nsamples, 1, CV_32FC1)};

    for(int n =0; n<nsamples; n++)
    {
        curr_sample.row(n).copyTo(curr_afnv);
        img_cut = imgaffine(img_frame, curr_afnv, template_size, L);
        img_cut_T = img_cut.t();
        img_cut_sh = img_cut_T.reshape(1,c);
        gly1.inrange.at<float>(n,0) = *L;
        img_cut_sh.copyTo(gly1.crop.col(n));
    }

    divide(gly1.crop, 255.0, gly1.crop);


    return gly1;

}


Tracker::normal Tracker::normalizeTemplate(Mat normal_in)
{
    struct normal normal1 = {Mat(normal_in.rows, normal_in.cols, CV_32FC1), Mat(1,normal_in.cols, CV_32FC1)};
    Mat m_in = Mat::zeros(normal_in.rows, normal_in.cols, CV_32FC1);
    multiply(normal_in, normal_in, m_in);


    for(int i=0; i<m_in.cols;i++)
    {
        normal1.norms.col(i) = sum(m_in.col(i));
    }

    sqrt(normal1.norms, normal1.norms);

    //normal1.norms = normal1.norms+pow(10.0, -14.0);
    normal1.norms = normal1.norms+0.00000000000001;
    Mat dum = Mat::ones(normal_in.rows, 1, CV_32FC1);
    Mat dum2  = dum*normal1.norms;

    divide(normal_in, dum2, normal1.data);

    return normal1;
}

Mat Tracker::APGLASSO(Mat samples, Mat dict, Mat lambda, float lip, int maxiter, int nT)
{
    int coldim = dict.rows;
    Mat xPrev = Mat::zeros(coldim,1, CV_32FC1);
    Mat x = Mat::zeros(coldim, 1, CV_32FC1);
    float tPrev =1.0;
    float t=1.0;
    Mat temp_lambda = Mat::zeros(coldim,1 , CV_32FC1);
    float tem_t;
    Mat tem_y, tem_y2;
    Mat m1 = Mat::zeros(1,1, CV_32FC1);
    Mat softx;

    for (int n =0; n<nT; n++)
    {
        temp_lambda.at<float>(n,0) = lambda.at<float>(0,0);
    }
    temp_lambda.at<float>(coldim-1,0) = lambda.at<float>(0,0);

    for(int iter=1;iter <=maxiter; iter++)
    {
        tem_t = (tPrev -1.0)/t;
        tem_y  = (1.0+tem_t)*x -tem_t*xPrev;
        temp_lambda(Range(nT,coldim-1), Range::all()) = lambda.at<float>(0,2)*tem_y(Range(nT,coldim-1), Range::all());
        divide(dict*tem_y -samples+temp_lambda, lip, tem_y2);
        tem_y = tem_y -tem_y2;

        x.copyTo(xPrev);

        max(tem_y(Range(0,nT), Range::all()), 0.0, x(Range(0,nT), Range::all()));
        m1.at<float>(0,0) = tem_y.at<float>(coldim-1,0);
        max(m1, 0.0, m1);
        x.at<float>(coldim-1,0) = m1.at<float>(0,0);


        softx = softthres(tem_y(Range(nT, coldim-1), Range::all()), lambda.at<float>(0,1)/lip);
        softx.copyTo(x(Range(nT, coldim-1), Range::all()));

        tPrev =t;

        t=(float)pow((double)t, 2.0);
        t = (1.0+sqrt(1.0+4.0*t))/2.0;

    }

    return x;
}

Mat Tracker::softthres(Mat x, float lambda)
{
    Mat y, y1, y2;
    max(x-lambda, 0.0, y1);
    max((-1.0*x)-lambda, 0.0, y2);
    y = y1-y2;

    return y;
}


Mat Tracker::resample(Mat curr_sample, Mat prob, Mat afnv)
{
    int nsamples = curr_sample.rows;
    Mat probsum = Mat::zeros(1,1, CV_32FC1);
    probsum = sum(prob);
    Mat map_afnv;
    Mat map_afnv_init;
    Mat count;
    int order;
    bool outerloop = true;

    if (probsum.at<float>(0,0) ==0)
    {
        map_afnv = Mat::ones(nsamples , 1, CV_32FC1);
        map_afnv = map_afnv*afnv;
        count = Mat::zeros(prob.rows, prob.cols, CV_32FC1);
    }
    else
    {
        order =0;
        divide(prob, probsum.at<float>(0,0), prob);
        count = Our_round( nsamples*prob);
        map_afnv_init = Mat::ones(nsamples , 1, CV_32FC1);
        map_afnv = map_afnv_init*afnv;


        for (int i=0; i< nsamples; i++)
        {
            for(int j=0; j < (int)count.at<float>(i,0); j++)
            {

                curr_sample.row(i).copyTo(map_afnv.row(order));
                order++;

                if (order >= nsamples){
                    outerloop = false;
                    break;

                }
            }
            if(outerloop==false)
                break;

        }

    }

    return map_afnv;

}

double Tracker::images_angle(Mat l1, Mat l2)
{
    Mat l1v, l2v;
    Mat l1vn = Mat::zeros(1,1, CV_32FC1);
    Mat l2vn = Mat::zeros(1,1, CV_32FC1);
    Mat l1nor = Mat::zeros(1,1, CV_32FC1);
    Mat l2nor= Mat::zeros(1,1, CV_32FC1);
    double a;
    Mat mull;
    pow(l1, 2.0, l1v);
    pow(l2, 2.0, l2v);

    l1vn = sum(l1v);
    l2vn = sum(l2v);

    sqrt(l1vn, l1vn);
    sqrt(l2vn, l2vn);

    divide(l1, l1vn.at<float>(0,0), l1nor);
    divide(l2, l2vn.at<float>(0,0), l2nor);

    mull = l1nor.t()*l2nor;
    Mat mull2 = Mat::ones(1,1, CV_64FC1);
    mull.convertTo(mull2, CV_64FC1);
    a =  acos(mull2.at<double>(0,0)) *180.0 /PI;

    return a;
}


Mat Tracker::aff2image(Mat aff_maps, Mat T_sz)
{
    float r = T_sz.at<float>(0,0);
    float c = T_sz.at<float>(0,1);
    int n = aff_maps.cols;
    Mat boxes = Mat::zeros(8,n, CV_32FC1);
    Mat aff, R, P, Q, Qt, Q_sh;
    R = Mat::zeros(2,3, CV_32FC1);
    P = Mat::ones(3,4, CV_32FC1);
    P.at<float>(0,1) = r;
    P.at<float>(0,3) = r;
    P.at<float>(1,2) = c;
    P.at<float>(1,3) = c;


    for (int ii=0;ii< n; ii++)
    {
        aff_maps.col(ii).copyTo(aff);
        R.at<float>(0,0) = aff.at<float>(0,0);
        R.at<float>(0,1) = aff.at<float>(1,0);
        R.at<float>(1,0) = aff.at<float>(2,0);
        R.at<float>(1,1) = aff.at<float>(3,0);
        R.at<float>(0,2) = aff.at<float>(4,0);
        R.at<float>(1,2) = aff.at<float>(5,0);
        Q = R*P;

        Qt = Q.t();
        Q_sh = Qt.reshape(0,8);
        Q_sh.copyTo(boxes.col(ii));
    }
    return boxes;
}

Mat Tracker::drawAffine(Mat afnv, Mat tsize)
{
    Mat boxes, rect, rectt, rect_sh;
    boxes = aff2image(afnv.t(), tsize);
    rect = Our_round(boxes);
    rect_sh = rect.reshape(0,4);
    rectt = rect_sh.t();

    return rectt;
}


Mat Tracker::startTracking(Mat frame1, Mat &frame0, Mat sz_T, Mat &TempReal, Mat &Dict, Mat &Temp_1, Mat &Temp_1inv, Mat &A, Mat &map_aff, Mat &aff_samples, struct crop &Tmp1, Mat &norms, Mat fixT, int numframes, Mat &trackrect0)
{
	Mat frame1_gray;
	if(frame1.channels()==3){
		cvtColor(frame1, frame1_gray, CV_RGB2GRAY);
	}
	else{
		frame1.copyTo(frame1_gray);
	}

	Mat vy = Mat::zeros(frame1_gray.rows,frame1_gray.cols, CV_32FC1);
	Mat vx = Mat::zeros(frame1_gray.rows, frame1_gray.cols, CV_32FC1);
	struct uv uv1 = {Mat::zeros(frame1_gray.rows, frame1_gray.cols, CV_32FC1), Mat::zeros(frame1_gray.rows,  frame1.cols, CV_32FC1)};

	Mat vx_avg = Mat::zeros(1,1,CV_32FC1);
	Mat vy_avg = Mat::zeros(1,1, CV_32FC1);
	Mat sc, sc1;
	sc = Mat::zeros(1,1, CV_32FC1);
	Mat sc_temp = Mat::ones(1,6, CV_32FC1);
	Mat std_aff;
	struct gly gly1 = {Mat::zeros(prod, n_sample, CV_32FC1), Mat::zeros(n_sample, 1, CV_32FC1)};
	struct whiten_out Y_whiten = {Mat::zeros(gly1.crop.rows,gly1.crop.cols,CV_32FC1), Mat::zeros(1,gly1.crop.cols,CV_32FC1), Mat::zeros(1,gly1.crop.cols,CV_32FC1)};
	struct normal Y_normal = {Mat::zeros(Y_whiten.output.rows, Y_whiten.output.cols, CV_32FC1), Mat::zeros(1, Y_whiten.output.cols, CV_32FC1)};
	float eta_max;
	Mat q, q_sort, q_idx;
	Mat forabs, absum_Y;
	Mat prob;
	int n;
	float tau;
	Mat coeff, lassoinput, acont, coeffcon, D_s;
	Mat sumD_s = Mat::zeros(1,1, CV_32FC1);
	Mat TempRealt;
	Mat muldum, sumcoeff;
	int id_max=0; Mat c_max, a_max;
	Point indA;
	double min_angle;
	float level;
	Mat trivial_coef, se, trivial_coef_bi, trivial_coef_se, trivial_coeft, trivial_coef_sh;
	Point indw;
	Mat areas, areasize;
	double maxarea=0; 
	Mat areathres; 
	float areath;
	struct normal normal2 = {Mat(Tmp1.cropimg.rows, Tmp1.cropimg.cols, CV_32FC1), Mat(1,Tmp1.cropimg.cols, CV_32FC1)}; 
	Mat trackrect; 

	if (numframes>1){

	uv1 = farneback_flow(frame0, frame1, numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, resize_img, rfactor);

	vx_avg = mean(uv1.vx(Range(((trackrect0.at<float>(0,0) > 1.0 ? trackrect0.at<float>(0,0):1.0) <frame1.rows ? trackrect0.at<float>(0,0): frame1.rows),((trackrect0.at<float>(0,3) > 1.0 ? trackrect0.at<float>(0,3):1.0) <frame1.rows ? trackrect0.at<float>(0,3): frame1.rows)),Range(((trackrect0.at<float>(1,0) > 1.0 ? trackrect0.at<float>(1,0):1.0) <frame1.cols ? trackrect0.at<float>(1,0): frame1.cols),((trackrect0.at<float>(1,3) > 1.0 ? trackrect0.at<float>(1,3):1.0) <frame1.cols ? trackrect0.at<float>(1,3): frame1.cols))));

	vy_avg = mean(uv1.vy(Range(((trackrect0.at<float>(0,0) > 1.0 ? trackrect0.at<float>(0,0):1.0) <frame1.rows ? trackrect0.at<float>(0,0): frame1.rows),((trackrect0.at<float>(0,3) > 1.0 ? trackrect0.at<float>(0,3):1.0) <frame1.rows ? trackrect0.at<float>(0,3): frame1.rows)),Range(((trackrect0.at<float>(1,0) > 1.0 ? trackrect0.at<float>(1,0):1.0) <frame1.cols ? trackrect0.at<float>(1,0): frame1.cols),((trackrect0.at<float>(1,3) > 1.0 ? trackrect0.at<float>(1,3):1.0) <frame1.cols ? trackrect0.at<float>(1,3): frame1.cols))));

	}

	pow(map_aff(Range::all(), Range(0,4)), 2.0 , sc1);
	sc = sum(sc1); 
	sc = sc/2.0;
	sqrt(sc,sc);

	sc_temp.at<float>(0,1) = sc.at<float>(0,0);
	sc_temp.at<float>(0,2) = sc.at<float>(0,0);
	sc_temp.at<float>(0,4) = sc.at<float>(0,0);
	sc_temp.at<float>(0,5) = sc.at<float>(0,0);

	multiply(rel_std_afnv, sc_temp, std_aff);
	map_aff = map_aff+0.00000000000001;
	log(aff_samples.col(0), aff_samples.col(0));
    log(aff_samples.col(3), aff_samples.col(3));
	randn(

	aff_samples = draw_sample(aff_samples, std_aff);
	Mat flow_sum = Mat::zeros(1,6, CV_32FC1);
	Mat dumflow = Mat::ones(n_sample, 1, CV_32FC1);
	Mat v_sample;
	if(numframes>1){
		flow_sum.at<float>(0,4) = vy_avg.at<float>(0,0)*2.0;
		flow_sum.at<float>(0,5) = vx_avg.at<float>(0,0)*2.0;
		//cout << vx_avg << endl;
		//cout << vy_avg << endl;
		v_sample = dumflow*flow_sum;
		aff_samples = aff_samples +v_sample;
	}

	gly1 = crop_candidates(frame1_gray, aff_samples, sz_T);
	Y_whiten = whitening(gly1.crop);
	Y_normal = normalizeTemplate(Y_whiten.output);
	eta_max = FLT_MIN;

	q = Mat::zeros(n_sample, 1, CV_32FC1);
	forabs = abs(Y_normal.data);
	absum_Y = Mat::zeros(1, forabs.cols, CV_32FC1);

	for(int j=0; j<n_sample; j++)
	{
		absum_Y = sum(forabs.col(j));
		if((gly1.inrange.at<float>(j,0)==0)||(absum_Y.at<float>(0,j)==0))
		{
			continue;
		}
		muldum =Y_normal.data.col(j)-(Temp_1*Y_normal.data.col(j));
		q.row(j) = norm( muldum, NORM_L2);
		pow(q.row(j), 2.0, q.row(j));
		q.row(j) = -1.0*alpha*q.row(j);
		exp(q.row(j), q.row(j));
	}

	q_sort = Mat::zeros(q.rows, q.cols, CV_32FC1);
	q_idx; 
	cv::sort(q,q_sort,CV_SORT_EVERY_COLUMN +CV_SORT_DESCENDING);
	cv::sortIdx(q,q_idx, CV_SORT_EVERY_COLUMN +CV_SORT_DESCENDING);

	//second stage
	prob = Mat::zeros(n_sample, 1, CV_32FC1);
	n =0;
	tau = 0.0;
	sumcoeff = Mat::zeros(1,1,CV_32FC1);		
	while((n<n_sample) && (q_sort.at<float>(n,0) >= tau))
	{
		TempRealt = TempReal.t(); 
		lassoinput = TempRealt*Y_normal.data.col(q_idx.at<int>(n,0));     ///////////////lassoinput \C0\CC \B3ʹ\AB \C0۾\C6!!!!!!//////////////////
		coeff = APGLASSO(lassoinput, Dict, lambda, Lip , Maxit, nT);
		hconcat(A(Range::all(), Range(0, nT)), fixT, acont);
		vconcat(coeff(Range(0,nT), Range::all()), coeff.row(coeff.rows-1), coeffcon);
		D_s = Y_normal.data.col(q_idx.at<int>(n,0))- (acont*coeffcon);
		pow(D_s, 2.0, D_s);
		sumD_s = sum(D_s);
		sumD_s = -1.0*alpha*sumD_s;
		exp(sumD_s, sumD_s);
		prob.at<float>(q_idx.at<int>(n,0),0) = sumD_s.at<float>(0,0);
		tau = tau+ ( prob.at<float>(q_idx.at<int>(n,0),0) /(2.0*(float)n_sample-1.0));
		sumcoeff = sum(coeff(Range(0, nT), Range::all()));
		if (sumcoeff.at<float>(0,0) <0)
			continue;
		else if(prob.at<float>(q_idx.at<int>(n,0),0) >eta_max)
		{
			id_max = q_idx.at<int>(n,0);
			coeff.copyTo(c_max);
			eta_max  = prob.at<float>(q_idx.at<int>(n,0),0);
		}
		n = n+1;
	}

	map_aff = aff_samples.row(id_max);
	a_max = c_max( Range(0,nT), Range::all());
	aff_samples =  resample(aff_samples, prob, map_aff);
	minMaxLoc(a_max, NULL, NULL, &indA, NULL, noArray());
	min_angle = images_angle(Y_normal.data.col(id_max), A.col(indA.y));

	// Template Update
	occlusionNf = occlusionNf-1;
	level = 0.03;
	areathres = Our_round(sz_T*0.25);
	areath = areathres.at<float>(0,0)* areathres.at<float>(0,1);
	if ((min_angle > angle_threshold) && (occlusionNf <0))
	{
		cout <<"Update!" << endl;
		trivial_coef = c_max(Range(nT, c_max.rows-1), Range::all());
		trivial_coef_sh = trivial_coef.reshape(0,(int)sz_T.at<float>(0,1));
		trivial_coeft = trivial_coef_sh.t();
		se = Mat::zeros(5,5,CV_8UC1);
		se.at<unsigned char>(1,2) = 1;
		se.at<unsigned char>(2,1) = 1;
		se.at<unsigned char>(2,2) = 1;
		se.at<unsigned char>(2,3) = 1;
		se.at<unsigned char>(3,2) = 1;
		threshold(trivial_coeft, trivial_coef_bi, level, 1.0, THRESH_BINARY);
		morphologyEx(trivial_coef_bi, trivial_coef_se, 1, se);
		areasize = findBlobs(trivial_coef_se);
		for(int k=0; k<areas.rows;k++)
		{
			areas = areasize.at<int>(k,0)*areasize.at<int>(k,1);	
			minMaxLoc(areas, NULL, &maxarea, NULL, NULL, noArray());
		}
		if (maxarea < areath)
		{
			minMaxLoc(a_max(Range(0,nT), Range::all()), NULL, NULL, &indw, NULL, noArray());
			Y_normal.data.col(id_max).copyTo(Tmp1.cropimg.col(indw.y));
			Y_whiten.outmean.col(id_max).copyTo(Tmp1.cropmean.col(indw.y));
			norms.col(indw.y)=Y_whiten.outstd.col(id_max)*Y_normal.norms.col(id_max);
			normal2 = normalizeTemplate(Tmp1.cropimg);
			normal2.data.copyTo(Tmp1.cropimg);
			Tmp1.cropimg.copyTo(A(Range::all(), Range(0,nT)));
			hconcat(A, fixT, TempReal);
			Dict = TempReal.t()*TempReal;
			hconcat(Tmp1.cropimg, fixT, Temp_1);

			// method 3: 
			SVD svd(Temp_1);
			for(int ii=0; ii<svd.w.rows;ii++){
				if (svd.w.at<float>(ii,0) < 0.001){
					svd.w.at<float>(ii,0) = 0.0; 
				}
			}
			Temp_1inv = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();
			Temp_1 = Temp_1*Temp_1inv;
		}
		else{
			occlusionNf = 5.0;
			lambda.at<float>(0,2) = 0.0;
		}
	}
	else if(occlusionNf <0){
		lambda.at<float>(0,2) = initlambda3.at<float>(0,0);
	}
	frame1.copyTo(frame0);
	trackrect = drawAffine(map_aff, sz_T);
	return trackrect;
}






Tracker::uv  Tracker::farneback_flow(Mat frame0_rgb_, Mat frame1_rgb_, int numLevels, double pyrScale, bool fastPyramids, int winSize, int numIters,	int polyN, double polySigma, bool resize_img, double rfactor)
{ 

	Mat frame0_rgb, frame1_rgb, frame0, frame1;
	GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
	Mat imgU, imgV;
	Mat motion_flow, flow_rgb;
	int width = 0, height = 0;
	// Show CUDA information
	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	if( resize_img == true )
	{
		frame1_rgb = cv::Mat(Size(cvRound(frame1_rgb_.cols/rfactor),cvRound(frame1_rgb_.rows/rfactor)),CV_8UC3);
		width = frame1_rgb.cols;
		height = frame1_rgb.rows;
		cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
		frame0_rgb = cv::Mat(Size(cvRound(frame0_rgb_.cols/rfactor),cvRound(frame0_rgb_.rows/rfactor)),CV_8UC3);
		//width = frame0_rgb.cols;
		//height = frame1_rgb.rows;
		cv::resize(frame0_rgb_,frame0_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
	}
	else
	{
		frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
		width = frame1_rgb.cols; //640
		height = frame1_rgb.rows; //480
		frame1_rgb_.copyTo(frame1_rgb);
		frame0_rgb = cv::Mat(Size(frame0_rgb_.cols,frame0_rgb_.rows),CV_8UC3);
		//width = frame1_rgb.cols; //640
		//height = frame1_rgb.rows; //480
		frame0_rgb_.copyTo(frame0_rgb);
	}
	//frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
	frame0 = cv::Mat(Size(width,height),CV_8UC1);
	frame1 = cv::Mat(Size(width,height),CV_8UC1);
	flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
	motion_flow = cv::Mat(Size(width,height),CV_8UC3);
	cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
	cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
	// Create the optical flow object
	cv::gpu::FarnebackOpticalFlow dflow;
	dflow.numLevels = numLevels;
	dflow.pyrScale = pyrScale;
	dflow.fastPyramids = fastPyramids;
	dflow.winSize = winSize;
	dflow.numIters = numIters;
	dflow.polyN = polyN;
	dflow.polySigma = polySigma;
	frame1GPU.upload(frame1);
	frame0GPU.upload(frame0);

	// Do the dense optical flow
	dflow(frame0GPU,frame1GPU,uGPU,vGPU);
	uGPU.download(imgU);
	vGPU.download(imgV);
	struct uv uv1 = {Mat::zeros(imgU.rows, imgU.cols, CV_32FC1), Mat::zeros(imgV.rows, imgV.cols, CV_32FC1)};
	imgU.copyTo(uv1.vx);
	imgV.copyTo(uv1.vy);
	return uv1;
}


Mat Tracker::findBlobs(Mat binary)
{
	Mat label_image;
	binary.convertTo(label_image, CV_32SC1);
	Mat blobarea;

	int label_count = 2;
	for(int y=0; y < label_image.rows; y++) {
		for (int x =0; x < label_image.cols; x++){
			if(label_image.at<int>(y, x) !=1){
				continue;
			}
			Rect rect;
			Mat blob;
			floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

			for(int i=rect.y; i<rect.y+rect.height; i++){
				for(int j=rect.x; j< rect.x+rect.width; j++){
					if(label_image.at<int>(i,j) != label_count){
						continue;}
					blob.push_back(Point2i(j,i));
				}
			}
			blobarea.push_back(blob.size());
			label_count++;
		}
	}
	return blobarea;
}




Mat Tracker::Our_round(Mat x)
{
	//Matlab's LCC does not contain round(),
	// use floor() to fake one.
	int numrow = x.rows;
	int numcol = x.cols;
	Mat output = Mat::zeros(numrow, numcol, CV_32FC1);
	for (int i =0; i< numrow; i++)
	{
		for(int j=0; j< numcol; j++)
		{
			output.at<float>(i,j) = floor(x.at<float>(i,j));
			if((((x.at<float>(i,j)- output.at<float>(i,j)) >= 0.5) && (output.at<float>(i,j) >= 0)) ||  (((x.at<float>(i,j)- output.at<float>(i,j)) > 0.5) && (output.at<float>(i,j) < 0)))
			{
				output.at<float>(i,j) = output.at<float>(i,j)+1;
		
			}
		}
	}
	return output;
}

Mat Tracker::imgaffine(Mat img, Mat map_afnv, Mat tsize, int *L)
{

	float value,value_1,value_2,total_value;
	int M,N;
	int M_in,N_in;
	int i,j,k, m,n;
	int v2, v1;
	double count;

	M_in = img.rows;
	N_in = img.cols;
	Mat R = Mat::zeros(3, 3, CV_32FC1);

	R.at<float>(0,0) = map_afnv.at<float>(0,0);
	R.at<float>(1,0) = map_afnv.at<float>(0,2);
	R.at<float>(2,0) = 0.0;
	R.at<float>(0,1) = map_afnv.at<float>(0,1);
	R.at<float>(1,1) = map_afnv.at<float>(0,3);
	R.at<float>(2,1) = 0.0;
	R.at<float>(0,2) = map_afnv.at<float>(0,4);
	R.at<float>(1,2) = map_afnv.at<float>(0,5);
	R.at<float>(2,2) = 1.0;

	M = (int)tsize.at<float>(0,0);
	N = (int)tsize.at<float>(0,1);

	Mat P = Mat::zeros(3, M*N, CV_32FC1);

	for(i=0,j=1,k=1;i<M*N;i++){
		P.at<float>(0,i) = j; j++;  
		P.at<float>(1,i) = k;
		if(j == M+1){
			j = 1.0;
			k++;
		}
		P.at<float>(2,i) = 1.0;
	}
	//	cout <<P;
	Mat K = Mat::zeros(3, M*N, CV_32FC1);
	K = R*P;//matrix_multiply(Rp,3,3,Pp,3,M*N,Kp);
	K = Our_round(K);
	Mat ROUT = Mat::zeros(M,N, CV_8UC1);

	count = 0;
	value = 0;
	total_value = 0;
	int r,c; 
	for(n=0, i=0 ;n<N;n++){ // col

		for(m=0 ;m<M;m++){ // row 
			ROUT.at<unsigned char>(m,n) = 0; 

			//use the first M*N entries of P to restore j.
			//P.at<float>(0,i) = 0;
			c = i/3; 
		
			r = i%3; 
			P.at<float>(r,c) = 0;

			value_1 = K.at<float>(0,i);//*(Kp+0+i*3);
			value_2 = K.at<float>(1,i);//*(Kp+1+i*3);

			//cout <<"val1:" <<value_1 <<","<< "val2:" <<value_2 <<"\n";
			if((value_1 >= 1) & (value_1 <= M_in)
				& (value_2 >= 1) & (value_2 < N_in))
			{
				//P.at<float>(0,i) = 1;     
				//P.at<float>(m,n) = 1; 
				P.at<float>(r,c) = 1;
				count++;   

				v2=(int)(value_2 - 1);
				v1=(int)(value_1 - 1);

				value = img.at<unsigned char>(v1,  v2);; //*M_in + (int)((value_2 - 1)*M_in + value_1) - 1);

				// value = *(RINp + (ptrdiff_t)((value_2 - 1)*M_in + value_1) - 1);
				ROUT.at<unsigned char>(m,n) = value;
				total_value += value;
			}
			i++;
		}
	}
	//find mean value
	value = total_value/count;
	for(n=0,i=0 ;n<N;n++){
		for(m=0 ;m<M;m++)
		{
			c = i/3; 
			r = i%3; 
			//P.at<float>(r,c) = 0;
			//if(P.at<float>(0,i) == 0)
			if(P.at<float>(r,c) == 0)
				ROUT.at<unsigned char>(m,n) = value;

			i++;
		}
	}

	if(count>0)
		*L = 1;
	else 
		*L=0;
	return ROUT;
}
