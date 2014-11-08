#ifndef _TRANSFORM_H
#define _TRANSFORM_H
#define GRADIENT_SCALE 100000

// 100000

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/types_c.h"
#include "../inc/fio.h"

#include <random>
#include <iostream>

class Transform
{
private:
	
	// transformation structures		
	cv::Mat transform_inv_;
	cv::Mat transform_;
	cv::Mat prev_transform_inv_;
	cv::Mat prev_transform_;
	cv::Mat feature_;
	// weights
	cv::Mat w_0_0_;
	cv::Mat w_0_1_;
	cv::Mat w_0_2_;
	cv::Mat w_1_0_;
	cv::Mat w_1_1_;
	cv::Mat w_1_2_;

	cv::Mat w_0_0_grad_;
	cv::Mat w_0_1_grad_;
	cv::Mat w_0_2_grad_;
	cv::Mat w_1_0_grad_;
	cv::Mat w_1_1_grad_;
	cv::Mat w_1_2_grad_;

	cv::Mat w_0_0_grad_batch_;
	cv::Mat w_0_1_grad_batch_;
	cv::Mat w_0_2_grad_batch_;
	cv::Mat w_1_0_grad_batch_;
	cv::Mat w_1_1_grad_batch_;
	cv::Mat w_1_2_grad_batch_;

	cv::Mat element_0_0_;
	cv::Mat element_0_1_;
	cv::Mat element_0_2_;
	cv::Mat element_1_0_;
	cv::Mat element_1_1_;
	cv::Mat element_1_2_;

	cv::Mat w_grad_;
	cv::Mat fisher_inv_;
	cv::Mat tmp_;
	cv::Mat tmp_grad_;
	cv::Mat natural_grad_;

	int input_dim_;
	int output_dim_;
	int transform_dim_;

	double alpha_;

	double ini_x_;
	double ini_y_;
	double ini_phi_;
	double ini_sx_;
	double ini_sy_;

	double w_0_0_rate_;
	double w_0_1_rate_;
	double w_0_2_rate_;
	double w_1_0_rate_;
	double w_1_1_rate_;
	double w_1_2_rate_;

	double epsilon_;

	double average_norm_;
    double ini_norm_;
    double lambda_;

	int curr_dim_;
public:
	Transform();	
	// gradients
	void CalculateGradient(cv::Mat& original_point, cv::Mat& predicted_point, cv::Mat& target_point, cv::Mat& feature);	
	void CalcMiniBatchInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature, int batch_count, int batch_idx);
	void UpdateWeightBatch(int iter, int current_dim);
	// calculate inverse transformation			
	void CalcTransformInv(cv::Mat& feature);	
	cv::Mat TransformDataPointInv(cv::Mat& point, int curr_flag);
	cv::Mat TransformToPreviousFrame(cv::Mat& curr_img_point);
	cv::Mat TransformToNextFrame(cv::Mat& prev_home_point);
	void SetLearningRates(double normal_rate, double natural_rate); // , double rate_0_2, double rate_1_0, double rate_1_1, double rate_1_2);
	// copy to previous transformation
	void CopyToPrev();
	cv::Mat AggregateGradients(int current_dim);
	void RetrieveGradients();
	// helper functions
	cv::Mat w_0_0();
	cv::Mat w_0_1();
	cv::Mat w_0_2();
	cv::Mat w_1_0();
	cv::Mat w_1_1();	
	cv::Mat w_1_2();	
	cv::Mat fisher_inv();
	cv::Mat transform_inv();
	cv::Mat prev_transform_inv();
	cv::Mat natural_grad();
	cv::Mat w_grad();
	void set_w_0_0(cv::Mat& w);
	void set_w_0_1(cv::Mat& w);
	void set_w_0_2(cv::Mat& w);
	void set_w_1_0(cv::Mat& w);
	void set_w_1_1(cv::Mat& w);
	void set_w_1_2(cv::Mat& w);
	// check gradient
	void CheckInvGradient();
};

#endif