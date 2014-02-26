#ifndef _TRANSFORM_H
#define _TRANSFORM_H
#define GRADIENT_SCALE 100

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/types_c.h"
#include "fio.h"

#include <random>
#include <iostream>

class Transform
{
private:
	// weights
	cv::Mat w_x_;
	cv::Mat w_y_;
	cv::Mat w_phi_;
	cv::Mat w_sx_;
	cv::Mat w_sy_;
	// gradient
	cv::Mat w_x_grad_;
	cv::Mat w_y_grad_;
	cv::Mat w_phi_grad_;
	cv::Mat w_sx_grad_;
	cv::Mat w_sy_grad_;

	cv::Mat w_x_grad_batch_;
	cv::Mat w_y_grad_batch_;
	cv::Mat w_phi_grad_batch_;
	cv::Mat w_sx_grad_batch_;
	cv::Mat w_sy_grad_batch_;

	// transformation structures	
	cv::Mat scaling_inv_;
	cv::Mat rotation_inv_;
	cv::Mat translate_inv_;	
	cv::Mat transform_inv_;
	cv::Mat prev_transform_inv_;
	cv::Mat transform_para_;	
	cv::Mat feature_;

	int input_dim_;
	int output_dim_;
	int transform_dim_;

	double alpha_;

	double ini_x_;
	double ini_y_;
	double ini_phi_;
	double ini_sx_;
	double ini_sy_;

	double w_x_rate_;
	double w_y_rate_;
	double w_angle_rate_;
	double w_sx_rate_;
	double w_sy_rate_;

public:
	Transform(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle);
	// gradients
	void CalcWXInvGradient(cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcWYInvGradient(cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcWPhiInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcWSxInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcWSyInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature);
	void CalcMiniBatchInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature, int batch_count, int batch_idx);
	void UpdateWeightBatch();
	// calculate inverse transformation		
	void set_scaling_inv(cv::Mat& feature, cv::Mat& w_sx, cv::Mat& w_sy);
	void set_rotation_inv(cv::Mat& feature, cv::Mat& w_phi);
	void set_translate_inv(cv::Mat& feature, cv::Mat& w_x, cv::Mat& w_y);	
	void CalcTransformMatrixInv(cv::Mat& feature, cv::Mat& w_x, cv::Mat& w_y, cv::Mat& w_phi, cv::Mat& w_sx, cv::Mat& w_sy);	
	cv::Mat EvaluateInvTransformation(cv::Mat& feature_float);	
	cv::Mat TransformDataPointInv(cv::Mat& point, int curr_flag);
	void SetLearningRates(double x_rate, double y_rate, double angle_rate, double sx_rate, double sy_double);
	// copy to previous transformation
	void CopyToPrev();
	// helper functions
	cv::Mat w_x();
	cv::Mat w_y();
	cv::Mat w_phi();
	cv::Mat w_sx();
	cv::Mat w_sy();	
	cv::Mat transform_inv();
	void set_w_x(cv::Mat& w);
	void set_w_y(cv::Mat& w);
	void set_w_phi(cv::Mat& w);
	void set_w_sx(cv::Mat& w);
	void set_w_sy(cv::Mat& w);
	// check gradient
	void CheckInvGradient();
};

#endif