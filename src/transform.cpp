#include "../inc/transform.h"

Transform::Transform(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle) : 
	ini_x_(initial_x), 
	ini_y_(initial_y),
	ini_sx_(initial_long_axis),
	ini_sy_(initial_short_axis),
	ini_phi_(initial_angle)
{
	input_dim_ = 10;
	output_dim_ = 1;
	transform_dim_ = 3;

	alpha_ = 0.2; // randomly picked learning rate...

	// initial parameters
	/*ini_x_ = 218; 
	ini_y_ = 230; 
	ini_phi_ = 1.0 * Pi / 8.5; 
	ini_sx_ = 30; 
	ini_sy_ = 22;  */

	w_x_rate_ = 0;
	w_y_rate_ = 0;
	w_angle_rate_ = 0;
	w_sx_rate_ = 0;
	w_sy_rate_ = 0;

	w_x_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_y_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_phi_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_sx_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_sy_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);

	w_x_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_y_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_phi_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_sx_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_sy_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);

	feature_ = cv::Mat::zeros(input_dim_, output_dim_, CV_64F);
	
	scaling_inv_ = cv::Mat::zeros(transform_dim_, transform_dim_, CV_64F);
	rotation_inv_ = cv::Mat::zeros(transform_dim_, transform_dim_, CV_64F);
	translate_inv_ = cv::Mat::zeros(transform_dim_, transform_dim_, CV_64F);
	transform_inv_ = cv::Mat::zeros(transform_dim_, transform_dim_, CV_64F);
	prev_transform_inv_ = cv::Mat::zeros(transform_dim_, transform_dim_, CV_64F);

	transform_para_ = cv::Mat::zeros(5, 1, CV_64F);

	// set initial inv transformation
	CalcTransformMatrixInv(feature_, w_x_, w_y_, w_phi_, w_sx_, w_sy_);

	std::random_device rd;
	std::mt19937 engine_uniform_rand(rd());
	double weight_range = 0.0001;
	std::uniform_real_distribution<double> uniform_rand(-weight_range, weight_range);
	// initialize all the weights
	for(int p = 0; p < output_dim_; p++)
	{
		for(int q = 0; q < input_dim_; q++)
		{
			w_x_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_y_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_phi_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_sx_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_sy_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
		}
	}

}


// set inverse transformation
void Transform::set_translate_inv(cv::Mat& feature, cv::Mat& w_x, cv::Mat& w_y)
{
	cv::Mat tmp;
	translate_inv_.at<double>(0, 0) = 1.0;
	translate_inv_.at<double>(1, 1) = 1.0;
	translate_inv_.at<double>(2, 2) = 1.0;
	tmp = w_x * feature;
	translate_inv_.at<double>(0, 2) = -(ini_x_ + tmp.at<double>(0, 0));
	transform_para_.at<double>(0, 0) = tmp.at<double>(0, 0); // shift x
	tmp = w_y * feature;
	translate_inv_.at<double>(1, 2) = -(ini_y_ + tmp.at<double>(0, 0));
	transform_para_.at<double>(1, 0) = tmp.at<double>(0, 0); // shift y
}

void Transform::set_rotation_inv(cv::Mat& feature, cv::Mat& w_phi)
{
	cv::Mat tmp;
	tmp = w_phi * feature;
	transform_para_.at<double>(2, 0) = tmp.at<double>(0, 0); // rotation angle
	rotation_inv_.at<double>(0, 0) = cos(-(ini_phi_ + tmp.at<double>(0, 0)));
	rotation_inv_.at<double>(0, 1) = -sin(-(ini_phi_ + tmp.at<double>(0, 0)));
	rotation_inv_.at<double>(1, 0) = sin(-(ini_phi_ + tmp.at<double>(0, 0)));
	rotation_inv_.at<double>(1, 1) = cos(-(ini_phi_ + tmp.at<double>(0, 0)));
	rotation_inv_.at<double>(2, 2) = 1.0;
}

void Transform::set_scaling_inv(cv::Mat& feature, cv::Mat& w_sx, cv::Mat& w_sy)
{
	cv::Mat tmp;
	tmp = w_sx * feature;
	scaling_inv_.at<double>(0, 0) = 1 / (ini_sx_ + tmp.at<double>(0, 0)); // delta with respect to 1
	transform_para_.at<double>(3, 0) = tmp.at<double>(0, 0); // x scaling
	tmp = w_sy * feature;
	scaling_inv_.at<double>(1, 1) = 1 / (ini_sy_ + tmp.at<double>(0, 0)); // delta with respect to 1
	transform_para_.at<double>(4, 0) = tmp.at<double>(0, 0); // y scaling
	scaling_inv_.at<double>(2, 2) = 1.0;
}

void Transform::CalcTransformMatrixInv(cv::Mat& feature, cv::Mat& w_x, cv::Mat& w_y, cv::Mat& w_phi, cv::Mat& w_sx, cv::Mat& w_sy)
{
	set_translate_inv(feature, w_x, w_y);
	set_rotation_inv(feature, w_phi);
	set_scaling_inv(feature, w_sx, w_sy);

	transform_inv_ = scaling_inv_ * rotation_inv_ * translate_inv_; 
}

cv::Mat Transform::EvaluateInvTransformation(cv::Mat& feature_float)
{
	for(int i = 0; i < input_dim_; i++)
		feature_.at<double>(i, 0) = (double)feature_float.at<double>(i, 0);
	CalcTransformMatrixInv(feature_, w_x_, w_y_, w_phi_, w_sx_, w_sy_);
	return transform_para_;
}

// calculate the transformation
cv::Mat Transform::TransformDataPointInv(cv::Mat& point, int curr_flag)
{
	cv::Mat transformed_point;
	if(curr_flag)
		transformed_point = transform_inv_ * point;
	else
		transformed_point = prev_transform_inv_ * point;
	return transformed_point;
}

cv::Mat Transform::InterFrameTransformImg(cv::Mat& point)
{
	cv::Mat transformed_point;
	cv::Mat transform;
	cv::invert(transform_inv_, transform);	
	transformed_point = transform * prev_transform_inv_ * point;			
	return transformed_point;
}

// calculate gradients
void Transform::CalcWXInvGradient(cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	// points
	double x_h = transformed_point.at<double>(0, 0);
	double y_h = transformed_point.at<double>(1, 0);
	double x_t = target_point.at<double>(0, 0);
	double y_t = target_point.at<double>(1, 0);
	// temporary values
	cv::Mat tmp_sx = w_sx_ * feature;
	cv::Mat tmp_sy = w_sy_ * feature;
	cv::Mat tmp_phi = w_phi_ * feature;
	double tmp_sx_val = ini_sx_ + tmp_sx.at<double>(0, 0);
	double tmp_sy_val = ini_sy_ + tmp_sy.at<double>(0, 0);
	double tmp_phi_val = ini_phi_ + tmp_phi.at<double>(0, 0);

	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));

	w_x_grad_ = feature.t() * exp_error * ((x_h - x_t) * (-cos(tmp_phi_val)) / tmp_sx_val + (y_h - y_t) * sin(tmp_phi_val) / tmp_sy_val) ;
}

void Transform::CalcWYInvGradient(cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	// points
	double x_h = transformed_point.at<double>(0, 0);
	double y_h = transformed_point.at<double>(1, 0);
	double x_t = target_point.at<double>(0, 0);
	double y_t = target_point.at<double>(1, 0);
	// temporary values
	cv::Mat tmp_sx = w_sx_ * feature;
	cv::Mat tmp_sy = w_sy_ * feature;
	cv::Mat tmp_phi = w_phi_ * feature;
	double tmp_sx_val = ini_sx_ + tmp_sx.at<double>(0, 0);
	double tmp_sy_val = ini_sy_ + tmp_sy.at<double>(0, 0);
	double tmp_phi_val = ini_phi_ + tmp_phi.at<double>(0, 0);

	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));

	w_y_grad_ = feature.t() * exp_error * ((x_h - x_t) * (-sin(tmp_phi_val)) / tmp_sx_val - (y_h - y_t) * cos(tmp_phi_val) / tmp_sy_val) ;
	// w_y_grad_ = feature.t() * (y_h - y_t) * exp_error;
}

void Transform::CalcWPhiInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	double x_0 = original_point.at<double>(0, 0);
	double y_0 = original_point.at<double>(1, 0);
	double x_h = transformed_point.at<double>(0, 0);
	double y_h = transformed_point.at<double>(1, 0);
	double x_t = target_point.at<double>(0, 0);
	double y_t = target_point.at<double>(1, 0);
	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));

	cv::Mat tmp_x = w_x_ * feature;
	cv::Mat tmp_y = w_y_ * feature;
	cv::Mat tmp_sx = w_sx_ * feature;
	cv::Mat tmp_sy = w_sy_ * feature;
	cv::Mat tmp_phi = w_phi_ * feature;
	double tmp_x_val = ini_x_ + tmp_x.at<double>(0, 0);
	double tmp_y_val = ini_y_ + tmp_y.at<double>(0, 0);
	double tmp_sx_val = ini_sx_ + tmp_sx.at<double>(0, 0);
	double tmp_sy_val = ini_sy_ + tmp_sy.at<double>(0, 0);
	double tmp_phi_val = ini_phi_ + tmp_phi.at<double>(0, 0);

	// expression of rotation weight gradient... need to check...
	w_phi_grad_ = exp_error * feature.t() * ((x_h - x_t) * (1 / tmp_sx_val) * (-sin(tmp_phi_val) * x_0 + cos(tmp_phi_val) * y_0 + tmp_x_val * sin(tmp_phi_val) - tmp_y_val * cos(tmp_phi_val)) + 
		(y_h - y_t) * (1 / tmp_sy_val) * (-cos(tmp_phi_val) * x_0 - sin(tmp_phi_val) * y_0 + tmp_x_val * cos(tmp_phi_val) + tmp_y_val * sin(tmp_phi_val)));

	// w_phi_grad_ = exp_error * ((x_h - x_t) * (x_0 * tmp_sx_val * -sin(tmp_phi_val) * feature.t() - (y_0 * tmp_sy_val * cos(tmp_phi_val)) * feature.t()) + 
	// 	(y_h - y_t) * (x_0 * tmp_sx_val * cos(tmp_phi_val) * feature.t() - (y_0 * tmp_sy_val * sin(tmp_phi_val)) * feature.t()));

}

void Transform::CalcWSxInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	double x_0 = original_point.at<double>(0, 0);
	double y_0 = original_point.at<double>(1, 0);
	double x_h = transformed_point.at<double>(0, 0);
	double y_h = transformed_point.at<double>(1, 0);
	double x_t = target_point.at<double>(0, 0);
	double y_t = target_point.at<double>(1, 0);
	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));

	cv::Mat tmp_phi = w_phi_ * feature;
	cv::Mat tmp_sx = w_sx_ * feature;
	double tmp_phi_val = ini_phi_ + tmp_phi.at<double>(0, 0);	
	double tmp_sx_val = ini_sx_ + tmp_sx.at<double>(0, 0);	

	// expression of sx weight gradient
	w_sx_grad_ = exp_error * ((x_h - x_t) * (-1 / tmp_sx_val) * x_h) * feature.t();

}

void Transform::CalcWSyInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	double x_0 = original_point.at<double>(0, 0);
	double y_0 = original_point.at<double>(1, 0);
	double x_h = transformed_point.at<double>(0, 0);
	double y_h = transformed_point.at<double>(1, 0);
	double x_t = target_point.at<double>(0, 0);
	double y_t = target_point.at<double>(1, 0);
	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));

	cv::Mat tmp_phi = w_phi_ * feature;
	cv::Mat tmp_sy = w_sy_ * feature;
	double tmp_phi_val = ini_phi_ + tmp_phi.at<double>(0, 0);		
	double tmp_sy_val = ini_sy_ + tmp_sy.at<double>(0, 0);	

	// expression of sx weight gradient
	w_sy_grad_ = exp_error * ((y_h - y_t) * (-1 / tmp_sy_val) * y_h) * feature.t();		
}

void Transform::CalcInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature)
{
	CalcWXInvGradient(transformed_point, target_point, feature);
	CalcWYInvGradient(transformed_point, target_point, feature);
	CalcWPhiInvGradient(original_point, transformed_point, target_point, feature);
	CalcWSxInvGradient(original_point, transformed_point, target_point, feature);
	CalcWSyInvGradient(original_point, transformed_point, target_point, feature);
}

void Transform::CalcMiniBatchInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature, int batch_count, int batch_idx)
{
	if(batch_idx == 0)
	{
		w_x_grad_batch_ = cv::Mat::zeros(batch_count, input_dim_, CV_64F);
		w_y_grad_batch_ = cv::Mat::zeros(batch_count, input_dim_, CV_64F);
		w_phi_grad_batch_ = cv::Mat::zeros(batch_count, input_dim_, CV_64F);
		w_sx_grad_batch_ = cv::Mat::zeros(batch_count, input_dim_, CV_64F);
		w_sy_grad_batch_ = cv::Mat::zeros(batch_count, input_dim_, CV_64F);
	}
	
	CalcInvGradient(original_point, transformed_point, target_point, feature);

	w_x_grad_.copyTo(w_x_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_y_grad_.copyTo(w_y_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_phi_grad_.copyTo(w_phi_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_sx_grad_.copyTo(w_sx_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_sy_grad_.copyTo(w_sy_grad_batch_.rowRange(batch_idx, batch_idx + 1));	
}

void Transform::UpdateWeightBatch()
{
	cv::reduce(w_x_grad_batch_, w_x_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_y_grad_batch_, w_y_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_phi_grad_batch_, w_phi_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_sx_grad_batch_, w_sx_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_sy_grad_batch_, w_sy_grad_, 0, CV_REDUCE_AVG);

	w_x_ = w_x_ - w_x_rate_ * w_x_grad_;
	w_y_ = w_y_ - w_y_rate_ * w_y_grad_;
	w_phi_ = w_phi_ - w_angle_rate_ * w_phi_grad_;
	w_sx_ = w_sx_ - w_sx_rate_ * w_sx_grad_;
	w_sy_ = w_sy_ - w_sy_rate_ * w_sy_grad_;

	/*w_x_ = w_x_ - 0.32 * w_x_grad_;
	w_y_ = w_y_ - 0.32 * w_y_grad_;
	w_phi_ = w_phi_ - 4e-5 * w_phi_grad_;
	w_sx_ = w_sx_ - 0.12 * w_sx_grad_;
	w_sy_ = w_sy_ - 0.03 * w_sy_grad_;*/
}

// copy to previous transformation
void Transform::CopyToPrev()
{
	transform_inv_.copyTo(prev_transform_inv_);
}

void Transform::CheckInvGradient()
{	
	// assign feature value
	char input_dir[400];
	cv::Mat feature = cv::Mat::zeros(input_dim_, 1, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/test/feature.bin");
	FileIO::ReadMatDouble(feature, input_dim_, 1, input_dir);
	
	// allocate data points
	cv::Mat original_point = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat transformed_point = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat target_point = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	// numerical gradient
	cv::Mat transformed_point_delta_1 = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat transformed_point_delta_2 = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat disturb = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	cv::Mat tmp_w;
	double e_1 = 0;
	double e_2 = 0;
	double disturb_value = 0.001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;
	
	target_point.at<double>(0, 0) = -0.3709615735; target_point.at<double>(1, 0) = -0.7597566886; target_point.at<double>(2, 0) = 1.0;	
	original_point.at<double>(0, 0) = 213.607177734375; original_point.at<double>(1, 0) = 210.383468628; original_point.at<double>(2, 0) = 1.0;
	// calculate the current feature vector
	CalcTransformMatrixInv(feature, w_x_, w_y_, w_phi_, w_sx_, w_sy_);
	// calculate the transformed point
	transformed_point = TransformDataPointInv(original_point, 1);
	// calculate analytical gradient
	CalcInvGradient(original_point, transformed_point, target_point, feature);
	// currently checking gradient for phi...
	for(int i = 0; i < input_dim_; i++)
	{
		disturb = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
		disturb.at<double>(0, i) = disturb_value;
		tmp_w = w_phi_ + disturb;
		CalcTransformMatrixInv(feature, w_x_, w_y_, tmp_w, w_sx_, w_sy_);
		transformed_point_delta_1 = TransformDataPointInv(original_point, 1);
		tmp_w = w_phi_ - disturb;
		CalcTransformMatrixInv(feature, w_x_, w_y_, tmp_w, w_sx_, w_sy_);
		transformed_point_delta_2 = TransformDataPointInv(original_point, 1);
		// reset back...
		CalcTransformMatrixInv(feature, w_x_, w_y_, w_phi_, w_sx_, w_sy_);
		transformed_point = TransformDataPointInv(original_point, 1);
		// calculate error
		e_1 = 1 - exp(-0.5 * GRADIENT_SCALE * (pow(target_point.at<double>(0, 0) - transformed_point_delta_1.at<double>(0, 0), 2) +
			pow(target_point.at<double>(1, 0) - transformed_point_delta_1.at<double>(1, 0), 2)));
		e_2 = 1 - exp(-0.5 * GRADIENT_SCALE * (pow(target_point.at<double>(0, 0) - transformed_point_delta_2.at<double>(0, 0), 2) +
			pow(target_point.at<double>(1, 0) - transformed_point_delta_2.at<double>(1, 0), 2)));
		numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
		analytical_gradient = w_phi_grad_.at<double>(0, i);

		std::cout << "iteration: " << i << std::endl;
		std::cout << "analytical gradient: " << analytical_gradient << " " << "numerical gradient: " << numerical_gradient << std::endl;
	}
	
}

cv::Mat Transform::w_x()
{
	return w_x_;
}

cv::Mat Transform::w_y()
{
	return w_y_;
}

cv::Mat Transform::w_phi()
{
	return w_phi_;
}

cv::Mat Transform::w_sx()
{
	return w_sx_;
}

cv::Mat Transform::w_sy()
{
	return w_sy_;
}

cv::Mat Transform::transform_inv()
{
	return transform_inv_;
}

cv::Mat Transform::prev_transform_inv()
{
	return prev_transform_inv_;
}

void Transform::set_w_x(cv::Mat& w)
{
	 w.copyTo(w_x_);
}

void Transform::set_w_y(cv::Mat& w)
{
	w.copyTo(w_y_);
}

void Transform::set_w_phi(cv::Mat& w)
{
	w.copyTo(w_phi_);
}

void Transform::set_w_sx(cv::Mat& w)
{
	w.copyTo(w_sx_);
}

void Transform::set_w_sy(cv::Mat& w)
{
	w.copyTo(w_sy_);
}

void Transform::SetLearningRates(double x_rate, double y_rate, double angle_rate, double sx_rate, double sy_rate)
{
	w_x_rate_ = x_rate;
	w_y_rate_ = y_rate;
	w_angle_rate_ = angle_rate;
	w_sx_rate_ = sx_rate;
	w_sy_rate_ = sy_rate;
}