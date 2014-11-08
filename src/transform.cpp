#include "../inc/transform.h"

Transform::Transform()
{
	// 
	input_dim_ = 8; // 8; // 8;
	output_dim_ = 1;
	transform_dim_ = 3;
	// learning rates
	w_0_0_rate_ = 0;
	w_0_1_rate_ = 0;
	w_0_2_rate_ = 0;
	w_1_0_rate_ = 0;
	w_1_1_rate_ = 0;
	w_1_2_rate_ = 0;	
	// weights
	w_0_0_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_0_1_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_0_2_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_0_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_1_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_2_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	// gradients
	w_0_0_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_0_1_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_0_2_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_0_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_1_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	w_1_2_grad_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);

	w_grad_ = cv::Mat::zeros(input_dim_ * 6, output_dim_, CV_64F);
	fisher_inv_ = cv::Mat::eye(input_dim_ * 6, input_dim_ * 6, CV_64F);
	// matrix elements
	element_0_0_ = cv::Mat::zeros(1, 1, CV_64F);
	element_0_1_ = cv::Mat::zeros(1, 1, CV_64F);
	element_0_2_ = cv::Mat::zeros(1, 1, CV_64F);
	element_1_0_ = cv::Mat::zeros(1, 1, CV_64F);
	element_1_1_ = cv::Mat::zeros(1, 1, CV_64F);
	element_1_2_ = cv::Mat::zeros(1, 1, CV_64F);
	// feature
	feature_ = cv::Mat::zeros(input_dim_, output_dim_, CV_64F);
	// transformations
	transform_inv_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
	prev_transform_inv_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);	
	transform_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
	prev_transform_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);	

	std::random_device rd;
	std::mt19937 engine_uniform_rand(rd());
	double weight_range = 0.0001;
	std::uniform_real_distribution<double> uniform_rand(-weight_range, weight_range);

	average_norm_ = 1;
    ini_norm_ = 1;
    lambda_ = 0.2;
	// initialize all the weights
	for(int p = 0; p < output_dim_; p++)
	{
		for(int q = 0; q < input_dim_; q++)
		{
			w_0_0_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_0_1_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_0_2_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_1_0_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_1_1_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
			w_1_2_.at<double>(p, q) = uniform_rand(engine_uniform_rand);
		}
	}
	
	// epsilon_ = 1e-5;
	epsilon_ = 1e-6;

}

void Transform::CalcTransformInv(cv::Mat& feature)
{
	element_0_0_ = w_0_0_.colRange(0, feature.rows) * feature; element_0_1_ = w_0_1_.colRange(0, feature.rows) * feature;	
	element_0_2_ = w_0_2_.colRange(0, feature.rows) * feature; element_1_0_ = w_1_0_.colRange(0, feature.rows) * feature;	
	element_1_1_ = w_1_1_.colRange(0, feature.rows) * feature; element_1_2_ = w_1_2_.colRange(0, feature.rows) * feature;	

	transform_inv_.at<double>(0, 0) = 1.0 + element_0_0_.at<double>(0, 0);	transform_inv_.at<double>(0, 1) = element_0_1_.at<double>(0, 0);	
	transform_inv_.at<double>(0, 2) = element_0_2_.at<double>(0, 0);		transform_inv_.at<double>(1, 0) = element_1_0_.at<double>(0, 0);	
	transform_inv_.at<double>(1, 1) = 1.0 + element_1_1_.at<double>(0, 0);	transform_inv_.at<double>(1, 2) = element_1_2_.at<double>(0, 0);	

	/*transform_.at<double>(0, 0) = 1.0 + element_0_0_.at<double>(0, 0);	transform_.at<double>(0, 1) = element_0_1_.at<double>(0, 0);	
	transform_.at<double>(0, 2) = element_0_2_.at<double>(0, 0);		transform_.at<double>(1, 0) = element_1_0_.at<double>(0, 0);	
	transform_.at<double>(1, 1) = 1.0 + element_1_1_.at<double>(0, 0);	transform_.at<double>(1, 2) = element_1_2_.at<double>(0, 0);	*/
	cv::invert(transform_inv_, transform_);

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

cv::Mat Transform::TransformToPreviousFrame(cv::Mat& curr_img_point)
{
	cv::Mat prev_predicted_point;
	cv::Mat prev_transform;
	cv::invert(prev_transform_inv_, prev_transform);	
	prev_predicted_point = prev_transform * transform_inv_ * curr_img_point;			
	return prev_predicted_point;
}

cv::Mat Transform::TransformToNextFrame(cv::Mat& prev_home_point)
{
	cv::Mat predicted_point;
	cv::Mat transform;
	cv::invert(transform_inv_, transform);	
	predicted_point = transform * prev_home_point;			
	return predicted_point;
}

//// calculate gradients
void Transform::CalculateGradient(cv::Mat& original_point, cv::Mat& predicted_point, cv::Mat& target_point, cv::Mat& feature)
{
	// points
	double x_0 = original_point.at<double>(0, 0); double y_0 = original_point.at<double>(1, 0);	
	double x_h = predicted_point.at<double>(0, 0); double y_h = predicted_point.at<double>(1, 0);	
	double x_t = target_point.at<double>(0, 0); double y_t = target_point.at<double>(1, 0);	
	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * (pow(x_t - x_h, 2) + pow(y_t - y_h, 2)));
	// gradient expressions... simple...
	w_0_0_grad_ = exp_error * (x_h - x_t) * x_0 * feature.t();
	w_0_1_grad_ = exp_error * (x_h - x_t) * y_0 * feature.t();
	w_0_2_grad_ = exp_error * (x_h - x_t) * 1.0 * feature.t();
	w_1_0_grad_ = exp_error * (y_h - y_t) * x_0 * feature.t();
	w_1_1_grad_ = exp_error * (y_h - y_t) * y_0 * feature.t();
	w_1_2_grad_ = exp_error * (y_h - y_t) * 1.0 * feature.t();
}

// calculate gradients
//void Transform::CalculateGradient(cv::Mat& original_point, cv::Mat& predicted_point, cv::Mat& target_point, cv::Mat& feature)
//{
//	// points
//	double x_0 = original_point.at<double>(0, 0); double y_0 = original_point.at<double>(1, 0);	
//	double x_h = predicted_point.at<double>(0, 0); double y_h = predicted_point.at<double>(1, 0);	
//	double x_t = target_point.at<double>(0, 0); double y_t = target_point.at<double>(1, 0);	
//	double exp_error = GRADIENT_SCALE * exp(-0.5 * GRADIENT_SCALE * ((x_t - x_h) * (x_t - x_h) + (y_t - y_h) * (y_t - y_h)));
//	double m_00 = transform_.at<double>(0, 0); double m_01 = transform_.at<double>(0, 1); double m_02 = transform_.at<double>(0, 2);
//	double m_10 = transform_.at<double>(1, 0); double m_11 = transform_.at<double>(1, 1); double m_12 = transform_.at<double>(1, 2);
//	double det = m_00 * m_11 - m_01 * m_10;
//	double a_x = (m_11 * x_0 - m_01 * y_0 + m_01 * m_12 - m_11 * m_02);
//	double a_y = (-m_10 * x_0 + m_00 * y_0 + m_02 * m_10 - m_00 * m_12);
//
//	w_0_0_grad_ = (exp_error / (det * det)) * ((x_h - x_t) * a_x * (-m_11) + (y_h - y_t) * ((y_0 - m_12) * (-m_01 * m_10) - m_11 * (m_02 * m_10 - m_10 * x_0))) * feature.t();
//	w_0_1_grad_ = (exp_error / (det * det)) * ((x_h - x_t) * ((m_12 - y_0) * (m_00 * m_11) - (-m_10) * (m_11 * x_0 - m_11 * m_02)) + (y_h - y_t) * a_y * m_10) * feature.t();
//	w_0_2_grad_ = exp_error / det * ((x_h - x_t) * (-m_11) + (y_h - y_t) * m_10) * feature.t();
//	// gradient expressions... simple...
//	// w_0_0_grad_ = exp_error * (x_h - x_t) * x_0 * feature.t();
//	// w_0_1_grad_ = exp_error * (x_h - x_t) * y_0 * feature.t();
//	// w_0_2_grad_ = exp_error * (x_h - x_t) * 1.0 * feature.t();
//	w_1_0_grad_ = (exp_error / (det * det)) * ((x_h - x_t) * a_x * m_01 + (y_h - y_t) * ((m_02 - x_0) * (m_00 * m_11) - (-m_01) * (m_00 * y_0 - m_00 * m_12))) * feature.t();
//	w_1_1_grad_ = (exp_error / (det * det)) * ((x_h - x_t) * ((x_0 - m_02) * (-m_01 * m_10) - m_00 * (m_01 * m_12 - m_01 * y_0)) + (y_h - y_t) * a_y * (-m_00)) * feature.t();
//	w_1_2_grad_ = exp_error / det * ((x_h - x_t) * m_01 + (y_h - y_t) * (-m_00)) * feature.t();
//	/*w_1_0_grad_ = exp_error * (y_h - y_t) * x_0 * feature.t();
//	w_1_1_grad_ = exp_error * (y_h - y_t) * y_0 * feature.t();
//	w_1_2_grad_ = exp_error * (y_h - y_t) * 1.0 * feature.t();*/
//
//	
//
//}

void Transform::CalcMiniBatchInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature, int batch_count, int batch_idx)
{
	int current_dim = feature.rows;

	if(batch_idx == 0)
	{
		w_0_0_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
		w_0_1_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
		w_0_2_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
		w_1_0_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
		w_1_1_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
		w_1_2_grad_batch_ = cv::Mat::zeros(batch_count, current_dim, CV_64F);
	}

	CalculateGradient(original_point, transformed_point, target_point, feature);
	w_0_0_grad_.copyTo(w_0_0_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_0_1_grad_.copyTo(w_0_1_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_0_2_grad_.copyTo(w_0_2_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_1_0_grad_.copyTo(w_1_0_grad_batch_.rowRange(batch_idx, batch_idx + 1));
	w_1_1_grad_.copyTo(w_1_1_grad_batch_.rowRange(batch_idx, batch_idx + 1));	
	w_1_2_grad_.copyTo(w_1_2_grad_batch_.rowRange(batch_idx, batch_idx + 1));	
}

// reshape the gradients in a certain way to facilitate fisher information estimation
cv::Mat Transform::AggregateGradients(int current_dim)
{
	cv::Mat tmp;
	cv::Mat w_grad_tmp = cv::Mat::zeros(current_dim * 6, output_dim_, CV_64F);
	tmp = w_0_0_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 0, i * 6 + 0 + 1));
	tmp = w_0_1_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 1, i * 6 + 1 + 1));
	tmp = w_0_2_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 2, i * 6 + 2 + 1));
	tmp = w_1_0_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 3, i * 6 + 3 + 1));
	tmp = w_1_1_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 4, i * 6 + 4 + 1));
	tmp = w_1_2_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 5, i * 6 + 5 + 1));

	return w_grad_tmp;
}

void Transform::UpdateWeightBatch(int iter, int current_dim)
{
	cv::reduce(w_0_0_grad_batch_, w_0_0_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_0_1_grad_batch_, w_0_1_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_0_2_grad_batch_, w_0_2_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_1_0_grad_batch_, w_1_0_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_1_1_grad_batch_, w_1_1_grad_, 0, CV_REDUCE_AVG);
	cv::reduce(w_1_2_grad_batch_, w_1_2_grad_, 0, CV_REDUCE_AVG);

	/************** normal gradient ********************/

	w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - w_0_0_rate_ * w_0_0_grad_;
	w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - w_0_1_rate_ * w_0_1_grad_;
	w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - w_0_2_rate_ * w_0_2_grad_;
	w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - w_1_0_rate_ * w_1_0_grad_;
	w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - w_1_1_rate_ * w_1_1_grad_;
	w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - w_1_2_rate_ * w_1_2_grad_;

	/************** natural gradient ******************/

	//cv::Mat curr_grad = AggregateGradients(current_dim);
	//cv::Mat curr_fisher = cv::Mat::zeros(current_dim * 6, current_dim * 6, CV_64F);
	//fisher_inv_(cv::Rect(0, 0, current_dim * 6, current_dim * 6)).copyTo(curr_fisher);
	//cv::Mat tmpv = curr_grad.t() * curr_fisher * curr_grad;
	//double tmp_value = tmpv.at<double>(0, 0);
	////
	//curr_fisher = (1 / (1 - epsilon_)) * (curr_fisher - epsilon_ / (1- epsilon_ + epsilon_ * tmp_value) * curr_fisher * (curr_grad * curr_grad.t()) * curr_fisher);
	//cv::Mat curr_natural_grad = curr_fisher * curr_grad;
	//curr_fisher.copyTo(fisher_inv_(cv::Rect(0, 0, current_dim * 6, current_dim * 6)));
	//double curr_norm = cv::norm(curr_natural_grad, cv::NORM_L2);
	//if(iter == 1)
	//{
	//	ini_norm_ = curr_norm;
	//	average_norm_ = curr_norm;       
	//}
	//else    
	//	average_norm_ = (1 - lambda_) * average_norm_ + lambda_ * curr_norm; 


	//cv::Mat tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 0, i * 6 + 0 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - (w_0_0_rate_ * ini_norm_ / average_norm_) * tmp;

	//tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 1, i * 6 + 1 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - (w_0_1_rate_ * ini_norm_ / average_norm_) * tmp;

	//tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 2, i * 6 + 2 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - (w_0_2_rate_ * ini_norm_ / average_norm_) * tmp;

	//tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 3, i * 6 + 3 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - (w_1_0_rate_ * ini_norm_ / average_norm_) * tmp;

	//tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 4, i * 6 + 4 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - (w_1_1_rate_ * ini_norm_ / average_norm_) * tmp;
	//
	//tmp = cv::Mat::zeros(current_dim, output_dim_, CV_64F);
	//for(int i = 0; i < current_dim; i++)
	//	curr_natural_grad.rowRange(i * 6 + 5, i * 6 + 5 + 1).copyTo(tmp.rowRange(i, i + 1));
	//w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - (w_1_2_rate_ * ini_norm_ / average_norm_) * tmp;




	//w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - (w_0_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(0, current_dim).t();
	//w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - (w_0_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(current_dim, 2 * current_dim).t();
	//w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - (w_0_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(2 * current_dim, 3 * current_dim).t();
	//w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - (w_1_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(3 * current_dim, 4 * current_dim).t();
	//w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - (w_1_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(4 * current_dim, 5 * current_dim).t();
	//w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - (w_1_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(5 * current_dim, 6 * current_dim).t();

	/**************************************/

	//tmp_ = w_0_0_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(0, input_dim_));
	//tmp_ = w_0_1_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(input_dim_, 2 * input_dim_));
	//tmp_ = w_0_2_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(2 * input_dim_, 3 * input_dim_));
	//tmp_ = w_1_0_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(3 * input_dim_, 4 * input_dim_));
	//tmp_ = w_1_1_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(4 * input_dim_, 5 * input_dim_));
	//tmp_ = w_1_2_grad_.t();
	//tmp_.copyTo(w_grad_.rowRange(5 * input_dim_, 6 * input_dim_));

	//// double epsilon = 1e-6; // 6e-7;
	//tmp_ = w_grad_.t() * fisher_inv_ * w_grad_;
	//double tmp_value = tmp_.at<double>(0, 0);
	//
	//fisher_inv_ = (1 / (1 - epsilon_)) * (fisher_inv_ - epsilon_ / (1- epsilon_ + epsilon_ * tmp_value) * fisher_inv_ * (w_grad_ * w_grad_.t()) * fisher_inv_);
	//natural_grad_ = fisher_inv_ * w_grad_;

	//// tmp_grad_ = w_grad_;

	//w_0_0_ = w_0_0_ - w_0_0_rate_ * natural_grad_.rowRange(0, input_dim_).t();
	//w_0_1_ = w_0_1_ - w_0_1_rate_ * natural_grad_.rowRange(input_dim_, 2 * input_dim_).t();
	//w_0_2_ = w_0_2_ - w_0_2_rate_ * natural_grad_.rowRange(2 * input_dim_, 3 * input_dim_).t();
	//w_1_0_ = w_1_0_ - w_1_0_rate_ * natural_grad_.rowRange(3 * input_dim_, 4 * input_dim_).t();
	//w_1_1_ = w_1_1_ - w_1_1_rate_ * natural_grad_.rowRange(4 * input_dim_, 5 * input_dim_).t();
	//w_1_2_ = w_1_2_ - w_1_2_rate_ * natural_grad_.rowRange(5 * input_dim_, 6 * input_dim_).t();

	/*******************************************/

	//cv::Mat tmp;
	//cv::Mat w_grad_tmp = cv::Mat::zeros(current_dim * 6, output_dim_, CV_64F);
	//tmp = w_0_0_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(0, current_dim));
	//tmp = w_0_1_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(current_dim, 2 * current_dim));
	//tmp = w_0_2_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(2 * current_dim, 3 * current_dim));
	//tmp = w_1_0_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(3 * current_dim, 4 * current_dim));
	//tmp = w_1_1_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(4 * current_dim, 5 * current_dim));
	//tmp = w_1_2_grad_.t();
	//tmp.copyTo(w_grad_tmp.rowRange(5 * current_dim, 6 * current_dim));

	//if(current_dim !=  prev_dim_)
	//{
	//	// reset...
	//	fisher_inv_ = cv::Mat::eye(6 * current_dim, 6 * current_dim, CV_64F);
	//	prev_dim_ = current_dim;
	//}
	//// tmp_ = w_grad_.t() * fisher_inv_ * w_grad_;
	//tmp = w_grad_tmp.t() * fisher_inv_ * w_grad_tmp;
	//// double tmp_value = tmp_.at<double>(0, 0);	
	//double tmp_value = tmp.at<double>(0, 0);
	//std::cout << tmp_value << std::endl;
	//fisher_inv_ = (1 / (1 - epsilon_)) * (fisher_inv_ - epsilon_ / (1- epsilon_ + epsilon_ * tmp_value) * fisher_inv_ * (w_grad_tmp * w_grad_tmp.t()) * fisher_inv_);
	//// curr_fisher_inv = (1 / (1 - epsilon_)) * (curr_fisher_inv - epsilon_ / (1- epsilon_ + epsilon_ * tmp_value) * curr_fisher_inv * (w_grad_tmp * w_grad_tmp.t()) * curr_fisher_inv);
	//cv::Mat curr_natural_grad = fisher_inv_ * w_grad_tmp;
	//double curr_norm = cv::norm(curr_natural_grad, cv::NORM_L2);
 //   if(iter == 1)
 //   {
 //       ini_norm_ = curr_norm;
 //       average_norm_ = curr_norm;       
 //   }
 //   else    
 //       average_norm_ = (1 - lambda_) * average_norm_ + lambda_ * curr_norm; 

	//w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - (w_0_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(0, current_dim).t();
	//w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - (w_0_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(current_dim, 2 * current_dim).t();
	//w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - (w_0_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(2 * current_dim, 3 * current_dim).t();
	//w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - (w_1_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(3 * current_dim, 4 * current_dim).t();
	//w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - (w_1_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(4 * current_dim, 5 * current_dim).t();
	//w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - (w_1_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(5 * current_dim, 6 * current_dim).t();

	//w_0_0_ = w_0_0_ - w_0_0_rate_ * w_0_0_grad_;
	//w_0_1_ = w_0_1_ - w_0_1_rate_ * w_0_1_grad_;
	//w_0_2_ = w_0_2_ - w_0_2_rate_ * w_0_2_grad_;
	//w_1_0_ = w_1_0_ - w_1_0_rate_ * w_1_0_grad_;
	//w_1_1_ = w_1_1_ - w_1_1_rate_ * w_1_1_grad_;
	//w_1_2_ = w_1_2_ - w_1_2_rate_ * w_1_2_grad_;

	/*w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - w_0_0_rate_ * w_0_0_grad_;
	w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - w_0_1_rate_ * w_0_1_grad_;
	w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - w_0_2_rate_ * w_0_2_grad_;
	w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - w_1_0_rate_ * w_1_0_grad_;
	w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - w_1_1_rate_ * w_1_1_grad_;
	w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - w_1_2_rate_ * w_1_2_grad_;*/

	// std::cout << cv::norm(w_0_0_grad_) << " " << cv::norm(w_0_1_grad_) << " " << cv::norm(w_0_2_grad_) << " " << cv::norm(w_1_0_grad_) << " " << cv::norm(w_1_1_grad_) << " " << cv::norm(w_1_2_grad_) << std::endl;
	
	/*tmp_ = w_0_0_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(0, input_dim_));
	tmp_ = w_0_1_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(input_dim_, 2 * input_dim_));
	tmp_ = w_0_2_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(2 * input_dim_, 3 * input_dim_));
	tmp_ = w_1_0_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(3 * input_dim_, 4 * input_dim_));
	tmp_ = w_1_1_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(4 * input_dim_, 5 * input_dim_));
	tmp_ = w_1_2_grad_.t();
	tmp_.copyTo(w_grad_.rowRange(5 * input_dim_, 6 * input_dim_));*/

	/*cv::Mat tmp;
	cv::Mat w_grad_tmp = cv::Mat::zeros(current_dim * 6, output_dim_, CV_64F);
	tmp = w_0_0_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 0, i * 6 + 0 + 1));
	tmp = w_0_1_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 1, i * 6 + 1 + 1));
	tmp = w_0_2_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 2, i * 6 + 2 + 1));
	tmp = w_1_0_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 3, i * 6 + 3 + 1));
	tmp = w_1_1_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 4, i * 6 + 4 + 1));
	tmp = w_1_2_grad_.t();
	for(int i = 0; i < current_dim; i++)
		tmp.rowRange(i, i + 1).copyTo(w_grad_tmp.rowRange(i * 6 + 5, i * 6 + 5 + 1));
	*/


	// w_rate_ 

	/*tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 0, i * 6 + 0 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - (w_0_0_rate_ * ini_norm_ / average_norm_) * tmp;

	tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 1, i * 6 + 1 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - (w_0_1_rate_ * ini_norm_ / average_norm_) * tmp;

	tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 2, i * 6 + 2 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - (w_0_2_rate_ * ini_norm_ / average_norm_) * tmp;

	tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 3, i * 6 + 3 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - (w_1_0_rate_ * ini_norm_ / average_norm_) * tmp;

	tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 4, i * 6 + 4 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - (w_1_1_rate_ * ini_norm_ / average_norm_) * tmp;
	
	tmp = cv::Mat(current_dim, output_dim_, CV_64F);
	for(int i = 0; i < current_dim; i++)
		curr_natural_grad.rowRange(i * 6 + 5, i * 6 + 5 + 1).copyTo(tmp.rowRange(i, i + 1));
	w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - (w_1_2_rate_ * ini_norm_ / average_norm_) * tmp;

	curr_fisher_inv.copyTo(fisher_inv_(cv::Rect(0, 0, current_dim * 6, current_dim * 6)));*/
	//w_0_0_.colRange(0, current_dim) = w_0_0_.colRange(0, current_dim) - (w_0_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(0, current_dim).t();
	//w_0_1_.colRange(0, current_dim) = w_0_1_.colRange(0, current_dim) - (w_0_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(current_dim, 2 * current_dim).t();
	//w_0_2_.colRange(0, current_dim) = w_0_2_.colRange(0, current_dim) - (w_0_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(2 * current_dim, 3 * current_dim).t();
	//w_1_0_.colRange(0, current_dim) = w_1_0_.colRange(0, current_dim) - (w_1_0_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(3 * current_dim, 4 * current_dim).t();
	//w_1_1_.colRange(0, current_dim) = w_1_1_.colRange(0, current_dim) - (w_1_1_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(4 * current_dim, 5 * current_dim).t();
	//w_1_2_.colRange(0, current_dim) = w_1_2_.colRange(0, current_dim) - (w_1_2_rate_ * ini_norm_ / average_norm_) * curr_natural_grad.rowRange(5 * current_dim, 6 * current_dim).t();
	
	// tmp_grad_ = w_grad_;
	// double epsilon = 1e-6; // 6e-7;
}

cv::Mat Transform::fisher_inv()
{
	return fisher_inv_;
}

cv::Mat Transform::w_grad()
{
	return w_grad_;
}

cv::Mat Transform::natural_grad()
{
	return natural_grad_;
}

// copy to previous transformation
void Transform::CopyToPrev()
{
	transform_inv_.copyTo(prev_transform_inv_);
	//transform_.copyTo(prev_transform_);
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
	cv::Mat predicted_point = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat target_point = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	// numerical gradient
	cv::Mat transformed_point_delta_1 = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat transformed_point_delta_2 = cv::Mat::zeros(transform_dim_, 1, CV_64F);
	cv::Mat disturb = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	cv::Mat tmp_w;
	double e_1 = 0;
	double e_2 = 0;
	double disturb_value = 0.000001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;
	
	target_point.at<double>(0, 0) = -0.0044846991077065468; target_point.at<double>(1, 0) = -0.36011973023414612; target_point.at<double>(2, 0) = 1.0;	
	original_point.at<double>(0, 0) = -0.0020943940617144108; original_point.at<double>(1, 0) = -0.36954641342163086; original_point.at<double>(2, 0) = 1.0;
	// calculate the current feature vector
	CalcTransformInv(feature);	
	// calculate the transformed point
	predicted_point = TransformDataPointInv(original_point, 1);
	// calculate analytical gradient
	CalculateGradient(original_point, predicted_point, target_point, feature);
	// currently checking gradient for phi...
	for(int i = 0; i < input_dim_; i++)
	{
		disturb = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
		disturb.at<double>(0, i) = disturb_value;
		w_0_2_ = w_0_2_ + disturb;
		CalcTransformInv(feature);
		transformed_point_delta_1 = TransformDataPointInv(original_point, 1);
		w_0_2_ = w_0_2_ - 2 * disturb;
		CalcTransformInv(feature);
		transformed_point_delta_2 = TransformDataPointInv(original_point, 1);
		// reset back...
		w_0_2_ = w_0_2_ + disturb;
		CalcTransformInv(feature);
		predicted_point = TransformDataPointInv(original_point, 1);
		// calculate error
		double tx = target_point.at<double>(0, 0);
		double ty = target_point.at<double>(1, 0);
		double px1 = transformed_point_delta_1.at<double>(0, 0);
		double py1 = transformed_point_delta_1.at<double>(1, 0);
		double px2 = transformed_point_delta_2.at<double>(0, 0);
		double py2 = transformed_point_delta_2.at<double>(1, 0);

		e_1 = 1 - exp(-0.5 * GRADIENT_SCALE * ((tx - px1) * (tx - px1) + (ty - py1) * (ty - py1)));		
		e_2 = 1 - exp(-0.5 * GRADIENT_SCALE * ((tx - px2) * (tx - px2) + (ty - py2) * (ty - py2)));

		numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
		analytical_gradient = w_0_2_grad_.at<double>(0, i);

		std::cout << "iteration: " << i << std::endl;
		std::cout << "analytical gradient: " << analytical_gradient << " " << "numerical gradient: " << numerical_gradient << std::endl;
	}
	
}

cv::Mat Transform::w_0_0()
{
	return w_0_0_;
}

cv::Mat Transform::w_0_1()
{
	return w_0_1_;
}

cv::Mat Transform::w_0_2()
{
	return w_0_2_;
}

cv::Mat Transform::w_1_0()
{
	return w_1_0_;
}

cv::Mat Transform::w_1_1()
{
	return w_1_1_;
}

cv::Mat Transform::w_1_2()
{
	return w_1_2_;
}

cv::Mat Transform::transform_inv()
{
	return transform_inv_;
}

cv::Mat Transform::prev_transform_inv()
{
	return prev_transform_inv_;
}

void Transform::set_w_0_0(cv::Mat& w)
{
	 w.copyTo(w_0_0_);
}

void Transform::set_w_0_1(cv::Mat& w)
{
	w.copyTo(w_0_1_);
}

void Transform::set_w_0_2(cv::Mat& w)
{
	w.copyTo(w_0_2_);
}

void Transform::set_w_1_0(cv::Mat& w)
{
	w.copyTo(w_1_0_);
}

void Transform::set_w_1_1(cv::Mat& w)
{
	w.copyTo(w_1_1_);
}

void Transform::set_w_1_2(cv::Mat& w)
{
	w.copyTo(w_1_2_);
}

void Transform::SetLearningRates(double normal_rate, double natural_rate)
{
	w_0_0_rate_ = normal_rate;
	w_0_1_rate_ = normal_rate;
	w_0_2_rate_ = normal_rate;
	w_1_0_rate_ = normal_rate;
	w_1_1_rate_ = normal_rate;
	w_1_2_rate_ = normal_rate;

	epsilon_ = natural_rate;

}