// With google c++ coding style here
#include "../inc/explorer.h"

COLOUR GetColour(double v, double vmin, double vmax)
{
	COLOUR c = {1.0,1.0,1.0}; // white
	double dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;

	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) 
	{
		c.r = 0;
		c.g = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) 
	{
		c.r = 0;
		c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) 
	{
		c.r = 4 * (v - vmin - 0.5 * dv) / dv;
		c.b = 0;
	}
	else
	{
		c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.b = 0;
	}

	return c;
}

// constructor
Explorer::Explorer(int dir_id, int num_iteration, int expanding_period, int load_all_flag, char* dir)
{    
	dir_id_ = dir_id;
	dim_action_ = 6; // dimension of action
	dim_feature_ = 8; // 9; // 8; // 8; // 10;
	trend_number_ = 15; // 14
	num_iteration_ = num_iteration;
	range_expanding_period_ = expanding_period;
	char input_dir[400];
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/");
	strcat(input_dir, dir);
	strcat(input_dir, "/num_data.bin");
	cv::Mat num_data = cv::Mat::zeros(2, 1, CV_64F);
	FileIO::ReadMatDouble(num_data, 2, 1, input_dir);
	num_train_data_ = (int)num_data.at<double>(0, 0); // 16400 * 19 / 20; // 17100; // 20250; // 17100; // 20250; // 20880; // 20250; // 17280; // 9900; // 18000;
	num_test_data_ = (int)num_data.at<double>(1, 0); // 16400 / 20; // 1900; // 2250; // 1900; // 2250; // 2320; // 2250; // 1920; // 1100; // 2000;
	max_exploration_range_ = 1.0; // 1.00 // 1.41; // 0.71; // sqrt(2) should be fine... 0.5;
	starting_exploration_range_ = 0.02; // 0.2; // 0.02; // 0.002; // 0.101;
    starting_exploration_range_rotation_ = 0.04;
    max_exploration_range_rotation_ = 4.5;
	avg_cost_ = 0;		
	path_count_ = 0;
	rotation_target_ = 1.0;
	rotation_speed_ = 0.05; // 3.6 degrees...
	target_p_1_ = 0;
	target_p_2_ = 0;
	prev_target_p_1_ = 0;
	prev_target_p_2_ = 0;
	load_all_flag_ = load_all_flag;

	prop_dim_ = 2; // 3;
	action_ = cv::Mat::zeros(dim_action_, 1, CV_64F);	
	train_prop_ = cv::Mat::zeros(num_train_data_, prop_dim_, CV_64F);
	test_prop_ = cv::Mat::zeros(num_test_data_, prop_dim_, CV_64F);	
	center_value_ = 70.0; // 55.0; // 70.0;
	target_ = cv::Mat::zeros(1, prop_dim_, CV_64F);
	prev_target_ = cv::Mat::ones(1, prop_dim_, CV_64F);
	prev_target_ = prev_target_ * center_value_;


	home_prop_ = cv::Mat::zeros(1, 2, CV_64F);	
	img_point_ = cv::Mat::zeros(3, 1, CV_64F);			img_point_.at<double>(2, 0) = 1; // image frame
	prev_img_point_ = cv::Mat::zeros(3, 1, CV_64F);		prev_img_point_.at<double>(2, 0) = 1; // image frame
	home_point_ = cv::Mat::zeros(3, 1, CV_64F);			home_point_.at<double>(2, 0) = 1; // reference frame
	prev_home_point_ = cv::Mat::zeros(3, 1, CV_64F);	prev_home_point_.at<double>(2, 0) = 1; // reference frame	
	curr_prop_ = cv::Mat::zeros(1, prop_dim_, CV_64F); 
	curr_prop_matrix_ = cv::Mat::zeros(num_train_data_, prop_dim_, CV_64F);
	prop_diff_ = cv::Mat::zeros(num_train_data_, prop_dim_, CV_64F);
	prop_dist_ = cv::Mat::zeros(num_train_data_, 1, CV_64F);
	aim_idx_matrix_ = cv::Mat::zeros(num_train_data_, 1, CV_32S);
	feature_data_ = cv::Mat::zeros(2, 1, CV_64F);
	feature_home_ = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	upper_bound_ = 80.0; // 70.0; // 80.0;
	lower_bound_ = 60.0; // 40.0; // 60.0;
	double sin_1 = abs(sin(upper_bound_ / 180.0 * PI) - sin(center_value_ / 180.0 * PI));
	double sin_2 = abs(sin(lower_bound_ / 180.0 * PI) - sin(center_value_ / 180.0 * PI));
	sin_scale_ = sin_1 > sin_2 ? sin_1 : sin_2;

	double cos_1 = abs(cos(upper_bound_  / 180.0 * PI) - cos(center_value_ / 180.0 * PI));
	double cos_2 = abs(cos(lower_bound_ / 180.0 * PI) - cos(center_value_ / 180.0 * PI));
	cos_scale_ = cos_1 > cos_2 ? cos_1 : cos_2;
	// SetFeatureSinusoidal(feature_home_); need to be reset back...

	rs_indices_ = cv::Mat::zeros(num_train_data_, 1, CV_32S);
	rs_dists_ = cv::Mat::zeros(num_train_data_, 1, CV_32F);

	sprintf(dir_, dir);

	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/input/elips_ini_%d.bin", dir_id_);
	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);
	FileIO::ReadMatDouble(elips_ini, 5, 1, input_dir);

	initial_x_ = elips_ini.at<double>(0, 0); // 380; // 300; // 218; 
	initial_y_ = elips_ini.at<double>(1, 0); // 220; // 260; // 230; 
	initial_long_axis_ = elips_ini.at<double>(2, 0); 
	initial_short_axis_ = elips_ini.at<double>(3, 0); 
	initial_phi_ = elips_ini.at<double>(4, 0); // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	
	image_width_ = 640;
	image_height_ = 480;
	ini_translate_ = cv::Mat::eye(3, 3, CV_64F);
	ini_rotation_ = cv::Mat::eye(3, 3, CV_64F);
	ini_scale_ = cv::Mat::eye(3, 3, CV_64F);

	ini_transformation_ = cv::Mat::eye(3, 3, CV_64F);

	// set initial transformation
	// suppose range is 0 to image_width_ - 1 and 0 to image_height_ - 1
	ini_translate_.at<double>(0, 2) = -1; ini_translate_.at<double>(1, 2) = -1; // maybe change starting point 1 to 0...
	ini_scale_.at<double>(0, 0) = 2 / (image_width_ - 1); ini_scale_.at<double>(1, 1) = 2 / (image_height_ - 1);
	ini_transformation_ = ini_translate_ * ini_rotation_ * ini_scale_;
	threashold_ = 1 - 1e-5;

	actual_feature_dim_ = 8;
}

Explorer::~Explorer(){
}

// get polynomial kernel
void Explorer::SetKernel(fL& kernel_list, cv::Mat& data, double* p_current_data, int dim_left, int curr_pos, int data_length, int kernel_dim, int value_flag)
{
	int pos = curr_pos;
	for(int dim = 0; dim <= dim_left; dim++)
	{
		if(pos == 0 && dim == 0)
		{			
			kernel_list.push_back(0.0); // bias
			int next_pos = pos + 1;
			double tmp_data = *p_current_data;
			if(kernel_dim != 0)
				SetKernel(kernel_list, data, p_current_data, kernel_dim , next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
		else if(dim == 0)
		{
			int next_pos = pos < data_length - 1 ? pos + 1 : pos;
			double tmp_data = *p_current_data;
			int actual_dim_left = dim_left - dim;
			if(kernel_dim != 0 && pos != next_pos)
				SetKernel(kernel_list, data, p_current_data, actual_dim_left , next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
		else if(dim != 0)
		{
			*p_current_data = (*p_current_data) * data.at<double>(pos, 0); // pow(data[pos], (double)dim);
			kernel_list.push_back(*p_current_data);
			int next_pos = pos < data_length - 1 ? pos + 1 : pos;
			int actual_dim_left = dim_left - dim;
			double tmp_data = *p_current_data;
			if(actual_dim_left != 0 && pos != next_pos)
				SetKernel(kernel_list, data, p_current_data, actual_dim_left, next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
	}
}
// get policy state
// it is the aim idx rather than the frame index because the prop loaded is sorted...
void Explorer::SetFeature(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop)
{
	double delta_p1 = prop.at<double>(aim_idx, 0) - home_prop.at<double>(0, 0);
	double delta_p2 = prop.at<double>(aim_idx, 1) - home_prop.at<double>(0, 1);
	
	double current_data = 1.0;
	double* p_current_data = &current_data;
	int kernel_dim = 3;
	int curr_pos = 0;
	double shift = 0.0;
	kernel_list_.clear();
	feature_data_.at<double>(0, 0) = delta_p1 + shift;
	feature_data_.at<double>(1, 0) = delta_p2 + shift;
	SetKernel(kernel_list_, feature_data_, p_current_data, kernel_dim, curr_pos, 2, kernel_dim, 0); 
	double sum = 0;
	for(int i = 0; i < kernel_list_.size(); i++)
		feature.at<double>(i, 0) = kernel_list_[i];
}
//
//void Explorer::SetFeatureSinusoidal(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop)
//{
//	double delta_p1 = prop.at<double>(aim_idx, 0) - home_prop.at<double>(0, 0);
//	double delta_p2 = prop.at<double>(aim_idx, 1) - home_prop.at<double>(0, 1);
//	cv::Mat p1 = cv::Mat::zeros(3, 1, CV_64F);
//	cv::Mat p2 = cv::Mat::zeros(3, 1, CV_64F);
//	int dim_joint = 3;	
//	p1.at<double>(0, 0) = 1.0; 
//	p1.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 0) / 180.0 * PI) - sin(center_value_ / 180.0 * PI); 
//	p1.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 0) / 180.0 * PI) - cos(center_value_ / 180.0 * PI);
//	p2.at<double>(0, 0) = 1.0; 
//	p2.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 1) / 180.0 * PI) - sin(center_value_ / 180.0 * PI); 
//	p2.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 1) / 180.0 * PI) - cos(center_value_ / 180.0 * PI);
//
//	// feature = cv::Mat::zeros(, 1, CV_64F);
//	for(int i = 0; i < dim_joint; i++)
//		for(int j = 0; j < dim_joint; j++)
//			feature.at<double>(i * dim_joint + j, 0) = p1.at<double>(i, 0) * p2.at<double>(j, 0);
//	feature.at<double>(0, 0) = 0.0;
//}

void Explorer::SetFeatureSinusoidal(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop)
{
	cv::Mat p1 = cv::Mat::zeros(3, 1, CV_64F);
	cv::Mat p2 = cv::Mat::zeros(3, 1, CV_64F);
	int dim_joint = 3;	

	p1.at<double>(0, 0) = 1.0; 
	p1.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 0) / 180.0 * PI); 
	p1.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 0) / 180.0 * PI);
	p2.at<double>(0, 0) = 1.0; 
	p2.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 1) / 180.0 * PI);
	p2.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 1) / 180.0 * PI);

	/*p1.at<double>(0, 0) = 1.0; 
	p1.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 0) / 180.0 * PI) - sin(center_value_ / 180.0 * PI); 
	p1.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 0) / 180.0 * PI) - cos(center_value_ / 180.0 * PI);
	p2.at<double>(0, 0) = 1.0; 
	p2.at<double>(1, 0) = sin(prop.at<double>(aim_idx, 1) / 180.0 * PI) - sin(center_value_ / 180.0 * PI);
	p2.at<double>(2, 0) = cos(prop.at<double>(aim_idx, 1) / 180.0 * PI) - cos(center_value_ / 180.0 * PI);*/

	/*feature.at<double>(0, 0) = p1.at<double>(0, 0) * p2.at<double>(2, 0);
	feature.at<double>(1, 0) = p1.at<double>(2, 0) * p2.at<double>(0, 0);
	feature.at<double>(2, 0) = p1.at<double>(1, 0) * p2.at<double>(1, 0);
	feature.at<double>(3, 0) = p1.at<double>(1, 0) * p2.at<double>(2, 0);
	feature.at<double>(4, 0) = p1.at<double>(2, 0) * p2.at<double>(1, 0);
	feature.at<double>(5, 0) = p1.at<double>(2, 0) * p2.at<double>(2, 0);*/


	// feature = cv::Mat::zeros(, 1, CV_64F);
	for(int i = 0; i < dim_joint; i++) {
		for(int j = 0; j < dim_joint; j++) {
			// if(!(i == 0 && j == 0))
			// feature.at<double>(i * dim_joint + j, 0) = p1.at<double>(i, 0) * p2.at<double>(j, 0);
			if(!(i == 0 && j == 0))
				feature.at<double>(i * dim_joint + j - 1, 0) = p1.at<double>(i, 0) * p2.at<double>(j, 0);
		}
	}
	feature = feature - feature_home_;
	// feature.at<double>(0, 0) = 0.01;
	// feature = feature.mul(scale);
	// double a = feature.at<double>(3, 0);
	// double b = feature.at<double>(2, 0);
}

void Explorer::SetFeatureSinusoidal(cv::Mat& home_feature)
{
	cv::Mat p1 = cv::Mat::zeros(3, 1, CV_64F);
	cv::Mat p2 = cv::Mat::zeros(3, 1, CV_64F);
	int dim_joint = 3;	
	p1.at<double>(0, 0) = 1.0; 
	p1.at<double>(1, 0) = sin(center_value_ / 180.0 * PI); 
	p1.at<double>(2, 0) = cos(center_value_ / 180.0 * PI);
	p2.at<double>(0, 0) = 1.0; 
	p2.at<double>(1, 0) = sin(center_value_ / 180.0 * PI); 
	p2.at<double>(2, 0) = cos(center_value_ / 180.0 * PI);

	// feature = cv::Mat::zeros(, 1, CV_64F);

	/*home_feature.at<double>(0, 0) = p1.at<double>(0, 0) * p2.at<double>(2, 0);
	home_feature.at<double>(1, 0) = p1.at<double>(2, 0) * p2.at<double>(0, 0);
	home_feature.at<double>(2, 0) = p1.at<double>(1, 0) * p2.at<double>(1, 0);
	home_feature.at<double>(3, 0) = p1.at<double>(1, 0) * p2.at<double>(2, 0);
	home_feature.at<double>(4, 0) = p1.at<double>(2, 0) * p2.at<double>(1, 0);
	home_feature.at<double>(5, 0) = p1.at<double>(2, 0) * p2.at<double>(2, 0);*/
	for(int i = 0; i < dim_joint; i++) {
		for(int j = 0; j < dim_joint; j++) {
			// if(i != 0 || j != 0)
			// if(!(i == 0 && j == 0))
			// home_feature.at<double>(i * dim_joint + j, 0) = p1.at<double>(i, 0) * p2.at<double>(j, 0);
			if(!(i == 0 && j == 0))
				home_feature.at<double>(i * dim_joint + j - 1, 0) = p1.at<double>(i, 0) * p2.at<double>(j, 0);
		}
	}
}

int Explorer::GenerateAimIndexKDTree(std::mt19937& engine, cv::flann::Index& kd_trees, std::vector<int>& path, int current_iteration, bool* update_flag)
{
	int aim_idx = 0;
	double current_range = 0;
    // double current_range_rotation = 0;
	double scale = 10; // 15; // 10; // add hoc fix of the new scaling problem due to change of sinusoidal basis
	double center = center_value_; // add hoc fix
	double max_speed = 0.6 * scale; // 0.4 * scale
    // double max_rotation_speed = 0.1; // 4.0 / 90.0 * 2.0; // 2.0 / 90.0 * 2.0
	double path_length = 0;
	// double radius = 0.2;
	int num_frame_path = 0;
	int prop_dim = train_prop_.cols;
    // double rotation_speed = 0;
	// rotation is for the wrist rotating experiment...
	// for normal exploration we don't need this
	// generate path
	if(path.size() == 0)
	{		
        // planar exploration range
		current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
		current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
		std::uniform_real_distribution<double> uniform(-scale * current_range + center, scale * current_range + center);  	
        // rotation exploration range
        /*current_range_rotation = starting_exploration_range_rotation_ + (max_exploration_range_rotation_ - starting_exploration_range_rotation_) * current_iteration / range_expanding_period_;	
		current_range_rotation = current_range_rotation > max_exploration_range_rotation_ ? max_exploration_range_rotation_ : current_range_rotation;
		std::uniform_real_distribution<double> uniform_rotation(-1.0 * current_range_rotation, 1.0 * current_range_rotation);	  */
        // first attempt to get a target	
        target_.at<double>(0, 0) = uniform(engine);	
		target_.at<double>(0, 1) = uniform(engine); 
		// target_.at<double>(0, 2) = uniform_rotation(engine);
		path_length = cv::norm(target_.colRange(0, 2) - prev_target_.colRange(0, 2), cv::NORM_L2);
		num_frame_path = (int)(path_length / max_speed) + 1;
        //rotation_speed = abs(target_.at<double>(0, 2) - prev_target_.at<double>(0, 2)) / num_frame_path;
        // if target doesn't satisfy the requirement
   //     while(rotation_speed > max_rotation_speed)
   //     {
   //         target_.at<double>(0, 0) = uniform(engine);
			//target_.at<double>(0, 1) = uniform(engine); 
			//// target_.at<double>(0, 2) = uniform_rotation(engine);
   //         path_length = cv::norm(target_.colRange(0, 2) - prev_target_.colRange(0, 2), cv::NORM_L2);
   //         num_frame_path = (int)(path_length / max_speed) + 1;
   //         rotation_speed = abs(target_.at<double>(0, 2) - prev_target_.at<double>(0, 2)) / num_frame_path;
   //     }

		path.clear();
		path_update_vector_.clear();
		// std::vector<double> prev_rotation(num_frame_path);
		for(int i = 1; i <= num_frame_path; i++)
		{
			cv::Mat tmp_target = cv::Mat::zeros(1, prop_dim, CV_64F);
			cv::Mat tmp_data = cv::Mat::zeros(1, prop_dim, CV_64F);			
			tmp_target = prev_target_ + (target_ - prev_target_) * i / num_frame_path;
			tmp_target.convertTo(tmp_target, CV_32F);
			kd_trees.knnSearch(tmp_target, rs_indices_, rs_dists_, 1, cv::flann::SearchParams(64));
            bool curr_update_flag = true;
			path_update_vector_.push_back(curr_update_flag);
			path.push_back(rs_indices_.at<int>(0, 0));			
		}	
		target_.copyTo(prev_target_);
	}		
	aim_idx = path[0];
	*update_flag = path_update_vector_[0];
	path.erase(path.begin());		
	path_update_vector_.erase(path_update_vector_.begin());
	return aim_idx;
}


void Explorer::EvaluateGradientAndUpdate(cv::Mat& feature, int update_flag, int aim_idx, Ellipse& elips, int iteration_count)
{	
	int prev_match_idx = 0;
	int aim_match_idx = 0;		
	int match_point_count = 0;
	int gradient_match_point_count = 0; // the points wihin the small radius, typically 1, used to calculate the batch gradient... radius is set to 4 normally...
	int gradient_batch_idx = 0;
	int match_point_idx = 0;
	int check_idx = 0;	
	int match_point_info_size = 15; // 13;	
	double curr_diff = 0;
	double predicted_motion_ratio = 0;
	match_point_info_.clear();
    unique_match_point_.clear();
	cv::Mat predicted_prev_img_point;
	cv::Mat predicted_img_point;
	
	cv::Mat transform = cv::Mat::zeros(3, 3, CV_64F);
	if(elips_descriptors_.rows != 0)
		matcher_.match(elips_prev_descriptors_, elips_descriptors_, matches_);	
	else
		return;

	if(matches_.size() == 0)
		return;
	// get raw reward
	for(int i = 0; i < matches_.size(); i++)
	{
		// base_match_idx = matches_[i].queryIdx;
		prev_match_idx = matches_[i].queryIdx;
		aim_match_idx = matches_[i].trainIdx;
		// key point location	
		prev_img_point_.at<double>(0, 0) = elips_prev_key_points_.at<float>(prev_match_idx, 0);
		prev_img_point_.at<double>(1, 0) = elips_prev_key_points_.at<float>(prev_match_idx, 1);	
		prev_img_point_.at<double>(2, 0) = 1.0;
		img_point_.at<double>(0, 0) = elips_key_points_.at<float>(aim_match_idx, 0);
		img_point_.at<double>(1, 0) = elips_key_points_.at<float>(aim_match_idx, 1);
		img_point_.at<double>(2, 0) = 1.0;
		match_point_info_.push_back(cv::Mat::zeros(1, match_point_info_size, CV_64F));
		// inverse transform to compare with target point in reference frame...
		prev_home_point_ = elips.transform_.TransformDataPointInv(prev_img_point_, 0);
		home_point_ = elips.transform_.TransformDataPointInv(img_point_, 1); // transform image coordinate to reference coordinate				
		predicted_prev_img_point = elips.transform_.TransformToPreviousFrame(img_point_);				
		// predicted_img_point = elips.transform_.TransformToNextFrame(prev_home_point_);	
		predicted_motion_ratio = 1 - cv::norm(prev_img_point_ - predicted_prev_img_point) / cv::norm(img_point_ - prev_img_point_);
		// predicted_motion_ratio = 1 - cv::norm(img_point_ - predicted_img_point) / cv::norm(img_point_ - prev_img_point_);
		predicted_motion_ratio = predicted_motion_ratio < 0 ? 0 : predicted_motion_ratio;
		
		curr_diff = 1 - exp(-0.5 * GRADIENT_SCALE * (pow(prev_home_point_.at<double>(0, 0) - home_point_.at<double>(0, 0), 2) + pow(prev_home_point_.at<double>(1, 0) - home_point_.at<double>(1, 0), 2)));		
		// curr_diff = 1 - exp(-0.5 * GRADIENT_SCALE * (pow(predicted_img_point.at<double>(0, 0) - img_point_.at<double>(0, 0), 2) + pow(predicted_img_point.at<double>(1, 0) - img_point_.at<double>(1, 0), 2)));		
		match_point_info_[i].at<double>(0, 0) = curr_diff; // reward		
		match_point_info_[i].at<double>(0, 1) = prev_match_idx; match_point_info_[i].at<double>(0, 2) = aim_match_idx; // base idx and aim idx		
		match_point_info_[i].at<double>(0, 3) = prev_img_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 4) = prev_img_point_.at<double>(1, 0);
		match_point_info_[i].at<double>(0, 5) = img_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 6) = img_point_.at<double>(1, 0); 
		match_point_info_[i].at<double>(0, 7) = prev_home_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 8) = prev_home_point_.at<double>(1, 0); 
		match_point_info_[i].at<double>(0, 9) = home_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 10) = home_point_.at<double>(1, 0); 
		// match_point_info_[i].at<double>(0, 9) = predicted_img_point.at<double>(0, 0); match_point_info_[i].at<double>(0, 10) = predicted_img_point.at<double>(1, 0); 
		match_point_info_[i].at<double>(0, 11) = elips_distance_.at<double>(aim_match_idx, 0);
		match_point_info_[i].at<double>(0, 12) = predicted_motion_ratio;	
		match_point_info_[i].at<double>(0, 13) = home_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 14) = home_point_.at<double>(1, 0); 
   
	}
	sort(match_point_info_.begin(), match_point_info_.end(), DistCompare());		
	avg_cost_ = 0;
	unique_match_point_.push_back(cv::Mat::zeros(1, match_point_info_size, CV_64F));	
	for(int i = 0; i < match_point_info_size; i++)
		unique_match_point_[match_point_count].at<double>(0, i) = match_point_info_[match_point_idx].at<double>(0, i); // assign first one
	if(unique_match_point_[match_point_count].at<double>(0, 11) <= 1.0)
	{
		avg_cost_ += unique_match_point_[match_point_count].at<double>(0, 0);
		gradient_match_point_count++;
	}					
	match_point_count++;	
	// either base point equal or aim point equal or base idx equal or aim idx equal, break...
	for(match_point_idx = 1; match_point_idx < matches_.size(); match_point_idx++)
	{
		for(check_idx = 0; check_idx < match_point_count; check_idx++)
		{
			if(match_point_info_[match_point_idx].at<double>(0, 1) == unique_match_point_[check_idx].at<double>(0, 1) || // base idx equal
				match_point_info_[match_point_idx].at<double>(0, 2) == unique_match_point_[check_idx].at<double>(0, 2) || // aim idx equal
				(match_point_info_[match_point_idx].at<double>(0, 3) == unique_match_point_[check_idx].at<double>(0, 3) &&
				match_point_info_[match_point_idx].at<double>(0, 4) == unique_match_point_[check_idx].at<double>(0, 4)) ||
				(match_point_info_[match_point_idx].at<double>(0, 5) == unique_match_point_[check_idx].at<double>(0, 5) &&
				match_point_info_[match_point_idx].at<double>(0, 6) == unique_match_point_[check_idx].at<double>(0, 6)))
			{
				break;
			}
		}
		if(check_idx == match_point_count)
		{
			unique_match_point_.push_back(cv::Mat::zeros(1, match_point_info_size, CV_64F));
			for(int i = 0; i < match_point_info_size; i++)
				unique_match_point_[match_point_count].at<double>(0, i) = match_point_info_[match_point_idx].at<double>(0, i);			
			if(unique_match_point_[match_point_count].at<double>(0, 11) <= 1.0)
			{
				avg_cost_ += unique_match_point_[match_point_count].at<double>(0, 0);
				gradient_match_point_count++;
			}
			match_point_count++;
			// if(match_point_count > 11)
			// 	break;
		}
	}
	// calculate average cost	
	if(gradient_match_point_count != 0)
		avg_cost_ = avg_cost_ / gradient_match_point_count;
	if(update_flag)
	{
		if(gradient_match_point_count != 0)
		{
			for(int i = 0; i < match_point_count; i++)
			{
				if(unique_match_point_[i].at<double>(0, 11) <= 1.0)
				{
					prev_home_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 7); prev_home_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 8);
					prev_home_point_.at<double>(2, 0) = 1.0;
					img_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 5); img_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 6);
					img_point_.at<double>(2, 0) = 1.0;
					/*predicted_img_point.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); predicted_img_point.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);
					predicted_img_point.at<double>(2, 0) = 1.0;*/
					home_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); home_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);
					home_point_.at<double>(2, 0) = 1.0;
					// inverse transform
					// elips.transform_.CalcMiniBatchInvGradient(prev_home_point_, predicted_img_point, img_point_, feature, gradient_match_point_count, gradient_batch_idx);	
					elips.transform_.CalcMiniBatchInvGradient(img_point_, home_point_, prev_home_point_, feature, gradient_match_point_count, gradient_batch_idx);	
					// CalcMiniBatchInvGradient(cv::Mat& original_point, cv::Mat& transformed_point, cv::Mat& target_point, cv::Mat& feature, int batch_count, int batch_idx)
					gradient_batch_idx++;
				}
			}	
			// update weights
			elips.transform_.UpdateWeightBatch(iteration_count, dim_feature_);
			// classify points in reference frame
			elips.ClassifyPointsHomeEllipse(unique_match_point_, matched_points_, motion_ratio_, maha_dist_);
			// update ellipse by the matched points
			// elips.UpdateHomeEllipse(matched_points_, motion_ratio_, maha_dist_);
		}
	}
	else
	{
		/*if(aim_idx % 1 == 1)
		{*/
			elips.ClassifyPointsHomeEllipse(unique_match_point_, matched_points_, motion_ratio_, maha_dist_);

			// elips.UpdateHomeEllipse(matched_points_, motion_ratio_, maha_dist_, &initial_x_, &initial_y_, &initial_phi_, &initial_long_axis_, &initial_short_axis_);

			char key_points_dir[300];
			sprintf(key_points_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/key_points/key_point_data_%d.bin", dir_id_, aim_idx);				
			FileIO::WriteMatDouble(matched_points_, matched_points_.rows, matched_points_.cols, key_points_dir);

			sprintf(key_points_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/key_points/motion_data_%d.bin", dir_id_, aim_idx);				
			FileIO::WriteMatDouble(motion_ratio_, motion_ratio_.rows, motion_ratio_.cols, key_points_dir);

			sprintf(key_points_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/key_points/maha_dist_%d.bin", dir_id_, aim_idx);				
			FileIO::WriteMatDouble(maha_dist_, maha_dist_.rows, maha_dist_.cols, key_points_dir);

			cv::Mat key_points_data_length = cv::Mat::zeros(1, 1, CV_64F);
			key_points_data_length.at<double>(0, 0) = matched_points_.rows;
			sprintf(key_points_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/key_points/data_count_%d.bin", dir_id_, aim_idx);
			FileIO::WriteMatDouble(key_points_data_length, 1, 1, key_points_dir);

			cv::Mat inv_transform = cv::Mat::zeros(3, 3, CV_64F);
			inv_transform = elips.transform_.transform_inv();
			sprintf(key_points_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/key_points/inv_transform_%d.bin", dir_id_, aim_idx);
			FileIO::WriteMatDouble(inv_transform, 3, 3, key_points_dir);
		//}

	}	
}

void Explorer::CopyToPrev()
{	
	elips_prev_key_points_ = elips_key_points_.clone();
	elips_prev_descriptors_ = elips_descriptors_.clone();
}

void Explorer::Train()
{		
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int display_interval = 500;
	int write_trend_interval = 5000;
	int write_diagnosis_interval = 5000;
	int percentage = 0; 		
	unsigned long iteration_count = 0;		
	bool append_flag = false;	
	double radius = 3.0;
	bool gradient_update_flag = false;

	max_exploration_range_ = 1.0;	

	// keep the exploration in a small range...	
	std::mt19937 engine(rd_());		
	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);		
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);	
	cv::Mat train_feature_mat = cv::Mat::zeros(num_train_data_, dim_feature_, CV_64F);			
	cv::Mat test_feature_mat = cv::Mat::zeros(num_test_data_, dim_feature_, CV_64F);	
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);		
	cv::Mat feature_home = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	// SetFeatureSinusoidal(feature_home);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);	
	cv::Mat cov = cv::Mat::zeros(2, 2, CV_64F);
	cv::Mat mu = cv::Mat::zeros(2, 1, CV_64F);
	cv::Mat inv_transform = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat transform = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat improvement_avg = cv::Mat::zeros(1, 1, CV_64F);
	// cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	fL* trend_array = new fL[trend_number_];
	std::vector<int> path(0);
	// objects...	
	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x_, initial_y_, initial_long_axis_, initial_short_axis_, initial_phi_, radius, ini_transformation_);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	// loader.FormatWeightsForDiagnosisDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);	
	loader.LoadLearningRates(elips);	
	loader.LoadPropSinusoidalFeature(train_feature_mat, test_feature_mat);
	// for tool extension experiment only...
	/*loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);
	loader.LoadEllipse(elips);	
	elips.set_a_home(1);
	elips.set_inv_a_home();
	elips.set_a();
	elips.set_inv_a();	*/


	/*loader.loadweightsfortest(elips.transform_, 1, dim_feature_);
	loader.loadellipse(elips);
	elips.setrefellipseparameters();*/
	// ********************************* //


	// establish the kd tree structure
	cv::Mat train_prop_float = cv::Mat::zeros(train_prop_.rows, train_prop_.cols, CV_32F);
	train_prop_.convertTo(train_prop_float, CV_32F); // just use two joints...
	cv::flann::Index kd_trees(train_prop_float, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree			
	
	// ***** try load all key points...******** //
	descriptors_all_.clear(); key_points_all_.clear();
	int shift = 0;//10;
	if(load_all_flag_)
		loader.LoadAllSiftKeyPoint(descriptors_all_, key_points_all_, 0 + shift, num_train_data_ + num_test_data_ + shift, ini_transformation_); // 23248); // num_train_data_ + num_test_data_);
	// main loop	
	for(iteration_count = 0; iteration_count < num_iteration_; iteration_count++)
	{				
		// aim_idx = GenerateAimIndexLinePath(engine, iteration_count);
		// aim_idx = GenerateAimIndex(iteration_count);
		// aim_frame_idx = aim_idx;
		aim_idx = GenerateAimIndexKDTree(engine, kd_trees, path, iteration_count, &gradient_update_flag);						
		aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);		

		if(!load_all_flag_)
		{
			loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx, ini_transformation_);	
		}
		else
		{
			if(aim_frame_idx < descriptors_all_.size() + shift && aim_frame_idx >= shift)
			{
				descriptors_ = descriptors_all_[aim_frame_idx - shift];	
				key_points_ = key_points_all_[aim_frame_idx - shift];
				// TransformKeyPointsMatrix(key_points_all_[aim_frame_idx], key_points_);
			}
			else
			{
				std::cout << "number of descriptor matrix incorrect... exit..." << std::endl;
				exit(0);
			}
		}
		// act & evaluate gradient
		if(iteration_count != 0)
		{
			// SetFeatureSinusoidal(feature, aim_idx, train_prop_, home_prop_);
			feature = train_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
			feature = feature.t();
			// OnlinePCA(feature, iteration_count, threashold_);
			// feature = feature - feature_home;	
			// feature dimension problem...
			elips.transform_.CalcTransformInv(feature);
			// elips.TransformEllipse();				
			elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);				
			EvaluateGradientAndUpdate(feature, 1, aim_idx, elips, iteration_count);					
		}	
		else
		{
			// only for the first iteration...
			elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
			feature = train_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
			feature = feature.t();
			// OnlinePCA(feature, iteration_count, threashold_);
		}		

		// copy to previous				
		elips.CopyToPrev();
		CopyToPrev();									
		cov = elips.home_cov();
		mu = elips.home_mu();

		
		// recording
		trend_array[0].push_back(cv::norm(elips.transform_.w_0_0(), cv::NORM_L2)); trend_array[1].push_back(cv::norm(elips.transform_.w_0_1(), cv::NORM_L2));      
		trend_array[2].push_back(cv::norm(elips.transform_.w_0_2(), cv::NORM_L2)); trend_array[3].push_back(cv::norm(elips.transform_.w_1_0(), cv::NORM_L2));      
		trend_array[4].push_back(cv::norm(elips.transform_.w_1_1(), cv::NORM_L2)); trend_array[5].push_back(cv::norm(elips.transform_.w_1_2(), cv::NORM_L2));		
		trend_array[6].push_back(cov.at<double>(0, 0)); trend_array[7].push_back(cov.at<double>(0, 1));	
		trend_array[8].push_back(cov.at<double>(1, 0)); trend_array[9].push_back(cov.at<double>(1, 1));
		trend_array[10].push_back(mu.at<double>(0, 0)); trend_array[11].push_back(mu.at<double>(1, 0));
		cv::Mat g = elips.transform_.natural_grad();
		trend_array[12].push_back(cv::norm(g, cv::NORM_L2));	 // aim_idx
		cv::Mat f = elips.transform_.fisher_inv();
		trend_array[13].push_back(cv::norm(f, cv::NORM_L2));	 // aim_idx
		trend_array[14].push_back(aim_idx);	 // aim_idx
		if(iteration_count % write_trend_interval == 0)
		{			
			append_flag = iteration_count == 0 ? 0 : 1;			
			loader.SaveTrend(trend_array, trend_number_, append_flag);			
			loader.SaveWeightsForTest(elips.transform_, 1, dim_feature_); // single policy is 1 dimensional
			loader.SaveEllipse(elips);
			for(int i = 0; i < trend_number_; i++)
				trend_array[i].clear();
		}
		if(iteration_count % write_diagnosis_interval == 0)
		{
			loader.SaveWeightsForDiagnosis(elips.transform_, elips, 1, dim_feature_, iteration_count / write_diagnosis_interval); // single policy is 1 dimensional
		}
		// display
		if(iteration_count % display_interval == 1)
		{
			inv_transform = elips.transform_.transform_inv();
			cv::invert(inv_transform, transform);
			percentage = iteration_count * 100.0 / num_iteration_; // how much finished...
			std::cout << "training iteration: " << iteration_count << std::endl;
			std::cout << "x shift: " << transform.at<double>(0, 2) << " " << "y shift: " << transform.at<double>(1, 2) << std::endl;
			std::cout << "other elements: " << transform.at<double>(0, 0) << " " << transform.at<double>(0, 1) << " " << transform.at<double>(1, 0) << " " << transform.at<double>(1, 1) << std::endl;				
			std::cout << "aim index: " << aim_idx << std::endl; std::cout << "avg cost: " << avg_cost_ << std::endl; std::cout << "completed: " << percentage << "%" << std::endl;
			/*std::cout << "exploration range: " << starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * iteration_count / range_expanding_period_ << std::endl;
			std::cout << "exploration parameters: " << max_exploration_range_ << " " << starting_exploration_range_ << " " << iteration_count << " " << range_expanding_period_ << std::endl;*/
		}				
	}			
	loader.SaveTrend(trend_array, trend_number_, append_flag);  
}

void Explorer::OnlinePCA(cv::Mat& feature, int iteration, double threshold)
{
	int count = iteration + 1;
	if(count == 1)
	{
		feature_mean_old_ = cv::Mat::zeros(feature.rows, feature.cols, CV_64F);
		feature_mean_new_ = cv::Mat::zeros(feature.rows, feature.cols, CV_64F);
		feature.copyTo(feature_mean_old_);
		feature.copyTo(feature_mean_new_);
		feature_cov_old_ = cv::Mat::zeros(feature.rows, feature.rows, CV_64F);
		feature_cov_new_ = cv::Mat::zeros(feature.rows, feature.rows, CV_64F);
		feature_eigen_value_ = cv::Mat::zeros(feature.rows, feature.rows, CV_64F);
		feature_eigen_vec_ = cv::Mat::zeros(feature.rows, feature.rows, CV_64F);
	}
	else
	{
		feature_mean_new_ = feature_mean_old_ + (feature - feature_mean_old_) / count;
		feature_cov_new_ = feature_cov_old_ + (feature - feature_mean_old_) * (feature - feature_mean_new_).t();
		feature_mean_new_.copyTo(feature_mean_old_);
		feature_cov_new_.copyTo(feature_cov_old_);

		/*cv::Mat tmp_mean;
		cv::reduce(feature, tmp_mean, 0, CV_REDUCE_AVG);
		cv::Mat tmp_data = feature - cv::repeat(tmp_mean, feature.rows, 1);
		cv::Mat tmp_cov = tmp_data.t() * tmp_data / tmp_data.rows;*/
		eigen(feature_cov_new_, feature_eigen_value_, feature_eigen_vec_);
		cv::Mat eigen_value_sum;
		cv::Mat current_sum;
		cv::reduce(feature_eigen_value_, eigen_value_sum, 0, CV_REDUCE_SUM);
		int num_dim = 1;
		// eigen values are in descending order...
		for(int i = 0; i < feature_eigen_value_.rows; i++)
		{
			cv::reduce(feature_eigen_value_.rowRange(0, i + 1), current_sum, 0, CV_REDUCE_SUM);
			double current_portion = current_sum.at<double>(0, 0) / eigen_value_sum.at<double>(0, 0);
			if(current_portion > threshold)
			{
				// std::cout << "current portion: " << current_portion << std::endl;
				break;
			}
			else
				num_dim++;
		}
		actual_feature_dim_ = num_dim > actual_feature_dim_ ? num_dim : actual_feature_dim_;
		// eigen vectors are stored in rows... descending order, keep the rows
		cv::Mat projection_vectors = feature_eigen_vec_.rowRange(0, actual_feature_dim_);
		feature = projection_vectors * feature;
	}
}

void Explorer::SaveSinusoidalFeature()
{
	// only work for 2dof...
	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);		
	cv::Mat train_feature_mat = cv::Mat::zeros(num_train_data_, dim_feature_, CV_64F);		
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);	
	cv::Mat test_feature_mat = cv::Mat::zeros(num_test_data_, dim_feature_, CV_64F);	
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);		
	cv::Mat scale = cv::Mat::zeros(1, dim_feature_, CV_64F);		
	
	Loader loader(dim_action_, trend_number_, dir_id_, dir_);	
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	// loader.FormatWeightsForDiagnosisDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);	
	SetFeatureSinusoidal(feature_home_);

	for(int i = 0; i < num_train_data_; i++)
	{
		SetFeatureSinusoidal(feature, i, train_prop_, home_prop_);		
		cv::Mat tmp = feature.t();
		tmp.copyTo(train_feature_mat.rowRange(i, i + 1));
	}

	// do pca...

	/*cv::Mat tmp = abs(train_feature_mat);
	cv::reduce(tmp, scale, 0, CV_REDUCE_MAX);
	cv::divide(train_feature_mat, cv::repeat(scale, train_feature_mat.rows, 1),train_feature_mat);*/
	for(int i = 0; i < num_test_data_; i++)
	{
		SetFeatureSinusoidal(feature, i, test_prop_, home_prop_);		
		cv::Mat tmp = feature.t();
		tmp.copyTo(test_feature_mat.rowRange(i, i + 1));
	}
	/*tmp = abs(test_feature_mat);
	cv::reduce(tmp, scale, 0, CV_REDUCE_MAX);
	cv::divide(test_feature_mat, cv::repeat(scale, test_feature_mat.rows, 1),test_feature_mat);*/

	loader.SavePropSinusoidalFeature(train_feature_mat, test_feature_mat);

}

void Explorer::Test(int display_flag, int single_frame_flag, int start_idx, int end_idx, int test_idx, int test_flag, int record_img_flag)
{
	char input_dir[400];		
	char output_dir[400];		
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int img_height = 480;
	int img_width = 640;
	int update_gradient_flag = 0;
	int radius = 3.0;
	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);

	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat feature_home = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat train_feature_mat = cv::Mat::zeros(num_train_data_, dim_feature_, CV_64F);			
	cv::Mat test_feature_mat = cv::Mat::zeros(num_test_data_, dim_feature_, CV_64F);	
	// SetFeatureSinusoidal(feature_home);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);
	cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x_, initial_y_, initial_long_axis_, initial_short_axis_, initial_phi_, radius, ini_transformation_);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);
	loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);
	loader.LoadEllipse(elips);	
	loader.LoadPropSinusoidalFeature(train_feature_mat, test_feature_mat);
	elips.set_a_home(1);
	elips.set_inv_a_home();
	elips.set_a();
	elips.set_inv_a();	
	elips.UpdateEllipseVisualizationParameters(0);

	
	// elips.SetRefEllipseParameters();


	if(single_frame_flag == 1)
	{
		start_idx = test_idx;
		end_idx = test_idx + 1;
	}
	/*if(end_idx > num_train_data_)
	{
		std::cout << "invalid test index... exiting..." << std::endl;
		exit(0);
	}*/
	// base template for compare...
	aim_idx = 0;

	if(test_flag)
	{
		aim_frame_idx = test_target_idx.at<double>(aim_idx, 0);
		// SetFeatureSinusoidal(feature, aim_idx, test_prop_, home_prop_);		
		feature = test_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
		feature = feature.t();
	}
	else
	{
		aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
		// aim_frame_idx = aim_idx;
		// SetFeatureSinusoidal(feature, aim_idx, train_prop_, home_prop_);		
		feature = train_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
		feature = feature.t();
	}
	// feature = feature - feature_home;
	loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx, ini_transformation_);
	/*cv::Mat tmp_key_points = cv::Mat::zeros(key_points_.rows, key_points_.cols, CV_32F);
	key_points_.copyTo(tmp_key_points);
	TransformKeyPointsMatrix(tmp_key_points, key_points_);*/
	elips.transform_.CalcTransformInv(feature);
	elips.TransformEllipse();
	elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
	EvaluateGradientAndUpdate(feature, update_gradient_flag, aim_idx, elips, 0);			
	elips.UpdateEllipseVisualizationParameters(0);
	elips.CopyToPrev();
	CopyToPrev();	

	cv::Mat inv_ini_transformation = cv::Mat::eye(3, 3, CV_64F);

	for(aim_idx = start_idx; aim_idx < end_idx; aim_idx++)
	{
		// load key points		
		if(test_flag)
		{
			aim_frame_idx = test_target_idx.at<double>(aim_idx, 0);
			// SetFeatureSinusoidal(feature, aim_idx, test_prop_, home_prop_);		
			feature = test_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
			feature = feature.t();
		}
		else
		{
			aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
			// aim_frame_idx = aim_idx;
			// SetFeatureSinusoidal(feature, aim_idx, train_prop_, home_prop_);		
			feature = train_feature_mat.rowRange(aim_idx, aim_idx + 1).clone();
			feature = feature.t();
		}
		// feature = feature - feature_home;
		loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx, ini_transformation_);		
		// act & evaluate gradient						
		elips.transform_.CalcTransformInv(feature);
		elips.TransformEllipse();
		elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
		EvaluateGradientAndUpdate(feature, update_gradient_flag, aim_idx, elips, 0);			
		elips.UpdateEllipseVisualizationParameters(0);
		if(display_flag == 1)
		{
			// display...			
			loader.LoadImage(aim_frame_idx, disp_img);
			elips.DrawEllipse(disp_img, 1.0); // draw ellipse
			cv::Mat transform_inv = elips.transform_.transform_inv();
			cv::Mat transform = cv::Mat::zeros(transform_inv.rows, transform_inv.cols, CV_64F);
			cv::invert(transform_inv, transform);
			cv::invert(ini_transformation_, inv_ini_transformation);
			

			for(int i = 0; i < matched_points_.rows; i++)
			{		
				
				home_point_.at<double>(0, 0) = matched_points_.at<double>(i, 0); home_point_.at<double>(1, 0) = matched_points_.at<double>(i, 1); home_point_.at<double>(2, 0) = 1.0;				
				img_point_ = inv_ini_transformation * transform * home_point_;
				// cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
				cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
				// line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
			}	
			/*for(int i = 0; i < unique_match_point_.size(); i++)
			{		
				prev_home_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 7); prev_home_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 8);				
				prev_home_point_.at<double>(2, 0) = 1.0;
				prev_img_point_ = inv_ini_transformation * transform * prev_home_point_;
				
				home_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); home_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);				
				home_point_.at<double>(2, 0) = 1.0;
				img_point_ = inv_ini_transformation * transform * home_point_;	

				cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
				cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
				line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
			}	*/

			if(record_img_flag)
			{
				sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/images/%d.pgm", dir_id_, aim_idx);
				imwrite(output_dir, disp_img);
				if(aim_idx % 50 == 0)
					std::cout << "tested frame: " << aim_idx << std::endl;
			}
			else
			{
				std::cout << aim_frame_idx << std::endl;
				std::cout << "x: " << action.at<double>(0, 0) << " y: " << action.at<double>(1, 0) << " a: " <<  action.at<double>(2, 0) << " lx: " << action.at<double>(3, 0) << " sx: " << action.at<double>(4, 0) << std::endl;
				std::cout << "aim_idx: " << aim_idx << std::endl;			
				cv::imshow("explorer", disp_img); // show ellipse
				cv::waitKey(0);
			}
		}
		else
		{
			if(aim_idx % 10 == 0)
				std::cout << "tested frame: " << aim_idx << std::endl;
		}

		// copy to previous		
		elips.CopyToPrev();
		CopyToPrev();							
		
	}				
	std::cout << "test finished..." << std::endl;	
}

void Explorer::RecursiveLeastSquareTest() // (int display_flag, int single_frame_flag, int start_idx, int end_idx, int test_idx, int test_flag, int record_img_flag)
{
	char input_dir[400];		
	char output_dir[400];				
	int img_height = 480;
	int img_width = 640;	
	int radius = 3.0;
	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);
	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x_, initial_y_, initial_long_axis_, initial_short_axis_, initial_phi_, radius, ini_transformation_);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);
	loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);

	int num_weights = 6;
	int feature_dim = 10;
	cv::Mat y = cv::Mat::zeros(num_train_data_, num_weights, CV_64F);
	cv::Mat x = cv::Mat::zeros(num_train_data_, feature_dim, CV_64F);
	cv::Mat tmp_transform = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat tmp_feature = cv::Mat::zeros(1, feature_dim, CV_64F);

	for(int idx = 0; idx < num_train_data_; idx++)
	{
		SetFeatureSinusoidal(feature, idx, train_prop_, home_prop_);
		elips.transform_.CalcTransformInv(feature);
		tmp_transform = elips.transform_.transform_inv() - cv::Mat::eye(3, 3, CV_64F);		
		tmp_transform = tmp_transform.rowRange(0, 2).reshape(1, 1);
		tmp_feature = feature.t();
		tmp_transform.copyTo(y.rowRange(idx, idx + 1));		
		tmp_feature.copyTo(x.rowRange(idx, idx + 1));
	}
	
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/x.bin", dir_id_);
	FileIO::WriteMatDouble(x, num_train_data_, feature_dim, output_dir);
	
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/y.bin", dir_id_);
	FileIO::WriteMatDouble(y, num_train_data_, num_weights, output_dir);

	// actually not necessary for testing the recursive least squares...
	/*loader.LoadEllipse(elips);	
	elips.set_a_home(1);
	elips.set_inv_a_home();
	elips.set_a();
	elips.set_inv_a();	
	elips.UpdateEllipseVisualizationParameters(0);*/
}

// temporary function to get video images...
void Explorer::GetVideoImages()
{
	char input_dir[400];		
	char output_dir[400];		
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int img_height = 480;
	int img_width = 640;
	int update_gradient_flag = 0;
	int radius = 3.0;
	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);

	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);
	cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	std::mt19937 engine(rd_());		

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x_, initial_y_, initial_long_axis_, initial_short_axis_, initial_phi_, radius, ini_transformation_);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);


	int length = 1000000;
	int diagnosis_interval = 2500;
	int num_diagnosis = length / diagnosis_interval;

	for(int i = 0; i < num_diagnosis; i++)
	{
		// load current diagnosis weight and ellipse
		loader.LoadWeightsForDiagnosis(elips.transform_, elips, 1, dim_feature_, i);
		elips.set_a_home(1);
		elips.set_inv_a_home();
		elips.set_a();
		elips.set_inv_a();	
		int repeat = 2;
		int shift = 14000;
		for(int k = 0; k < repeat; k++)
		{
			// aim_idx = i * repeat + k + 14000; // GenerateAimIndexLinePath(engine, 1000000);
			aim_frame_idx = i * repeat + k + shift; // train_target_idx.at<double>(aim_idx, 0);			
			SetFeatureSinusoidal(feature, aim_frame_idx, train_prop_, home_prop_);
			elips.transform_.CalcTransformInv(feature);
			elips.TransformEllipse();				
			elips.UpdateEllipseVisualizationParameters(0);
			loader.LoadImage(aim_frame_idx, disp_img);
			elips.DrawEllipse(disp_img, 1.0); // draw ellipse
			// cv::imshow("explorer", disp_img); // show ellipse
			// cv::waitKey(0);
			sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/images/%d.pgm", dir_id_, shift + i * repeat + k);
			imwrite(output_dir, disp_img);
		}
		if(i % 10 == 0)
			std::cout << "diagnosis iteration: " << i << std::endl;

		elips.CopyToPrev();
		CopyToPrev();		

	}								
	std::cout << "video image recording finished..." << std::endl;	
}








// display key points...
// not displaying the filtered key points... just all the unique matched key points..
// reason: we can potentially find out whether the matched points' distance has been altered...
// and examine what is the proper threshold for the classification
//for(int i = 0; i < unique_match_point_.size(); i++)
//{		
//	prev_ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 7); prev_ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 8);				
//	prev_ref_point_.at<double>(2, 0) = 1.0;
//	prev_img_point_ = transform * prev_ref_point_;
//	/*prev_img_point_.copyTo(tmp);
//	TransformSinglePoint(tmp, prev_img_point_);*/
//	/*tmp = rotation * prev_img_point_;
//	prev_img_point_.at<double>(0, 0) = tmp.at<double>(0, 0) + initial_x; prev_img_point_.at<double>(1, 0) = tmp.at<double>(1, 0) + initial_y;*/
//	// prev_img_point_.at<double>(0, 0) = tmp.at<double>(0, 0) + 0; prev_img_point_.at<double>(1, 0) = tmp.at<double>(1, 0) + 0;
//	// prev_img_point_.at<double>(0, 0) += initial_x; prev_img_point_.at<double>(1, 0) += initial_y;
//	ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);				
//	ref_point_.at<double>(2, 0) = 1.0;
//	img_point_ = transform * ref_point_;	
//	/*img_point_.copyTo(tmp);
//	TransformSinglePoint(tmp, img_point_);*/
//	/*tmp = rotation * img_point_;
//	img_point_.at<double>(0, 0) = tmp.at<double>(0, 0) + initial_x; img_point_.at<double>(1, 0) = tmp.at<double>(1, 0) + initial_y;*/
//	// img_point_.at<double>(0, 0) = tmp.at<double>(0, 0) + 0; img_point_.at<double>(1, 0) = tmp.at<double>(1, 0) + 0;
//	// img_point_.at<double>(0, 0) += initial_x; img_point_.at<double>(1, 0) += initial_y;
//	cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
//	cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
//	line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
//}		
//for(int i = 0; i < matched_points_.rows; i++)
//{		
//	
//	ref_point_.at<double>(0, 0) = matched_points_.at<double>(i, 0); ref_point_.at<double>(1, 0) = matched_points_.at<double>(i, 1);
//	ref_point_.at<double>(2, 0) = 1.0;
//	img_point_ = transform * ref_point_;
//	// cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
//	cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
//	// line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
//}	



//int aim_idx = 0;
//	double current_range = 0;
//	double max_speed = 0.4;
//	double path_length = 0;
//	int num_frame_path = 0;
//	int prop_dim = train_prop_.cols;
//	// generate path
//	if(path.size() == 0)
//	{		
//		current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
//		current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
//		std::uniform_real_distribution<double> uniform(-1.0 * current_range, 1.0 * current_range);	  	
//		std::uniform_real_distribution<double> uniform_speed(0.0, 0.01);	  
//
//
//		for(int i = 0; i < prop_dim - 1; i++)
//			target_.at<double>(0, i) = uniform(engine);		
//		path_length = cv::norm(target_.colRange(0, prop_dim - 1) - prev_target_.colRange(0, prop_dim - 1), cv::NORM_L2);
//		num_frame_path = (int)(path_length / max_speed) + 1;
//		path.clear();
//
//		std::vector<double> rotation_targets(num_frame_path);
//		for(int i = 1; i <= num_frame_path; i++)
//		{
//			cv::Mat tmp_target = cv::Mat::zeros(1, prop_dim, CV_64F);
//			cv::Mat tmp_data = cv::Mat::zeros(1, prop_dim, CV_64F);
//			tmp_target.colRange(0, prop_dim - 1) = prev_target_.colRange(0, prop_dim - 1) + (target_.colRange(0, prop_dim - 1) - prev_target_.colRange(0, prop_dim - 1)) * i / num_frame_path;
//			if(abs(prev_target_.at<double>(0, 2) - rotation_target_) < 0.05)			
//				rotation_target_ = -rotation_target_;
//			rotation_speed_ = uniform_speed(engine);
//			if(i == 1)
//				rotation_targets[i - 1] = rotation_target_ == 1 ? prev_target_.at<double>(0, 2) + rotation_speed_ : prev_target_.at<double>(0, 2) - rotation_speed_;
//			else
//				rotation_targets[i - 1] = rotation_target_ == 1 ? rotation_targets[i - 2]  + rotation_speed_ : rotation_targets[i - 1]  - rotation_speed_;
//			tmp_target.at<double>(0, 2) = rotation_targets[i - 1]; // rotation_target_ == 1 ? tmp_target.at<double>(0, 2) + rotation_speed_ : tmp_target.at<double>(0, 2) - rotation_speed_;
//			tmp_target.convertTo(tmp_target, CV_32F);
//			// radius search
//			double radius = 0.1;
//			kd_trees.radiusSearch(tmp_target, rs_indices_, rs_dists_, radius, 20, cv::flann::SearchParams(64)); 
//			// make sure it is not 0...
//			// row number denotes the number of query data... column number denotes the number of neighbor you wanted...
//			while(rs_indices_.cols == 0)
//			{
//				radius += 0.1;
//				kd_trees.radiusSearch(tmp_target, rs_indices_, rs_dists_, radius, 20, cv::flann::SearchParams(64)); 
//			}
//			int curr_idx = rs_indices_.at<int>(0, 0);
//			int min_idx = curr_idx;
//			/*for(int j = 0; j < rs_indices_.cols; j++)
//			{
//				curr_idx = rs_indices_.at<int>(0, j);
//				if(j == 0)
//				{
//					train_prop_.rowRange(curr_idx, curr_idx + 1).copyTo(tmp_data);
//					min_idx = curr_idx;
//				}
//				else if(abs(train_prop_.at<double>(curr_idx, 2) - tmp_target.at<float>(0, 2)) < abs(tmp_data.at<double>(0, 2) - tmp_target.at<float>(0, 2)))
//				{
//					train_prop_.rowRange(curr_idx, curr_idx + 1).copyTo(tmp_data);
//					min_idx = curr_idx;
//				}
//			}*/
//			path.push_back(min_idx);			
//		}	
//		target_.at<double>(0, 2) = rotation_targets[num_frame_path - 1];
//		target_.copyTo(prev_target_);
//	}		
//	aim_idx = path[0];
//	path.erase(path.begin());		
//	return aim_idx;
