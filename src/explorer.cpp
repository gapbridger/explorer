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
Explorer::Explorer
	(	
	int dir_id, 
	int num_iteration,
	int expanding_period,
	char* dir
	)
{    
	dir_id_ = dir_id;
	dim_action_ = 5; // dimension of action
	dim_feature_ = 10;
	trend_number_ = 12;
	num_iteration_ = num_iteration;
	range_expanding_period_ = expanding_period;
	num_train_data_ = 16400 * 19 / 20; // 17100; // 20250; // 17100; // 20250; // 20880; // 20250; // 17280; // 9900; // 18000;
	num_test_data_ = 16400 / 20; // 1900; // 2250; // 1900; // 2250; // 2320; // 2250; // 1920; // 1100; // 2000;
	max_exploration_range_ = 1.0; // 1.00 // 1.41; // 0.71; // sqrt(2) should be fine... 0.5;
	starting_exploration_range_ = 0.04; // 0.002; // 0.101;
	avg_cost_ = 0;		
	path_count_ = 0;
	target_p_1_ = 0;
	target_p_2_ = 0;
	prev_target_p_1_ = 0;
	prev_target_p_2_ = 0;

	action_ = cv::Mat::zeros(dim_action_, 1, CV_64F);	
	train_prop_ = cv::Mat::zeros(num_train_data_, 2, CV_64F);
	test_prop_ = cv::Mat::zeros(num_test_data_, 2, CV_64F);	
	home_prop_ = cv::Mat::zeros(1, 2, CV_64F);	
	img_point_ = cv::Mat::zeros(3, 1, CV_64F);			img_point_.at<double>(2, 0) = 1; // image frame
	prev_img_point_ = cv::Mat::zeros(3, 1, CV_64F);		prev_img_point_.at<double>(2, 0) = 1; // image frame
	ref_point_ = cv::Mat::zeros(3, 1, CV_64F);			ref_point_.at<double>(2, 0) = 1; // reference frame
	prev_ref_point_ = cv::Mat::zeros(3, 1, CV_64F);		prev_ref_point_.at<double>(2, 0) = 1; // reference frame	
	curr_prop_ = cv::Mat::zeros(1, 2, CV_64F); 
	curr_prop_matrix_ = cv::Mat::zeros(num_train_data_, 2, CV_64F);
	prop_diff_ = cv::Mat::zeros(num_train_data_, 2, CV_64F);
	prop_dist_ = cv::Mat::zeros(num_train_data_, 1, CV_64F);
	aim_idx_matrix_ = cv::Mat::zeros(num_train_data_, 1, CV_32S);
	feature_data_ = cv::Mat::zeros(2, 1, CV_64F);

	sprintf(dir_, dir);
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

int Explorer::GenerateAimIndex(std::mt19937& engine, int current_iteration)
{
	int aim_idx = 0;   	
	double current_range = 0;	
	double interval_range_low = 0;
	double interval_range_high = 0;
	double dist = 0;
	current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
	current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
	std::uniform_real_distribution<double> uniform(-1.0 * current_range, 1.0 * current_range);
	interval_range_low = current_range / 4.5;
	interval_range_high = current_range / 3.5;
	/*curr_prop_.at<double>(0, 0) = uniform(engine); 
	curr_prop_.at<double>(0, 1) = uniform(engine);*/

	// ensure interval...
	do
	{
		target_p_1_ = uniform(engine);
		target_p_2_ = uniform(engine);
		dist = sqrt(pow(target_p_1_ - prev_target_p_1_, 2) + pow(target_p_2_ - prev_target_p_2_, 2));
	}
	while(dist < interval_range_low || dist > interval_range_high);
	curr_prop_.at<double>(0, 0) = target_p_1_; 
	curr_prop_.at<double>(0, 1) = target_p_2_;
	prev_target_p_1_ = target_p_1_;
	prev_target_p_2_ = target_p_2_;	

	//*************************//

	curr_prop_matrix_ = repeat(curr_prop_, num_train_data_, 1);
	prop_diff_ = train_prop_ - curr_prop_matrix_;
	prop_diff_ = prop_diff_.mul(prop_diff_);
	reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
	sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
	aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);		
	return aim_idx;
}

void Explorer::GenerateLinePath(fL& path_p_1, fL& path_p_2, double target_p_1, double target_p_2, double prev_target_p_1, double prev_target_p_2)
{		
	double max_speed = 8 / 20.0;
	double path_length = sqrt(pow(target_p_1 - prev_target_p_1, 2) + pow(target_p_2 - prev_target_p_2, 2));
	int num_frame_path = (int)(path_length / max_speed) + 1;

	path_p_1.clear();
	path_p_2.clear();

	for(int i = 1; i <= num_frame_path; i++)
	{
		path_p_1.push_back(prev_target_p_1 + (target_p_1 - prev_target_p_1) * i / num_frame_path);
		path_p_2.push_back(prev_target_p_2 + (target_p_2 - prev_target_p_2) * i / num_frame_path);
	}

}

int Explorer::GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration)
{
	int aim_idx = 0;
	double current_range = 0;
	
	current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
	current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
	std::uniform_real_distribution<double> uniform(-1.0 * current_range, 1.0 * current_range);	  	
	
	// generate path
	if(path_count_ == 0)
	{
		target_p_1_ = uniform(engine);
		target_p_2_ = uniform(engine);
		GenerateLinePath(path_p_1_, path_p_2_, target_p_1_, target_p_2_, prev_target_p_1_, prev_target_p_2_);
		prev_target_p_1_ = target_p_1_;
		prev_target_p_2_ = target_p_2_;
		path_count_ = path_p_1_.size();
	}

	curr_prop_.at<double>(0, 0) = path_p_1_[path_p_1_.size() - path_count_]; 
	curr_prop_.at<double>(0, 1) = path_p_2_[path_p_1_.size() - path_count_];
	path_count_--;
	// path_p_1_.erase(path_p_1_.begin());
	// path_p_2_.erase(path_p_2_.begin());

	curr_prop_matrix_ = repeat(curr_prop_, num_train_data_, 1);
	prop_diff_ = train_prop_ - curr_prop_matrix_;
	prop_diff_ = prop_diff_.mul(prop_diff_);
	reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
	sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
	aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);

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
	int match_point_info_size = 13;	
	double curr_diff = 0;
	double predicted_motion_ratio = 0;
	match_point_info_.clear();
    unique_match_point_.clear();
	cv::Mat predicted_img_point;
	
	cv::Mat transform = cv::Mat::zeros(3, 3, CV_64F);
	if(elips_descriptors_.rows != 0)
	{
		// interface between double and float...		
		/*cv::Mat tmp_prev_descriptors = cv::Mat::zeros(elips_prev_descriptors_.rows, elips_prev_descriptors_.cols, CV_32F);
		cv::Mat tmp_descriptors = cv::Mat::zeros(elips_descriptors_.rows, elips_descriptors_.cols, CV_32F);
		elips_prev_descriptors_.convertTo(tmp_prev_descriptors, CV_32F);
		elips_descriptors_.convertTo(tmp_descriptors, CV_32F);
		matcher_.match(tmp_prev_descriptors, tmp_descriptors, matches_);	*/

		//cv::Mat tmp_prev_descriptors = cv::Mat::zeros(elips_prev_descriptors_.rows, elips_prev_descriptors_.cols, CV_32F);
		//cv::Mat tmp_descriptors = cv::Mat::zeros(elips_descriptors_.rows, elips_descriptors_.cols, CV_32F);
		//elips_prev_descriptors_.convertTo(tmp_prev_descriptors, CV_32F);
		//elips_descriptors_.convertTo(tmp_descriptors, CV_32F);
		matcher_.match(elips_prev_descriptors_, elips_descriptors_, matches_);	
	}
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
		prev_ref_point_ = elips.transform_.TransformDataPointInv(prev_img_point_, 0);
		ref_point_ = elips.transform_.TransformDataPointInv(img_point_, 1); // transform image coordinate to reference coordinate		
		predicted_img_point = elips.transform_.InterFrameTransformImg(prev_img_point_);				
		predicted_motion_ratio = 1 - cv::norm(img_point_ - predicted_img_point) / cv::norm(img_point_ - prev_img_point_);
		predicted_motion_ratio = predicted_motion_ratio < 0 ? 0 : predicted_motion_ratio;
		
		curr_diff = 1 - exp(-0.5 * GRADIENT_SCALE * (pow(prev_ref_point_.at<double>(0, 0) - ref_point_.at<double>(0, 0), 2) + pow(prev_ref_point_.at<double>(1, 0) - ref_point_.at<double>(1, 0), 2)));		
		match_point_info_[i].at<double>(0, 0) = curr_diff; // reward		
		match_point_info_[i].at<double>(0, 1) = prev_match_idx; match_point_info_[i].at<double>(0, 2) = aim_match_idx; // base idx and aim idx		
		match_point_info_[i].at<double>(0, 3) = prev_img_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 4) = prev_img_point_.at<double>(1, 0); // image frame base point
		match_point_info_[i].at<double>(0, 5) = img_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 6) = img_point_.at<double>(1, 0); // image frame aim point
		match_point_info_[i].at<double>(0, 7) = prev_ref_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 8) = prev_ref_point_.at<double>(1, 0); // back in reference frame
		match_point_info_[i].at<double>(0, 9) = ref_point_.at<double>(0, 0); match_point_info_[i].at<double>(0, 10) = ref_point_.at<double>(1, 0); // image frame predicting point
		match_point_info_[i].at<double>(0, 11) = elips_distance_.at<double>(aim_match_idx, 0); // elips.CalcMahaDist(img_point_.rowRange(0, 2));
		match_point_info_[i].at<double>(0, 12) = predicted_motion_ratio;		
   
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
					prev_ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 7); prev_ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 8);		
					prev_ref_point_.at<double>(2, 0) = 1.0;
					img_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 5); img_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 6);
					img_point_.at<double>(2, 0) = 1.0;
					ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);
					ref_point_.at<double>(2, 0) = 1.0;
					// inverse transform
					elips.transform_.CalcMiniBatchInvGradient(img_point_, ref_point_, prev_ref_point_, feature, gradient_match_point_count, gradient_batch_idx);	
					gradient_batch_idx++;
				}
			}	
			// update weights
			elips.transform_.UpdateWeightBatch();
			// classify points in reference frame
			elips.ClassifyPointsForReferenceFrame(unique_match_point_, matched_points_, motion_ratio_, maha_dist_);
			// update ellipse by the matched points
			elips.UpdateRefEllipse(matched_points_, motion_ratio_, maha_dist_);
		}
	}
	else
	{
		/*if(aim_idx % 1 == 1)
		{*/
			elips.ClassifyPointsForReferenceFrame(unique_match_point_, matched_points_, motion_ratio_, maha_dist_);

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
	int write_diagnosis_interval = 2500;
	int percentage = 0; 		
	unsigned long iteration_count = 0;		

	bool append_flag = false;	

	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);

	//double initial_x = 300; // 218; 
	//double initial_y = 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 20; // 22
	//double initial_angle = -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	//double initial_x = 300; // 380; // 300; // 218; 
	//double initial_y = 193; // 220; // 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 12; 
	//double initial_angle = -1.0 * PI / 9; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 1.0; // 2.0

	
	//double initial_x = 370; // 380; // 300; // 218; 
	//double initial_y = 225; // 220; // 260; // 230; 
	//double initial_long_axis = 45; 
	//double initial_short_axis = 35; 
	//double initial_angle = 0; // -1.0 * PI / 3; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/input/elips_ini_%d.bin", dir_id_);
	FileIO::ReadMatDouble(elips_ini, 5, 1, input_dir);

	double initial_x = elips_ini.at<double>(0, 0); // 380; // 300; // 218; 
	double initial_y = elips_ini.at<double>(1, 0); // 220; // 260; // 230; 
	double initial_long_axis = elips_ini.at<double>(2, 0); 
	double initial_short_axis = elips_ini.at<double>(3, 0); 
	double initial_angle = elips_ini.at<double>(4, 0); // -1.0 * PI / 20; // 1.0 * PI / 8.5; 

	double radius = 3.0;

	max_exploration_range_ = 1.0;	

	// keep the exploration in a small range...	
	std::mt19937 engine(rd_());		
	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);		
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);		
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);		
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);	
	cv::Mat cov = cv::Mat::zeros(2, 2, CV_64F);
	cv::Mat mu = cv::Mat::zeros(2, 1, CV_64F);
	cv::Mat improvement_avg = cv::Mat::zeros(1, 1, CV_64F);
	// cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	fL* trend_array = new fL[trend_number_];
	// objects...	
	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle, radius);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	// loader.FormatWeightsForDiagnosisDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);	
	loader.LoadLearningRates(elips);	
	// for tool extension experiment only...
	//loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);
	//loader.LoadEllipse(elips);
	//elips.SetRefEllipseParameters();
	// ********************************* //

	// ***** try load all key points...******** //
	descriptors_all_.clear(); key_points_all_.clear();
	loader.LoadAllSiftKeyPoint(descriptors_all_, key_points_all_, 0, num_train_data_ + num_test_data_); // 23248); // num_train_data_ + num_test_data_);
	// main loop
	for(iteration_count = 0; iteration_count < num_iteration_; iteration_count++)
	{		
		// aim_idx = GenerateAimIndex(engine, iteration_count);
		aim_idx = GenerateAimIndexLinePath(engine, iteration_count);
		aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);

		// loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx);					
		if(aim_frame_idx < descriptors_all_.size())
		{
			descriptors_ = descriptors_all_[aim_frame_idx];
			key_points_ = key_points_all_[aim_frame_idx];
		}
		else
		{
			std::cout << "number of descriptor matrix incorrect... exit..." << std::endl;
			exit(0);
		}

		// act & evaluate gradient
		if(iteration_count != 0)
		{
			SetFeature(feature, aim_idx, train_prop_, home_prop_);
			action = elips.transform_.EvaluateInvTransformation(feature);
			elips.UpdateEllipseByAction(action);  		
			elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);	
			EvaluateGradientAndUpdate(feature, 1, aim_idx, elips, iteration_count);		
			/*if(improvement_.rows != 0)
				cv::reduce(improvement_, improvement_avg, 0, CV_REDUCE_AVG);*/
		}	
		else
		{
			// only for the first iteration...
			elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
		}		

		// copy to previous		
		elips.CopyToPrev();
		CopyToPrev();							
		cov = elips.ref_cov();
		mu = elips.ref_mu();

		
		// recording
		trend_array[0].push_back(cv::norm(elips.transform_.w_x(), cv::NORM_L2)); trend_array[1].push_back(cv::norm(elips.transform_.w_y(), cv::NORM_L2));      
		trend_array[2].push_back(cv::norm(elips.transform_.w_phi(), cv::NORM_L2)); trend_array[3].push_back(cv::norm(elips.transform_.w_sx(), cv::NORM_L2));      
		trend_array[4].push_back(cv::norm(elips.transform_.w_sy(), cv::NORM_L2)); 
		trend_array[5].push_back(cov.at<double>(0, 0)); trend_array[6].push_back(cov.at<double>(0, 1));	
		trend_array[7].push_back(cov.at<double>(1, 0)); trend_array[8].push_back(cov.at<double>(1, 1));
		trend_array[9].push_back(mu.at<double>(0, 0)); trend_array[10].push_back(cov.at<double>(1, 0));
		trend_array[11].push_back(aim_idx);	 // aim_idx
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
			percentage = iteration_count * 100.0 / num_iteration_; // how much finished...
			std::cout << "training iteration: " << iteration_count << std::endl;
			std::cout << "x shift: " << action.at<double>(0, 0) << " " << "y shift: " << action.at<double>(1, 0) << std::endl;
			std::cout << "rotation: " << action.at<double>(2, 0) << " " << "x scaling: " << action.at<double>(3, 0) << " " << "y scaling: " << action.at<double>(4, 0) << std::endl;			
			std::cout << "aim index: " << aim_idx << std::endl; std::cout << "avg cost: " << avg_cost_ << std::endl; std::cout << "completed: " << percentage << "%" << std::endl;
		}				
	}			
	loader.SaveTrend(trend_array, trend_number_, append_flag);  
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
	cv::Mat elips_ini = cv::Mat::zeros(5, 1, CV_64F);
	//double initial_x = 300; // 218; 
	//double initial_y = 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 20; 
	//double initial_angle = -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	//double initial_x = 300; // 380; // 300; // 218; 
	//double initial_y = 193; // 220; // 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 12; 
	//double initial_angle = -1.0 * PI / 9; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 1.0; // 2.0;

	//double initial_x = 370; // 380; // 300; // 218; 
	//double initial_y = 225; // 220; // 260; // 230; 
	//double initial_long_axis = 45; 
	//double initial_short_axis = 35; 
	//double initial_angle = 0; // -1.0 * PI / 3; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/input/elips_ini_%d.bin", dir_id_);
	FileIO::ReadMatDouble(elips_ini, 5, 1, input_dir);

	double initial_x = elips_ini.at<double>(0, 0); // 380; // 300; // 218; 
	double initial_y = elips_ini.at<double>(1, 0); // 220; // 260; // 230; 
	double initial_long_axis = elips_ini.at<double>(2, 0); 
	double initial_short_axis = elips_ini.at<double>(3, 0); 
	double initial_angle = elips_ini.at<double>(4, 0); // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	double radius = 3.0;

	//double initial_x = 350; // 380; // 300; // 218; 
	//double initial_y = 160; // 220; // 260; // 230; 
	//double initial_long_axis = 50; 
	//double initial_short_axis = 30; 
	//double initial_angle = -1.0 * PI / 9; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);
	cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle, radius);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);
	loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);
	loader.LoadEllipse(elips);
	elips.SetRefEllipseParameters();


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
		SetFeature(feature, aim_idx, test_prop_, home_prop_);		
	}
	else
	{
		aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
		SetFeature(feature, aim_idx, train_prop_, home_prop_);		
	}

	loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx);
	action = elips.transform_.EvaluateInvTransformation(feature);
	elips.UpdateEllipseByAction(action);  
	elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
	EvaluateGradientAndUpdate(feature, update_gradient_flag, aim_idx, elips, 0);			
	elips.CopyToPrev();
	CopyToPrev();	

	for(aim_idx = start_idx; aim_idx < end_idx; aim_idx++)
	{
		// load key points		
		if(test_flag)
		{
			aim_frame_idx = test_target_idx.at<double>(aim_idx, 0);
			SetFeature(feature, aim_idx, test_prop_, home_prop_);		
		}
		else
		{
			aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
			SetFeature(feature, aim_idx, train_prop_, home_prop_);		
		}

		loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx);		
		// act & evaluate gradient				
		action = elips.transform_.EvaluateInvTransformation(feature);
		elips.UpdateEllipseByAction(action);  
		elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
		EvaluateGradientAndUpdate(feature, update_gradient_flag, aim_idx, elips, 0);			
		if(display_flag == 1)
		{
			// display...			
			loader.LoadImage(aim_frame_idx, disp_img);
			elips.DrawEllipse(disp_img, 1.0); // draw ellipse
			cv::Mat transform_inv = elips.transform_.transform_inv();
			cv::Mat transform = cv::Mat::zeros(transform_inv.rows, transform_inv.cols, CV_64F);
			cv::invert(transform_inv, transform);
			// display key points...
			// not displaying the filtered key points... just all the unique matched key points..
			// reason: we can potentially find out whether the matched points' distance has been altered...
			// and examine what is the proper threshold for the classification
			//for(int i = 0; i < unique_match_point_.size(); i++)
			//{		
			//	prev_ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 7); prev_ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 8);				
			//	prev_ref_point_.at<double>(2, 0) = 1.0;
			//	prev_img_point_ = transform * prev_ref_point_;
			//	ref_point_.at<double>(0, 0) = unique_match_point_[i].at<double>(0, 9); ref_point_.at<double>(1, 0) = unique_match_point_[i].at<double>(0, 10);				
			//	ref_point_.at<double>(2, 0) = 1.0;
			//	img_point_ = transform * ref_point_;								
			//	cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
			//	cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
			//	line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
			//}		
			//// /****************** for debugging *********************/
			//cv::Mat prev_key_points_ref_debug = cv::Mat::zeros(unique_match_point_.size(), 2, CV_64F);
			//cv::Mat curr_key_points_ref_debug = cv::Mat::zeros(unique_match_point_.size(), 2, CV_64F);
			//for(int i = 0; i < unique_match_point_.size(); i++)
			//{
			//	prev_key_points_ref_debug.at<double>(i, 0) = unique_match_point_[i].at<double>(0, 7);
			//	prev_key_points_ref_debug.at<double>(i, 1) = unique_match_point_[i].at<double>(0, 8);
			//	curr_key_points_ref_debug.at<double>(i, 0) = unique_match_point_[i].at<double>(0, 9);
			//	curr_key_points_ref_debug.at<double>(i, 1) = unique_match_point_[i].at<double>(0, 10);
			//}
			//sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/matches_debug/prev_key_points_%d.bin", dir_id_, aim_idx);
			//FileIO::WriteMatDouble(prev_key_points_ref_debug, unique_match_point_.size(), 2, output_dir);
			//sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/matches_debug/curr_key_points_%d.bin", dir_id_, aim_idx);
			//FileIO::WriteMatDouble(curr_key_points_ref_debug, unique_match_point_.size(), 2, output_dir);


			for(int i = 0; i < matched_points_.rows; i++)
			{		
				
				ref_point_.at<double>(0, 0) = matched_points_.at<double>(i, 0); ref_point_.at<double>(1, 0) = matched_points_.at<double>(i, 1);
				ref_point_.at<double>(2, 0) = 1.0;
				img_point_ = transform * ref_point_;
				// cv::circle(disp_img, cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), 2, cv::Scalar(200, 0, 0));
				// cv::circle(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), 2, cv::Scalar(0, 200, 0));
				// line(disp_img, cv::Point2f(img_point_.at<double>(0, 0), img_point_.at<double>(1, 0)), cv::Point2f(prev_img_point_.at<double>(0, 0), prev_img_point_.at<double>(1, 0)), cv::Scalar(0, 0, 200));
			}

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
			if(aim_idx % 500 == 0)
				std::cout << "tested frame: " << aim_idx << std::endl;
		}

		// copy to previous		
		elips.CopyToPrev();
		CopyToPrev();							
		
	}				
	std::cout << "test finished..." << std::endl;	
}




void Explorer::PlotDiagnosis(int test_idx)
{
	char input_dir[400];		
	char output_dir[400];		
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int home_frame_idx = 0;
	int img_height = 480;
	int img_width = 640;
	int update_gradient_flag = 0;

	//double initial_x = 300; // 380; // 300; // 218; 
	//double initial_y = 193; // 220; // 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 12; 
	//double initial_angle = -1.0 * PI / 9; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 1.0; // 2.0;

	//double initial_x = 370; // 380; // 300; // 218; 
	//double initial_y = 225; // 220; // 260; // 230; 
	//double initial_long_axis = 45; 
	//double initial_short_axis = 35; 
	//double initial_angle = 0; // -1.0 * PI / 3; // 1.0 * PI / 8.5; 
	//double radius = 3.0;


	double initial_x = 350; // 380; // 300; // 218; 
	double initial_y = 160; // 220; // 260; // 230; 
	double initial_long_axis = 40; 
	double initial_short_axis = 40; 
	double initial_angle = 0; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	double radius = 2.0;

	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);
	cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	cv::Mat home_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle, radius);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);

	aim_idx = test_idx;	

	if(aim_idx > num_train_data_)
	{
		std::cout << "invalid test index... exiting..." << std::endl;
		exit(0);
	}

	aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
	home_frame_idx = train_target_idx.at<double>(0, 0);
	loader.LoadImage(aim_frame_idx, disp_img);
	// loader.LoadImage(home_frame_idx, home_img);
	// cv::addWeighted(disp_img, 0.0, home_img, 1.0, 0.0, disp_img);

	float diagnosis_idx[10] = {0, 10, 20, 30, 40, 50, 60, 80, 100, 146};
	for(int i = 0; i < 10; i++)
	{
		loader.LoadWeightsForDiagnosis(elips.transform_, elips, 1, dim_feature_, diagnosis_idx[i]);
		elips.SetRefEllipseParameters();
		
		SetFeature(feature, aim_idx, train_prop_, home_prop_);		
		// loader.LoadSiftKeyPoint(descriptors_, key_points_, aim_frame_idx);
		action = elips.transform_.EvaluateInvTransformation(feature);
		elips.UpdateEllipseByAction(action);  
		// elips.GetKeyPointInEllipse(descriptors_, key_points_, elips_descriptors_, elips_key_points_, elips_distance_, 1);
		// EvaluateGradientAndUpdate(feature, update_gradient_flag, aim_idx, elips, 0);			
		elips.CopyToPrev();
		CopyToPrev();			
		COLOUR c = GetColour(9 - i, 0, 9);		
		elips.DrawEllipse(disp_img, 1.0, c); // draw ellipse				
	}

	cv::imshow("explorer", disp_img); // show ellipse
	cv::waitKey(0);
	
	std::cout << "diagnosis finished..." << std::endl;	
}

void Explorer::PlotTransformationGrid()
{
	char input_dir[400];		
	char output_dir[400];		
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int home_frame_idx = 0;
	int img_height = 480;
	int img_width = 640;
	int update_gradient_flag = 0;
	int kernel_dim = 3;
	int grid_number = 3;
	int curr_pos = 0;
	double current_data = 1.0;
	double* p_current_data = &current_data;
	fL kernel_list;
	

	//double initial_x = 300; // 380; // 300; // 218; 
	//double initial_y = 193; // 220; // 260; // 230; 
	//double initial_long_axis = 30; 
	//double initial_short_axis = 12; 
	//double initial_angle = -1.0 * PI / 9; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	//double radius = 1.0; // 2.0;

	//double initial_x = 370; // 380; // 300; // 218; 
	//double initial_y = 225; // 220; // 260; // 230; 
	//double initial_long_axis = 45; 
	//double initial_short_axis = 35; 
	//double initial_angle = 0; // -1.0 * PI / 3; // 1.0 * PI / 8.5; 
	//double radius = 3.0;

	double initial_x = 350; // 380; // 300; // 218; 
	double initial_y = 160; // 220; // 260; // 230; 
	double initial_long_axis = 40; 
	double initial_short_axis = 40; 
	double initial_angle = 0; // -1.0 * PI / 20; // 1.0 * PI / 8.5; 
	double radius = 2.0;

	cv::Mat train_target_idx = cv::Mat::zeros(num_train_data_, 1, CV_64F);	
	cv::Mat test_target_idx = cv::Mat::zeros(num_test_data_, 1, CV_64F);
	cv::Mat feature = cv::Mat::zeros(dim_feature_, 1, CV_64F);
	cv::Mat action = cv::Mat::zeros(dim_action_, 1, CV_64F);
	cv::Mat grid_prop = cv::Mat::zeros(grid_number * grid_number, 2, CV_64F);
	cv::Mat grid_mu = cv::Mat::zeros(grid_number * grid_number, 2, CV_64F);
	cv::Mat disp_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	cv::Mat curr_mu = cv::Mat::zeros(1, 2, CV_64F);
	// just plot the centers... hypothetical.. do not need the real data...

	Loader loader(dim_action_, trend_number_, dir_id_, dir_);
	Ellipse elips(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle, radius);

	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx, test_target_idx);
	loader.LoadWeightsForTest(elips.transform_, 1, dim_feature_);
	loader.LoadEllipse(elips);
	elips.SetRefEllipseParameters();

	for(int i = 0; i < grid_number; i++)
	{
		for(int j = 0; j < grid_number; j++)
		{
			grid_prop.at<double>(i * grid_number + j, 0) = -1 + 2.0 / (grid_number - 1) * i;
			grid_prop.at<double>(i * grid_number + j, 1) = -1 + 2.0 / (grid_number - 1) * j;
		}
	}

	aim_idx = 0;	
	aim_frame_idx = train_target_idx.at<double>(aim_idx, 0);
	loader.LoadImage(aim_frame_idx, disp_img);
	
	for(int idx = 0; idx < grid_number * grid_number; idx++)
	{
		
		double current_data = 1.0;
		double* p_current_data = &current_data;
		feature_data_.at<double>(0, 0) = grid_prop.at<double>(idx, 0) - home_prop_.at<double>(0, 0);
		feature_data_.at<double>(1, 0) = grid_prop.at<double>(idx, 1) - home_prop_.at<double>(0, 1);
		kernel_list.clear();
		SetKernel(kernel_list, feature_data_, p_current_data, kernel_dim, curr_pos, 2, kernel_dim, 0); 
		for(int i = 0; i < kernel_list.size(); i++)
			feature.at<double>(i, 0) = kernel_list[i];

		action = elips.transform_.EvaluateInvTransformation(feature);
		elips.UpdateEllipseByAction(action); 
		elips.CopyToPrev();
		CopyToPrev();	
		curr_mu = elips.mu().t();
		curr_mu.copyTo(grid_mu.rowRange(idx, idx + 1));

		COLOUR c = GetColour(1, 0, 1);		
		elips.DrawEllipse(disp_img, 1.0, c); // draw ellipse		
		// elips.DrawEllipse(disp_img, 1.0); // draw ellipse		
	}

	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/grid_mu.bin", dir_id_);
	FileIO::WriteMatDouble(grid_mu, grid_number * grid_number, 2, output_dir);

	cv::imshow("explorer", disp_img); // show ellipse
	cv::waitKey(0);
	
	std::cout << "diagnosis finished..." << std::endl;	
}