#include "../inc/loader.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"


Loader::Loader(int action_number, int trend_number, int dir_id, char* dir)
{
	// initialize weights
	action_number_ = action_number;
	diagnosis_number_ = action_number_ + 2;
	trend_number_ = trend_number;  
	dir_id_ = dir_id;
	// just check...
	if(action_number_ < 5 || trend_number_ < 6)
		std::cout << "directory number incorrect..." << std::endl;

	test_weights_dir_ = new char*[action_number_];  
	diagnosis_weights_dir_ = new char*[diagnosis_number_];  
	trend_dir_ = new char*[trend_number_];  

	int len = 400;
	for(int i = 0; i < action_number_; i++)
		test_weights_dir_[i] = new char[len];
	for(int i = 0; i < trend_number_; i++)
		trend_dir_[i] = new char[len]; 	
	for(int i = 0; i < diagnosis_number_; i++)
		diagnosis_weights_dir_[i] = new char[len]; 	

	sprintf(common_output_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/", dir_id_);
	sprintf(common_diagnosis_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/diagnosis_%d/", dir_id_);
	sprintf(common_data_prefix_, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/");
	strcat(common_data_prefix_, dir);
	strcat(common_data_prefix_, "/"); // march 10 2014/"); // feb 23
}

/*****************Directory Formating functions**********************/

// format test weight directory (final weight loaded for test)
void Loader::FormatWeightsForTestDirectory()
{
	// char dir_idx_str[5];
	// sprintf(dir_idx_str, "%d/", dir_id_);
	for(int i = 0; i < action_number_; i++){
		memset(&test_weights_dir_[i][0], 0, sizeof(test_weights_dir_[i]));
		strcpy(test_weights_dir_[i], common_output_prefix_);
		// strcpy(test_weights_dir_[i], "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_");
		// strcat(test_weights_dir_[i], dir_idx_str);
	}
	AppendActionWeightName(test_weights_dir_);
	for(int i = 0; i < action_number_; i++)
		strcat(test_weights_dir_[i], ".bin");	
}

void Loader::FormatWeightsForDiagnosisDirectory()
{
	// char dir_idx_str[5];
	// sprintf(dir_idx_str, "%d/", dir_id_);
	for(int i = 0; i < diagnosis_number_; i++){
		memset(&diagnosis_weights_dir_[i][0], 0, sizeof(diagnosis_weights_dir_[i]));
		strcpy(diagnosis_weights_dir_[i], common_diagnosis_prefix_);
		// strcpy(test_weights_dir_[i], "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_");
		// strcat(test_weights_dir_[i], dir_idx_str);
	}
	AppendDiagnosisWeightName(diagnosis_weights_dir_);	
}


// format trend directory
void Loader::FormatTrendDirectory()
{
	// char dir_idx_str[5];
	// sprintf(dir_idx_str, "%d/", dir_id_);
	for(int i = 0; i < trend_number_; i++){
		memset(&trend_dir_[i][0], 0, sizeof(trend_dir_[i]));
		strcpy(trend_dir_[i], common_output_prefix_); // "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_"
		// strcat(trend_dir_[i], dir_idx_str);
	}
	AppendTrendName(trend_dir_);
	for(int i = 0; i < trend_number_; i++){
		strcat(trend_dir_[i], "_trend.bin");
	}	
}

void Loader::AppendActionWeightName(char** dir_str_array){
	if(dir_str_array != NULL){
		strcat(dir_str_array[0], "xw");
		strcat(dir_str_array[1], "yw");
		strcat(dir_str_array[2], "aw");
		strcat(dir_str_array[3], "lxw");
		strcat(dir_str_array[4], "sxw");    
	}
}

void Loader::AppendDiagnosisWeightName(char** dir_str_array)
{
	if(dir_str_array != NULL)
	{
		strcat(dir_str_array[0], "xw");
		strcat(dir_str_array[1], "yw");
		strcat(dir_str_array[2], "aw");
		strcat(dir_str_array[3], "lxw");
		strcat(dir_str_array[4], "sxw");    
		strcat(dir_str_array[5], "ref_mu");    
		strcat(dir_str_array[6], "ref_cov");    
	}
}

void Loader::AppendTrendName(char** trend_dir_str_array){
	if(trend_dir_str_array != NULL){
		strcat(trend_dir_str_array[0], "xw");
		strcat(trend_dir_str_array[1], "yw");
		strcat(trend_dir_str_array[2], "aw");
		strcat(trend_dir_str_array[3], "lxw");
		strcat(trend_dir_str_array[4], "sxw");
		strcat(trend_dir_str_array[5], "cov_1_1");
		strcat(trend_dir_str_array[6], "cov_1_2");
		strcat(trend_dir_str_array[7], "cov_2_1");
		strcat(trend_dir_str_array[8], "cov_2_2");
		strcat(trend_dir_str_array[9], "mu_1");
		strcat(trend_dir_str_array[10], "mu_2");
		strcat(trend_dir_str_array[11], "idx");
	}
}

/*****************Loading functions**********************/

void Loader::LoadWeightsForTest(Transform& transform, int output_dim, int input_dim)
{
	cv::Mat current_weight = cv::Mat::zeros(output_dim, input_dim, CV_64F);	
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[0]); 
	transform.set_w_x(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[1]); 
	transform.set_w_y(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[2]); 
	transform.set_w_phi(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[3]); 
	transform.set_w_sx(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[4]); 
	transform.set_w_sy(current_weight);

}

void Loader::SaveWeightsForTest(Transform& transform, int output_dim, int input_dim)
{
	cv::Mat current_weight = cv::Mat::zeros(output_dim, input_dim, CV_64F);
	current_weight = transform.w_x();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[0]);
	current_weight = transform.w_y();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[1]);
	current_weight = transform.w_phi();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[2]);
	current_weight = transform.w_sx();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[3]);
	current_weight = transform.w_sy();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, test_weights_dir_[4]);
}

void Loader::SaveWeightsForDiagnosis(Transform& transform, Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx)
{
	char tmp_dir[20];	
	FormatWeightsForDiagnosisDirectory();
	sprintf(tmp_dir, "_%d.bin", diagnosis_idx);
	for(int i = 0; i < diagnosis_number_; i++)
		strcat(diagnosis_weights_dir_[i], tmp_dir);	

	cv::Mat current_weight = cv::Mat::zeros(output_dim, input_dim, CV_64F);
	current_weight = transform.w_x();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[0]);
	current_weight = transform.w_y();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[1]);
	current_weight = transform.w_phi();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[2]);
	current_weight = transform.w_sx();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[3]);
	current_weight = transform.w_sy();
	FileIO::WriteMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[4]);

	cv::Mat elips_ref_mu = ellipse.ref_mu();
	FileIO::WriteMatDouble(elips_ref_mu, 2, 1, diagnosis_weights_dir_[5]);
	cv::Mat elips_ref_cov = ellipse.ref_cov();
	FileIO::WriteMatDouble(elips_ref_cov, 2, 2, diagnosis_weights_dir_[6]);
}

void Loader::LoadWeightsForDiagnosis(Transform& transform, Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx)
{
	char tmp_dir[20];	
	FormatWeightsForDiagnosisDirectory();
	sprintf(tmp_dir, "_%d.bin", diagnosis_idx);
	for(int i = 0; i < diagnosis_number_; i++)
		strcat(diagnosis_weights_dir_[i], tmp_dir);	

	cv::Mat current_weight = cv::Mat::zeros(output_dim, input_dim, CV_64F);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[0]); 
	transform.set_w_x(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[1]); 
	transform.set_w_y(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[2]); 
	transform.set_w_phi(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[3]); 
	transform.set_w_sx(current_weight);
	FileIO::ReadMatDouble(current_weight, output_dim, input_dim, diagnosis_weights_dir_[4]); 
	transform.set_w_sy(current_weight);

		cv::Mat elips_ref_mu = cv::Mat::zeros(2, 1, CV_64F);
	FileIO::ReadMatDouble(elips_ref_mu, 2, 1, diagnosis_weights_dir_[5]);
	ellipse.set_ref_mu(elips_ref_mu);
	
	cv::Mat elips_ref_cov = cv::Mat::zeros(2, 2, CV_64F);
	FileIO::ReadMatDouble(elips_ref_cov, 2, 2, diagnosis_weights_dir_[6]);
	ellipse.set_ref_cov(elips_ref_cov);	

}

// save value trend: either output average value or weight norm...
void Loader::SaveTrend(fL* trend_array, int trend_number, int append_flag)
{
  // FormatValueDirectory(dir_idx, true); // should be called in constructor
	int data_len = 0;
	cv::Mat data;
	for(int i = 0; i < trend_number; i++){
		data_len = trend_array[i].size();
		data = cv::Mat::zeros(data_len, 1, CV_64F);
		for(int j = 0; j < data_len; j++)
			data.at<double>(j, 0) = trend_array[i][j];
		FileIO::RecordMatDouble(data, data_len, 1, trend_dir_[i], append_flag); 
	}
}

void Loader::SaveEllipse(Ellipse& ellipse)
{
	char output_dir[400];	
	// strcat(output_dir, dir_idx_str);
	strcpy(output_dir, common_output_prefix_);
	strcat(output_dir, "ref_mu.bin");
	// sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/ref_mu.bin", dir_id_);
	
	cv::Mat elips_ref_mu = ellipse.ref_mu();
	FileIO::WriteMatDouble(elips_ref_mu, 2, 1, output_dir);
	strcpy(output_dir, common_output_prefix_);
	strcat(output_dir, "ref_cov.bin");
	// sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/ref_cov.bin", dir_id_);
	cv::Mat elips_ref_cov = ellipse.ref_cov();
	FileIO::WriteMatDouble(elips_ref_cov, 2, 2, output_dir);
}

void Loader::LoadEllipse(Ellipse& ellipse)
{
	char input_dir[400];
	// mu
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/ref_mu.bin", dir_id_);
	strcpy(input_dir, common_output_prefix_);
	strcat(input_dir, "ref_mu.bin");
	cv::Mat elips_ref_mu = cv::Mat::zeros(2, 1, CV_64F);
	FileIO::ReadMatDouble(elips_ref_mu, 2, 1, input_dir);
	ellipse.set_ref_mu(elips_ref_mu);
	// cov
	strcpy(input_dir, common_output_prefix_);
	strcat(input_dir, "ref_cov.bin");
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_%d/ref_cov.bin", dir_id_);
	cv::Mat elips_ref_cov = cv::Mat::zeros(2, 2, CV_64F);
	FileIO::ReadMatDouble(elips_ref_cov, 2, 2, input_dir);
	ellipse.set_ref_cov(elips_ref_cov);
}

char** Loader::test_weight_dir(){
  return test_weights_dir_;
}

void Loader::LoadLearningRates(Ellipse& ellipse)
{
	char input_dir[400];
	int num_rates = 6;
	cv::Mat rates = cv::Mat::zeros(num_rates, 1, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/expansion/input/rate_%d.bin", dir_id_);
	FileIO::ReadMatDouble(rates, num_rates, 1, input_dir);
	ellipse.transform_.SetLearningRates(rates.at<double>(0, 0), rates.at<double>(1, 0), rates.at<double>(2, 0), rates.at<double>(3, 0), rates.at<double>(4, 0));
	ellipse.set_eta(rates.at<double>(5, 0));
}

void Loader::LoadImage(int frame_idx, cv::Mat& disp_img)
{
	// only load "current" image 
	char input_dir[400];
	char tmp_dir[40];
	cv::Mat img;
	sprintf(tmp_dir, "images/%d.pgm", frame_idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);		
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/images/%d.pgm", frame_idx);
	img = cv::imread(input_dir, CV_LOAD_IMAGE_GRAYSCALE);
	cv::cvtColor(img, disp_img, CV_GRAY2RGB);
}

// load sift key point
void Loader::LoadSiftKeyPoint(cv::Mat& descriptors, cv::Mat& key_points, int frame_idx)
{
	int num_rows = 0; 
	int num_cols = 0;
	char input_dir[400];
	char tmp_dir[40];
	cv::Mat data_h = cv::Mat::zeros(1, 1, CV_64F);
	cv::Mat data_w = cv::Mat::zeros(1, 1, CV_64F);
	sprintf(tmp_dir, "descriptors/%d_w.bin", frame_idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_w.bin", frame_idx);
	FileIO::ReadFloatMatToDouble(data_w, 1, 1, input_dir);
	sprintf(tmp_dir, "descriptors/%d_h.bin", frame_idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_h.bin", frame_idx);
	FileIO::ReadFloatMatToDouble(data_h, 1, 1, input_dir);
	num_rows = (int)data_h.at<double>(0, 0); 
	num_cols = (int)data_w.at<double>(0, 0); 
	descriptors.create(num_rows, num_cols, CV_32F);
	key_points.create(num_rows, 2, CV_32F);
	sprintf(tmp_dir, "descriptors/%d.bin", frame_idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d.bin", frame_idx);
	FileIO::ReadMatFloat(descriptors, num_rows, num_cols, input_dir);
	sprintf(tmp_dir, "descriptors/%d_pt.bin", frame_idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_pt.bin", frame_idx);
	FileIO::ReadMatFloat(key_points, num_rows, 2, input_dir);
}

void Loader::LoadAllSiftKeyPoint(MatL& descriptors_list, MatL& key_points_list, int start_idx, int end_idx)
{
	int num_rows = 0; 
	int num_cols = 0;
	char input_dir[400];
	char tmp_dir[40];
	cv::Mat data_h = cv::Mat::zeros(1, 1, CV_64F);
	cv::Mat data_w = cv::Mat::zeros(1, 1, CV_64F);

	for(int idx = start_idx; idx < end_idx; idx++)
	{
		sprintf(tmp_dir, "descriptors/%d_w.bin", idx);
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, tmp_dir);
		FileIO::ReadFloatMatToDouble(data_w, 1, 1, input_dir);

		sprintf(tmp_dir, "descriptors/%d_h.bin", idx);
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, tmp_dir);
		FileIO::ReadFloatMatToDouble(data_h, 1, 1, input_dir);

		num_rows = (int)data_h.at<double>(0, 0); 
		num_cols = (int)data_w.at<double>(0, 0); 

		cv::Mat descriptors = cv::Mat::zeros(num_rows, num_cols, CV_32F);
		cv::Mat key_points = cv::Mat::zeros(num_rows, 2, CV_32F);

		sprintf(tmp_dir, "descriptors/%d.bin", idx);
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, tmp_dir);
		FileIO::ReadMatFloat(descriptors, num_rows, num_cols, input_dir);

		sprintf(tmp_dir, "descriptors/%d_pt.bin", idx);
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, tmp_dir);		
		FileIO::ReadMatFloat(key_points, num_rows, 2, input_dir);

		descriptors_list.push_back(descriptors);
		key_points_list.push_back(key_points);

		if(idx % 100 == 1)
			std::cout << "current frame: " << idx << std::endl;
	}
	
}

// load explained variance
void Loader::LoadProprioception(int num_train_data, int num_test_data, cv::Mat& train_prop, cv::Mat& test_prop, cv::Mat& home_prop, cv::Mat& train_target_idx, cv::Mat& test_target_idx)
{
	char input_dir[400];
	cv::Mat p_tmp_train = cv::Mat::zeros(num_train_data, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_p0.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_train, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_train, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_p0.bin");
	p_tmp_train.copyTo(train_prop.colRange(0, 1));
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_p3.bin");
	FileIO::ReadFloatMatToDouble(p_tmp_train, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_train, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_p3.bin");
	p_tmp_train.copyTo(train_prop.colRange(1, 2));

	cv::Mat p_tmp_test = cv::Mat::zeros(num_test_data, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_p0.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_test, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_test, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_p0.bin");
	p_tmp_test.copyTo(test_prop.colRange(0, 1));
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_p3.bin");
	FileIO::ReadFloatMatToDouble(p_tmp_test, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_test, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_p3.bin");
	p_tmp_test.copyTo(test_prop.colRange(1, 2));
  
	cv::Mat p_tmp_home = cv::Mat::zeros(2, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "prop_home.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_home, 2, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_home, 2, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/prop_home.bin");
	home_prop = p_tmp_home.t();
	std::cout << "home prop: " << home_prop.at<double>(0, 0) << " " << home_prop.at<double>(0, 1) << std::endl;
	
	// train frame index
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_prop_idx.bin");	
	FileIO::ReadFloatMatToDouble(train_target_idx, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(train_target_idx, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_prop_idx.bin");
	// test frame index
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_prop_idx.bin");	
	FileIO::ReadFloatMatToDouble(test_target_idx, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(test_target_idx, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_prop_idx.bin");
}

void Loader::RecordSiftKeyPoints()
{
	char input_dir[400];
	char output_dir[400];
	char tmp_dir[40];
	int start_frame = 0; // 19201;
	int end_frame = 16400; // 19000; // 23247; // 19228; // 11374;
	int image_width = 640;
	int image_height = 480;
	cv::Mat curr_img;
	cv::Mat mask;
	cv::Mat descriptor;
	cv::SiftFeatureDetector sift_detector(0, 3, 0.04, 10, 1.2);
    cv::FlannBasedMatcher matcher;
    cv::SiftDescriptorExtractor sift_extractor;	
	std::vector<cv::KeyPoint> key_point_list;

	// mask...
	mask = cv::Mat::zeros(image_height, image_width, CV_64F);
	sprintf(tmp_dir, "mask.bin");
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	FileIO::ReadFloatMatToDouble(mask, mask.rows, mask.cols, input_dir);
	// "D:/Document/HKUST/Year 5/Research/Data/Arm Images/march_11_2014/mask.bin"
	mask.convertTo(mask, CV_8UC1);		
	//cv::imshow("mask", mask); // show ellipse
	//cv::waitKey(0);
	for(int frame_idx = start_frame; frame_idx <= end_frame; frame_idx++)
	{
		sprintf(tmp_dir, "images/%d.pgm", frame_idx);
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, tmp_dir);
		curr_img = cv::imread(input_dir, CV_LOAD_IMAGE_GRAYSCALE);				
		sift_detector.detect(curr_img, key_point_list , mask);//, mask); // , evarBinary); // evDstBinaryCurr);
		sift_extractor.compute(curr_img, key_point_list, descriptor);
		// recording
		// assign key points value
		cv::Mat key_point_matrix = cv::Mat::zeros(descriptor.rows, 2, CV_32F);
		cv::Mat num_cols = cv::Mat::zeros(1, 1, CV_32F);
        cv::Mat num_rows = cv::Mat::zeros(1, 1, CV_32F);
		num_rows.at<float>(0, 0) = descriptor.rows;
        num_cols.at<float>(0, 0) = descriptor.cols;
        for(int i = 0; i < descriptor.rows; i++)
		{
			key_point_matrix.at<float>(i, 0) = key_point_list[i].pt.x;
			key_point_matrix.at<float>(i, 1) = key_point_list[i].pt.y;
        }
		sprintf(tmp_dir, "descriptors/%d_w.bin", frame_idx);
		strcpy(output_dir, common_data_prefix_);
		strcat(output_dir, tmp_dir);
		// sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_w.bin", frame_idx);
		FileIO::WriteMatFloat(num_cols, 1, 1, output_dir);        
		sprintf(tmp_dir, "descriptors/%d_h.bin", frame_idx);
		strcpy(output_dir, common_data_prefix_);
		strcat(output_dir, tmp_dir);
        // sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_h.bin", frame_idx);
		FileIO::WriteMatFloat(num_rows, 1, 1, output_dir);
		sprintf(tmp_dir, "descriptors/%d_pt.bin", frame_idx);
		strcpy(output_dir, common_data_prefix_);
		strcat(output_dir, tmp_dir);
        // sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d_pt.bin", frame_idx);
		FileIO::WriteMatFloat(key_point_matrix, key_point_matrix.rows, key_point_matrix.cols, output_dir);
		sprintf(tmp_dir, "descriptors/%d.bin", frame_idx);
		strcpy(output_dir, common_data_prefix_);
		strcat(output_dir, tmp_dir);
        // sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/descriptors/%d.bin", frame_idx);
        FileIO::WriteMatFloat(descriptor, descriptor.rows, descriptor.cols, output_dir);

		if(frame_idx % 100 == 1)
			std::cout << "current frame: " << frame_idx << std::endl;
	}
}