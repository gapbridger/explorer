#ifndef _EXPLORER_H
#define _EXPLORER_H


#define GRADIENT_SCALE 100

#include <iostream>
#include <random>
#include <time.h>
#include <queue>
#include <deque>
#include <array>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

#include "../inc/fio.h"
#include "../inc/loader.h"
#include "../inc/transform.h"
#include "../inc/ellipse.h"

// typedef std::vector<double> fL;
typedef std::vector<cv::DMatch> DMatchL;

class Explorer{
private:
       
	int dir_id_; 
	int dim_action_;
	int dim_feature_;
	int trend_number_;
	int num_train_data_;
	int num_test_data_;	
	unsigned long num_iteration_;
	
    double max_exploration_range_;
    double starting_exploration_range_;	
	double range_expanding_period_;
	double avg_cost_;	
	
	std::random_device rd_;

	cv::Mat action_;
    cv::Mat train_prop_;
    cv::Mat test_prop_;    
    cv::Mat home_prop_;
	cv::Mat curr_prop_;
	cv::Mat curr_prop_matrix_;
	cv::Mat prop_diff_;
	cv::Mat prop_dist_;
	cv::Mat aim_idx_matrix_;
	cv::Mat elips_descriptors_;    
    cv::Mat elips_prev_descriptors_;
	cv::Mat elips_key_points_;
	cv::Mat elips_prev_key_points_;     
	cv::Mat elips_distance_;
	// cv::Mat elips_prev_distance_;
    cv::Mat descriptors_;   
	cv::Mat key_points_;    
    cv::Mat prev_descriptors_;    
    cv::Mat prev_key_points_;
    cv::Mat img_point_;
	cv::Mat ref_point_;
	cv::Mat prev_img_point_;    
    cv::Mat prev_ref_point_;
	cv::Mat feature_data_;
	cv::Mat matched_points_;
	cv::Mat motion_ratio_;	
	cv::Mat maha_dist_;
 
    cv::FlannBasedMatcher matcher_;

    DMatchL matches_;    

	MatL match_point_info_;
    MatL unique_match_point_;    
        
	MatL descriptors_all_;
	MatL key_points_all_;
    fL kernel_list_;
	fL path_p_1_;
	fL path_p_2_;

	double target_p_1_;
	double target_p_2_;
	double prev_target_p_1_;
	double prev_target_p_2_;
	int path_count_;
	char dir_[40];
    
public:
    // initialization
    Explorer(int dir_id, int num_iteration, int expanding_period, char* dir);
    ~Explorer();
	void SetFeature(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop);
	void SetKernel(fL& kernel_list, cv::Mat& data, double* p_current_data, int dim_left, 
		int curr_pos, int data_length, int kernel_dim, int value_flag);
	int GenerateAimIndex(std::mt19937& engine, int explore_iter);
	void GenerateLinePath(fL& path_p_1, fL& path_p_2, double target_p_1, double target_p_2, double prev_target_p_1, double prev_target_p_2);
	int GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration);
	void EvaluateGradientAndUpdate(cv::Mat& feature, int update_flag, int aim_idx, Ellipse& elips, int iteration_count);	
	void CopyToPrev();
	void Train();	
	void Test(int display_flag, int single_frame_flag, int start_idx, int end_idx, int test_idx, int test_flag, int record_img_flag); // last flag is used to signify whether use test proprioception or train proprioception
	void PlotDiagnosis(int test_idx);
	void PlotTransformationGrid();
};

struct DistCompare{
    // the fifth component is the distance...
    inline bool operator()(const cv::Mat& a, const cv::Mat& b){
        return a.at<double>(0, 0) < b.at<double>(0, 0);
    }
};


#endif
