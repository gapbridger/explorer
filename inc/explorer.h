#ifndef _EXPLORER_H
#define _EXPLORER_H


#define GRADIENT_SCALE 100000
#define PI 3.14159265

#include <iostream>
#include <random>
#include <time.h>
#include <queue>
#include <deque>
#include <array>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

#include "../inc/fio.h"
#include "../inc/loader.h"
#include "../inc/transform.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "gtest/gtest.h"


// typedef std::vector<double> fL;

class Explorer{
private:
       
	int id_; 	
	int dim_feature_;
	int num_joints_;	
	int num_trend_;
	int train_data_size_;
	int test_data_size_;
	long train_iteration_;	
	long expand_iteration_;
	int path_count_;	
	int dim_transform_;
	int num_weights_;
	int icm_iteration_;
	int max_num_neighbors_;
	char data_set_[100];
	
    double max_exploration_range_;
    double ini_exploration_range_;	
	double avg_cost_;	
	double normal_learning_rate_;
	double neighborhood_range_;
	double icm_beta_;
	double icm_sigma_;
	
	std::random_device rd_;

	std::vector<double> targets_;
	std::vector<double> prev_targets_;
	std::vector<std::vector<double>> path_;
	std::vector<double> kernel_list_;

	cv::Mat joint_idx_;
	cv::Mat joint_range_limit_;
	cv::Mat action_;
    cv::Mat train_prop_;
	cv::Mat train_target_idx_;    
	cv::Mat test_prop_;
	cv::Mat test_target_idx_;   
    cv::Mat home_prop_;
	cv::Mat curr_prop_;
	// cv::Mat curr_prop_matrix_;
	cv::Mat prop_diff_;
	cv::Mat prop_dist_;
	cv::Mat aim_idx_matrix_;
	cv::Mat feature_;
	cv::Mat feature_home_;

	cv::Mat explore_path_target_;
	cv::Mat prev_explore_path_target_;
	cv::Mat explore_path_kdtree_indices_;
	cv::Mat explore_path_kdtree_dists_;

	cv::Mat cloud_;
	cv::Mat prev_cloud_;
	std::vector<cv::Mat> home_cloud_;
	cv::Mat home_cloud_label_; // number of clouds are treated as number of columns
	cv::Mat home_cloud_indices_;
	cv::Mat home_cloud_min_dists_;
	std::vector<cv::Mat> predicted_cloud_;
	cv::Mat tmp_cloud_;

	Transform transform_;

	// int cloud_scale_;
    
public:
    // initialization
    Explorer(int dir_id, char* data_set, int train_iteration, int expand_iteration, int dim_transform, int num_joints, double normal_learning_rate, double ini_exploration_range, 
		int train_data_size, int test_data_size, const cv::Mat& joint_idx, const cv::Mat& joint_range_limit, double neighborhood_range, int icm_iteration, double icm_beta, double icm_sigma, int max_num_neighbors);
    ~Explorer();	
	void RecordData(Loader& loader, std::vector<std::vector<double>>& trend_array, int aim_idx, int iteration_count, int record_trend_interval, int record_diagnosis_interval);
	// void Segment(std::vector<cv::Mat>& matched_target_cloud, const cv::Mat& home_cloud_label, const cv::Mat& target_cloud, const cv::Mat& index, int num_joints);
	void Segment(std::vector<cv::Mat>& segmented_target_cloud, std::vector<cv::Mat>& segmented_home_cloud, std::vector<cv::Mat>& segmented_prediction_cloud, const cv::Mat& home_cloud_label, const cv::Mat& target_cloud, const cv::Mat& home_cloud, 
					   const std::vector<cv::Mat>& prediction_cloud, const std::vector<cv::Mat>& indices, int num_joints);
	void ShowLearningProgress(int iteration_count);
	void GenerateLinePath(std::vector<std::vector<double>>& path, std::vector<double>& targets, std::vector<double>& prev_targets);
	int GenerateAimIndex(std::mt19937& engine, cv::flann::Index& kd_trees, std::vector<int>& path, int iteration_count, const cv::Mat& scale);
	void LoadHomeCloud(Loader& loader);
	int GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration);
	void Train();
	// void Test(int single_frame_flag, int display_flag, int test_idx);
	void Test(bool single_frame, bool display, int test_idx);
	void ShowTransformationGrid(int num_grid, int weight_idx);
	void Explorer::LearningFromPointCloudTest();
	void DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud);
	void DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud);
	void ShowCloudSequence();
	static void SetFeature(cv::Mat& feature, cv::Mat& feature_home, int num_joints, const cv::Mat& curr_prop);
	void PreprocessingAndSavePointCloud();
	void PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat& cloud_mat);
	void Mat2PCD(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void ReOrder(cv::Mat& input, cv::Mat& output, cv::Mat& input_indices);
	void RecordingTrend(Transform& transform, Loader& loader, std::vector<std::vector<double>>& trend_array, int iter, int write_trend_interval, int aim_idx);
	static void BuildModelGraph(const cv::Mat& home_cloud, int num_joints, cv::Mat& home_cloud_indices, cv::Mat& home_cloud_min_dists, double neighborhood_range, int max_num_neighbors);
	static void InitializeModelLabel(const std::vector<cv::Mat>& min_dists, int num_joints, cv::Mat& home_cloud_label);
	static void IteratedConditionalModes(const cv::Mat& home_cloud_neighbor_indices, const std::vector<cv::Mat>& min_dists, cv::Mat& home_cloud_label, cv::Mat& potential, int num_joints, int icm_iterations, int max_num_neighbors, double beta, double sigma);
	void UpdateTransform();
};

struct DistCompare{
    // the fifth component is the distance...
    inline bool operator()(const cv::Mat& a, const cv::Mat& b){
        return a.at<double>(0, 0) < b.at<double>(0, 0);
    }
};


#endif
