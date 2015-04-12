#ifndef _TRANSFORM_H
#define _TRANSFORM_H
#define GRADIENT_SCALE 100000

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include <opencv2/flann.hpp>
#include "../inc/fio.h"

#include <random>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef struct {
    double r,g,b;
} COLOUR;
COLOUR GetColour(double v, double vmin, double vmax);
class Transform
{
private:
	
	// transformation structures		
	cv::Mat e_;
	// weights
	std::vector<cv::Mat> w_;
	std::vector<cv::Mat> w_grad_;
	std::vector<cv::Mat> natural_w_grad_;	
	std::vector<cv::Mat> fisher_inv_;
	std::vector<cv::Mat> transform_inv_;
	std::vector<cv::Mat> transform_;
	std::vector<cv::Mat> prev_transform_inv_;
	std::vector<cv::Mat> prev_transform_;
	std::vector<cv::Mat> transform_elements_;

    cv::Mat home_cloud_label_;
    cv::Mat vote_accumulation_;	
    cv::Mat home_cloud_template_;
    cv::Mat home_cloud_template_float_;
    // cv::flann::Index kd_trees_;
	cv::Mat tmp_;
	// cv::Mat fisher_inv_;
	
	// cv::Mat natural_w_grad_;
	// cv::Mat natural_w_grad_vec_;


	Eigen::Matrix4d eigen_transform_inv_;
	// dimensions...
	int feature_dim_;	
	int transform_dim_;
	int num_joints_;
	int num_weights_;
	int gradient_iter_;

	double w_rate_;
	std::vector<double> average_norm_;
	std::vector<double> ini_norm_;
	double lambda_;
	
	double rejection_threshold_;
	double w_natural_rate_;

	double ca_; // cos alpha
	double sa_; // sine alpha
	double cb_; // cos beta
	double sb_; // sine beta
	double cg_; // cos gamma
	double sg_; // sine gamma
	// std::vector<double> w_rate_; // including natural gradient rate...	

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> estimation_;
	pcl::Correspondences corr_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr curr_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr prev_home_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr curr_home_cloud_;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
public:
	Transform(int transform_dim_, int num_joints, double normal_learning_rate);	
	void CalcTransformInv(cv::Mat& feature);
	void CalcTransformation(cv::Mat& feature);
	void SetTransformationByElements(cv::Mat& transform, const cv::Mat& elements);
	void TransformCloud(const std::vector<cv::Mat>& input_cloud, const std::vector<cv::Mat>& transforms, std::vector<cv::Mat>& output_cloud);
	void CalculateGradient(const std::vector<cv::Mat>& matched_target_cloud, const std::vector<cv::Mat>& prediction_cloud, const std::vector<cv::Mat>& query_cloud, const cv::Mat& feature);
	std::vector<cv::Mat> get_transform();
	void CalcGradient(std::vector<cv::Mat>& target_cloud, std::vector<cv::Mat>& prediction_cloud, std::vector<cv::Mat>& query_cloud, cv::Mat& feature, std::vector<int>& segmentation_count);
    void CalcGradient(std::vector<cv::Mat>& matched_target_cloud, std::vector<cv::Mat>& prediction_cloud, std::vector<cv::Mat>& query_cloud, cv::Mat& feature, std::vector<cv::Mat>& probabilities);
	void cv2eigen(const cv::Mat& input, Eigen::Matrix4d& output);
	// gradients		
	void Update(int iter);
	void Update();
	void CopyTransformToPrev();
	// calculate inverse transformation			
	
	cv::Mat TransformDataPointInv(cv::Mat& point, int curr_flag);
	cv::Mat TransformToPreviousFrame(cv::Mat& curr_img_point);
	cv::Mat TransformToNextFrame(cv::Mat& prev_img_point);
	void Rejection(cv::Mat& diff, cv::Mat& filtered_diff, cv::Mat& query_cloud, cv::Mat& filtered_query_cloud, double threshold);
	void SegmentationAndUpdate(std::vector<cv::Mat>& prev_home_cloud, std::vector<cv::Mat>& home_cloud, cv::Mat& query_cloud, cv::Mat& feature, int iteration_count);
	void SegmentationAndUpdateFixedHomePos(std::vector<cv::Mat>& home_cloud, cv::Mat& query_cloud, cv::Mat& feature, int iteration_count);
	void ShowHomePoseLabel();
	// void GetNearestNeighborMatches(cv::Mat& target_cloud, cv::Mat& prediction_cloud, cv::Mat& matched_target_cloud, cv::Mat& cost, int cost_flag);
	void set_w_rate(double w_rate);
	void set_w_natural_rate(double natural_rate);
	// void SetLearningRate(double rate)
	// copy to previous transformation
	void CopyToPrev();
	// helper functions
	cv::Mat w(int idx);
	cv::Mat fisher_inv(int idx);
	cv::Mat natural_w_grad(int idx);
	cv::Mat transform_inv(int idx);
	cv::Mat prev_transform_inv(int idx);
	cv::Mat w_grad(int idx);
	std::vector<cv::Mat> w_grad();
	void set_w(cv::Mat& w, int idx);	
	cv::Mat get_w(int idx);
	void set_fisher_inv();
	
	// check gradient
	void CheckInvGradient();
	void Mat2PCD_Trans(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void ReOrder_Trans(std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, std::vector<cv::Mat>& input_indices);

    // functions for segmentation
    void SetHomeCloud(std::vector<cv::Mat>& home_cloud);     


};

#endif

