#include <iostream>
#include "../inc/test.h"

void UnifiedLearningTest::SetUp()
{
	test_data_dir_prefix_ = new char[200];
	sprintf(test_data_dir_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/test/");
	double_epsilon_ = 1e-7;
}

void UnifiedLearningTest::TearDown()
{
	delete test_data_dir_prefix_;
}


TEST_F(UnifiedLearningTest, TestCalcFeature2D)
{
	int dim_sinusoidal = 3;
	int num_joints_2d = 2;
	int dim_feature_2d = pow((double)dim_sinusoidal, num_joints_2d) - 1;

	char test_prop_2d_dir[200];
	char test_prop_feature_2d_dir[200];
	strcpy(test_prop_2d_dir, test_data_dir_prefix_);
	strcat(test_prop_2d_dir, "test_prop_2d.bin");
	strcpy(test_prop_feature_2d_dir, test_data_dir_prefix_);
	strcat(test_prop_feature_2d_dir, "test_prop_feature_2d.bin");

	// column vectors
	cv::Mat test_prop_2d = cv::Mat::zeros(1, num_joints_2d, CV_64F);
	cv::Mat test_prop_feature_2d = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);
	cv::Mat expected_test_prop_feature_2d = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);

	FileIO::ReadMatDouble(test_prop_2d, 1, num_joints_2d, test_prop_2d_dir);
	FileIO::ReadMatDouble(expected_test_prop_feature_2d, dim_feature_2d, 1, test_prop_feature_2d_dir);

	Explorer::SetFeature(test_prop_feature_2d, feature_zero, num_joints_2d, test_prop_2d);
	for(int i = 0; i < dim_feature_2d; i++)
	{
		EXPECT_NEAR(expected_test_prop_feature_2d.at<double>(i, 0), test_prop_feature_2d.at<double>(i, 0), double_epsilon_);
	}
}

TEST_F(UnifiedLearningTest, TestCalcFeature3D)
{
	int dim_sinusoidal = 3;
	int num_joints_3d = 3;
	int dim_feature_3d = pow((double)dim_sinusoidal, num_joints_3d) - 1;

	char test_prop_3d_dir[200];
	char test_prop_feature_3d_dir[200];
	strcpy(test_prop_3d_dir, test_data_dir_prefix_);
	strcat(test_prop_3d_dir, "test_prop_3d.bin");
	strcpy(test_prop_feature_3d_dir, test_data_dir_prefix_);
	strcat(test_prop_feature_3d_dir, "test_prop_feature_3d.bin");

	// column vectors
	cv::Mat test_prop_3d = cv::Mat::zeros(1, num_joints_3d, CV_64F);
	cv::Mat test_prop_feature_3d = cv::Mat::zeros(dim_feature_3d, 1, CV_64F);
	cv::Mat expected_test_prop_feature_3d = cv::Mat::zeros(dim_feature_3d, 1, CV_64F);
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_3d, 1, CV_64F);

	FileIO::ReadMatDouble(test_prop_3d, 1, num_joints_3d, test_prop_3d_dir);
	FileIO::ReadMatDouble(expected_test_prop_feature_3d, dim_feature_3d, 1, test_prop_feature_3d_dir);

	Explorer::SetFeature(test_prop_feature_3d, feature_zero, num_joints_3d, test_prop_3d);
	for(int i = 0; i < dim_feature_3d; i++)
	{
		EXPECT_NEAR(expected_test_prop_feature_3d.at<double>(i, 0), test_prop_feature_3d.at<double>(i, 0), double_epsilon_);
	}
}


TEST_F(UnifiedLearningTest, TestCalculateGradient)
{
	char test_query_cloud_dir[200];
	char test_target_cloud_dir[200];
	int num_joint = 1; // currently only have data for 1 joint
	int num_cloud_points = 1000;
	int dim_transform = 4;
	int dim_feature = pow(3.0, num_joint) - 1;
	int num_transform_elements = dim_transform * (dim_transform - 1);
	double disturb_value = 0.0001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;

	std::vector<cv::Mat> predicted_cloud(num_joint);   
    std::vector<cv::Mat> target_cloud(num_joint); 
    std::vector<cv::Mat> query_cloud(num_joint);   
	std::vector<cv::Mat> w_grad(num_joint);   

	for(int i = 0; i < num_joint; i++)
	{
		predicted_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		target_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		query_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		w_grad[i] = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
	}

	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "test_query_cloud.bin");
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "test_target_cloud.bin");

	FileIO::ReadMatDouble(target_cloud[0], num_cloud_points, dim_transform, test_target_cloud_dir);
	FileIO::ReadMatDouble(query_cloud[0], num_cloud_points, dim_transform, test_query_cloud_dir);

	cv::Mat feature = cv::Mat::zeros(dim_feature, 1, CV_64F);
	cv::randu(feature, cv::Scalar::all(0), cv::Scalar::all(1));
	Transform transform(dim_transform, num_joint, 0.01);
	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud, transform.get_transform(), predicted_cloud);
	transform.CalculateGradient(target_cloud, predicted_cloud, query_cloud, feature);
	w_grad = transform.w_grad();

	cv::Mat diff; // , filtered_diff, filtered_query_cloud;
	cv::Mat disturb, dist, new_w;
	double e_1, e_2;
	for(int idx = 0; idx < num_joint; idx++)
	{
		for(int i = 0; i < num_transform_elements; i++)
		{
			for(int j = 0; j < dim_feature; j++)
			{
				disturb = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
				disturb.at<double>(i, j) = disturb_value;
				// e_1
				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud, transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
				cv::reduce(diff, dist, 0, CV_REDUCE_AVG);
				e_1 = dist.at<double>(0, 0);

				// e_2
				new_w = transform.get_w(idx) - 2 * disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud, transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
				cv::reduce(diff, dist, 0, CV_REDUCE_AVG);
				e_2 = dist.at<double>(0, 0);

				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);

				numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
				analytical_gradient = w_grad[idx].at<double>(i, j);

				EXPECT_NEAR(numerical_gradient, analytical_gradient, double_epsilon_);
			}
		}
	}
}

TEST_F(UnifiedLearningTest, TestBuildModelGraph)
{
	// load testing home cloud
	char test_input_dir[400];
	int cloud_size;
	int dim = 4;
	double neighborhood_range = 1e-2;
	double max_num_neighbor = 20;
	cv::Mat home_cloud_indices, home_cloud_min_dists;
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	
	cv::Mat test_home_cloud = cv::Mat::ones(cloud_size, dim, CV_64F);	
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud.bin");
	FileIO::ReadMatDouble(test_home_cloud.colRange(0, dim - 1), cloud_size, dim - 1, test_input_dir);		

	Explorer::BuildModelGraph(test_home_cloud, 1, home_cloud_indices, home_cloud_min_dists, neighborhood_range, max_num_neighbor);

	// randomly assert several values...
	EXPECT_EQ(test_home_cloud.rows, home_cloud_indices.rows);
	EXPECT_EQ(max_num_neighbor, home_cloud_indices.cols);
	EXPECT_EQ(test_home_cloud.rows, home_cloud_min_dists.rows);
	EXPECT_EQ(max_num_neighbor, home_cloud_min_dists.cols);

	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_neighbor_dists.bin");
	cv::Mat expected_test_home_cloud_neighbor_dists = cv::Mat::ones(cloud_size, max_num_neighbor, CV_64F);	
	FileIO::ReadMatDouble(expected_test_home_cloud_neighbor_dists, cloud_size, max_num_neighbor, test_input_dir);		

	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < max_num_neighbor; j++)
		{
			EXPECT_NEAR((float)expected_test_home_cloud_neighbor_dists.at<double>(i, j), home_cloud_min_dists.at<float>(i, j), double_epsilon_);
		}
	}
}

TEST_F(UnifiedLearningTest, TestInitializeModelLabel)
{
	// load testing home cloud
	char test_input_dir[400];
	char min_dist_dir[40];
	int cloud_size = 0;
	int num_joints = 3;
	double sigma = 1.0;
	cv::Mat home_cloud_label, expected_home_cloud_label;
	std::vector<cv::Mat> min_dists(num_joints);
	// specify cloud size
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	// initialize cloud label
	home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	expected_home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label.bin");
	FileIO::ReadMatDouble(expected_home_cloud_label, cloud_size, num_joints, test_input_dir);
	for(int i = 0; i < num_joints; i++)
	{
		min_dists[i] = cv::Mat::zeros(cloud_size, 1, CV_32F);
		strcpy(test_input_dir, test_data_dir_prefix_);
		sprintf(min_dist_dir, "min_dist_%d.bin", i);
		strcat(test_input_dir, min_dist_dir);
		FileIO::ReadMatFloat(min_dists[i], cloud_size, 1, test_input_dir);
	}
	Explorer::InitializeModelLabel(min_dists, num_joints, home_cloud_label);
	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < num_joints; j++)
		{
			EXPECT_EQ(expected_home_cloud_label.at<double>(i, j), home_cloud_label.at<double>(i, j));
		}
	}
}

TEST_F(UnifiedLearningTest, TestIteratedConditionalMode)
{
	// load testing home cloud
	char test_input_dir[400];
	char min_dist_dir[40];
	int cloud_size = 0;
	int num_joints = 3;
	int max_num_neighbors = 20;
	std::vector<cv::Mat> min_dists(num_joints);
	cv::Mat home_cloud_label, home_cloud_neighbor_indices, potential, expected_potential, expected_home_cloud_label;
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	// read in cloud label
	home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	potential = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label.bin");
	FileIO::ReadMatDouble(home_cloud_label, cloud_size, num_joints, test_input_dir);
	// read in neighbor indices
	home_cloud_neighbor_indices = cv::Mat::zeros(cloud_size, max_num_neighbors, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_neighbor_indices.bin");
	FileIO::ReadMatDouble(home_cloud_neighbor_indices, cloud_size, max_num_neighbors, test_input_dir);
	home_cloud_neighbor_indices.convertTo(home_cloud_neighbor_indices, CV_32S);
	// read in expected potential
	expected_potential = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_potential.bin");
	FileIO::ReadMatDouble(expected_potential, cloud_size, num_joints, test_input_dir);
	// read in expected label after update
	expected_home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label_after_icm.bin");
	FileIO::ReadMatDouble(expected_home_cloud_label, cloud_size, num_joints, test_input_dir);
	// read in min dists
	for(int i = 0; i < num_joints; i++)
	{
		min_dists[i] = cv::Mat::zeros(cloud_size, 1, CV_32F);
		strcpy(test_input_dir, test_data_dir_prefix_);
		sprintf(min_dist_dir, "min_dist_%d.bin", i);
		strcat(test_input_dir, min_dist_dir);
		FileIO::ReadMatFloat(min_dists[i], cloud_size, 1, test_input_dir);
	}
	// execute the ICM algorithm for one iteration with beta equal to 1
	// Explorer::IteratedConditionalModes(home_cloud_neighbor_indices, home_cloud_label, potential, num_joints, 1, max_num_neighbors, 1.0);
	Explorer::IteratedConditionalModes(home_cloud_neighbor_indices, min_dists, home_cloud_label, potential, num_joints, 1, max_num_neighbors, 1.0, 1.0);
	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < num_joints; j++)
		{
			EXPECT_EQ(expected_home_cloud_label.at<double>(i, j), home_cloud_label.at<double>(i, j));
			EXPECT_NEAR(expected_potential.at<double>(i, j), potential.at<double>(i, j), 1e-4);
		}
	}
}

