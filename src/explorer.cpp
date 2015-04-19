// with google c++ coding style here
#include "../inc/explorer.h"

// constructor
Explorer::Explorer(int id, 
				   char* data_set,
				   int train_iteration, 
				   int expand_iteration, 
				   int dim_transform, 
				   int num_joints, 
				   double normal_learning_rate,
				   double ini_exploration_range,
				   int train_data_size,
				   int test_data_size,
				   const cv::Mat& joint_idx,
				   const cv::Mat& joint_range_limit,
				   double neighborhood_range, 
				   int icm_iteration, 
				   double icm_beta, 
				   double icm_sigma,
				   int max_num_neighbors) 
	: transform_(dim_transform, 
				 num_joints, 
				 normal_learning_rate)	  
{    	
	sprintf(data_set_, data_set);		
	id_ = id; 	
	train_iteration_ = train_iteration;
	expand_iteration_ = expand_iteration;
	num_joints_ = num_joints;
	dim_transform_ = dim_transform;
	num_weights_ = dim_transform_ * (dim_transform_ - 1);
	train_data_size_ = train_data_size;
	test_data_size_ = test_data_size;
	normal_learning_rate_ = normal_learning_rate;
	// 3 is the dim of sinusoidal component, 1 sin cos
	dim_feature_ = pow(3.0, num_joints_) - 1;
	// this should also be changable and don't want to put it in config...
	num_trend_ = num_weights_ * num_joints + 1;		
	path_count_ = 0;		
    max_exploration_range_ = 1;
    ini_exploration_range_ = ini_exploration_range;	
	avg_cost_ = 0;	
	
	targets_ = std::vector<double>(num_joints_);
	prev_targets_ = std::vector<double>(num_joints_);
	explore_path_target_ = cv::Mat::zeros(1, num_joints_, CV_64F);
	prev_explore_path_target_ = cv::Mat::zeros(1, num_joints_, CV_64F);
	train_prop_ = cv::Mat::zeros(train_data_size_, num_joints_, CV_64F);    
	test_prop_ = cv::Mat::zeros(test_data_size_, num_joints_, CV_64F);    
    home_prop_ = cv::Mat::zeros(1, num_joints_, CV_64F); // previous column, now row
	curr_prop_ = cv::Mat::zeros(1, num_joints, CV_64F); // previous column, now row
	prop_diff_ = cv::Mat::zeros(train_data_size_, num_joints_, CV_64F);
	prop_dist_ = cv::Mat::zeros(train_data_size_, 1, CV_64F);
	train_target_idx_ = cv::Mat::zeros(train_data_size_, 1, CV_64F);    
	test_target_idx_ = cv::Mat::zeros(test_data_size_, 1, CV_64F);    
	aim_idx_matrix_ = cv::Mat::zeros(train_data_size_, 1, CV_64F);
	feature_ = cv::Mat::zeros(dim_feature_, 1, CV_64F);	
	feature_home_ = cv::Mat::zeros(dim_feature_, 1, CV_64F);	
	// home_cloud_ = std::vector<cv::Mat>(num_joints_);
	predicted_cloud_ = std::vector<cv::Mat>(num_joints_);
	joint_idx.copyTo(joint_idx_);
	explore_path_kdtree_indices_ = cv::Mat::zeros(train_data_size_, 1, CV_32S);
	explore_path_kdtree_dists_ = cv::Mat::zeros(train_data_size_, 1, CV_32F);

	joint_idx_ = cv::Mat::zeros(num_joints_, 1, CV_64F);
	joint_range_limit_ = cv::Mat::zeros(num_joints_, 2, CV_64F);
	joint_idx.copyTo(joint_idx_);
	joint_range_limit.copyTo(joint_range_limit_);

	
	neighborhood_range_ = neighborhood_range;
	icm_beta_ = icm_beta;
	icm_sigma_ = icm_sigma;
	icm_iteration_ = icm_iteration;
	max_num_neighbors_ = max_num_neighbors;

}

Explorer::~Explorer()
{
}


void Explorer::BatchTrain()
{
	char input_dir[400];
	char output_dir[400];	
	int record_trend_interval = 2;
	int record_diagnosis_interval = 2;
	int home_frame_idx = 0;
	cv::Mat initial_cloud_display = cv::Mat::zeros(num_joints_, 1, CV_64F);
	// temporary float variables
	std::mt19937 engine(rd_());		
	cv::Mat predicted_cloud_f; 
	cv::Mat train_prop_f;
	cv::Mat cloud_f;
	std::vector<cv::Mat> cloud_f_batch(train_data_size_);
	// batch variables...
	std::vector<cv::Mat> cloud_batch(train_data_size_);
	std::vector<std::vector<cv::Mat>> predicted_cloud_batch(train_data_size_, std::vector<cv::Mat>(num_joints_));
	std::vector<cv::flann::Index> cloud_kd_tree_batch;
	std::vector<std::vector<cv::Mat>> indices_batch(train_data_size_, std::vector<cv::Mat>(num_joints_));
    std::vector<std::vector<cv::Mat>> min_dists_batch(train_data_size_, std::vector<cv::Mat>(num_joints_));    
	cv::Mat feature_batch = cv::Mat::zeros(dim_feature_, train_data_size_, CV_64F);
	// segmentation variables
	std::vector<std::vector<cv::Mat>> segmented_target_cloud(train_data_size_, std::vector<cv::Mat>(num_joints_));
	std::vector<std::vector<cv::Mat>> segmented_home_cloud(train_data_size_, std::vector<cv::Mat>(num_joints_));
	std::vector<std::vector<cv::Mat>> segmented_prediction_cloud(train_data_size_, std::vector<cv::Mat>(num_joints_));
	std::vector<cv::Mat> avg_min_dists(num_joints_);
	// others	
	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));
	cv::Mat home_cloud_neighbor_indices, home_cloud_neighbor_dists;
	// loader initialization
	Loader loader(num_weights_, num_joints_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	// load proprioception
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	train_prop_.convertTo(train_prop_f, CV_32F);
	home_frame_idx = train_target_idx_.at<double>(0, 0);
	// loading cloud
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	cv::Mat home_cloud_label = cv::Mat::zeros(home_cloud_.rows, num_joints_, CV_64F);
	cv::Mat potential = cv::Mat::zeros(home_cloud_.rows, num_joints_, CV_64F);
	
	LoadCloudInBatch(loader, train_data_size_, cloud_batch);
	// initialize home cloud neighbors	
	BuildModelGraph(home_cloud_, num_joints_, home_cloud_neighbor_indices, home_cloud_neighbor_dists, neighborhood_range_, max_num_neighbors_);
	cv::flann::Index kd_trees(train_prop_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
	// initialize home feature 
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	// initialize data batch needed including feature and kd trees
	for(int i = 0; i < train_data_size_; i++)
	{
		curr_prop_ = train_prop_.rowRange(i, i + 1);
		SetFeature(feature_batch.colRange(i, i + 1), feature_home_, num_joints_, curr_prop_);
	}
	cv::Mat scale = cv::Mat::zeros(num_joints_, 2, CV_64F);
	for(int i = 0; i < num_joints_; i++)
	{
		scale.at<double>(i, 0) = joint_range_limit_.at<double>(i, 0) - home_prop_.at<double>(0, i);
		scale.at<double>(i, 1) = joint_range_limit_.at<double>(i, 1) - home_prop_.at<double>(0, i);
	}

	

	// always start from home pose
	/********************* just for display ************************/
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	
	/********************* just for display ************************/

	for(unsigned long iteration_count = 0; iteration_count < train_iteration_; iteration_count++)
	{		
		// initialize avg_min_dists
		for(int i = 0; i < num_joints_; i++)
		{
			avg_min_dists[i] = cv::Mat::zeros(home_cloud_.rows, 1, CV_32F);
		}
		int batch_size = 20;
		cv::Mat aim_indices_batch;
		GenerateAimIndexBatch(engine, kd_trees, aim_indices_batch, batch_size, iteration_count, scale);
		for(int i = 0; i < batch_size; i++)
		{
			int cloud_idx = aim_indices_batch.at<int>(i, 0);
			transform_.CalcTransformation(feature_batch.colRange(cloud_idx, cloud_idx + 1));
			transform_.TransformCloud(home_cloud_, transform_.get_transform(), predicted_cloud_batch[cloud_idx]); // need to investigate home cloud issue
			cloud_batch[cloud_idx].convertTo(cloud_f, CV_32F);
			cv::flann::Index curr_cloud_kd_tree(cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
			// cloud_batch[cloud_idx].convertTo(cloud_f, CV_32F);
			// cv::flann::Index test_kd_tree(cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
			// need to accumulate the distances...
			for(int joint_idx = 0; joint_idx < num_joints_; joint_idx++)
			{       
				// need to do initialize...
				indices_batch[cloud_idx][joint_idx] = cv::Mat::zeros(predicted_cloud_batch[cloud_idx][joint_idx].rows, 1, CV_32S);
				min_dists_batch[cloud_idx][joint_idx] = cv::Mat::zeros(predicted_cloud_batch[cloud_idx][joint_idx].rows, 1, CV_32F);
				// the value range of indices is the target cloud, e.g. max(indices) = size(cloud_batch[prop_idx]), but the size of indices is the size of home cloud, e.g. size(home_cloud)
				predicted_cloud_batch[cloud_idx][joint_idx].convertTo(predicted_cloud_f, CV_32F); 		       
				curr_cloud_kd_tree.knnSearch(predicted_cloud_f, indices_batch[cloud_idx][joint_idx], min_dists_batch[cloud_idx][joint_idx], 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
				avg_min_dists[joint_idx] = avg_min_dists[joint_idx] + min_dists_batch[cloud_idx][joint_idx];
			}
		}
		// at this stage, knn search ended, we should now aggregate the results...
		for(int joint_idx = 0; joint_idx < num_joints_; joint_idx++)
		{
			avg_min_dists[joint_idx] = avg_min_dists[joint_idx] / train_data_size_; // train_data_size_;
		}
		// get the labels...
		if(iteration_count == 0)
			InitializeModelLabel(avg_min_dists, num_joints_, home_cloud_label);
		else
			IteratedConditionalModes(home_cloud_neighbor_indices, avg_min_dists, home_cloud_label, potential, num_joints_, icm_iteration_, max_num_neighbors_, icm_beta_, icm_sigma_); // update label, only one iteration first...
		// segment
		for(int i = 0; i < batch_size; i++)
		{
			int cloud_idx = aim_indices_batch.at<int>(i, 0);
			Segment(segmented_target_cloud[cloud_idx], segmented_home_cloud[cloud_idx], segmented_prediction_cloud[cloud_idx], home_cloud_label, cloud_batch[cloud_idx], home_cloud_, predicted_cloud_batch[cloud_idx], indices_batch[cloud_idx], num_joints_);
		}
		transform_.CalculateGradientBatch(segmented_target_cloud, segmented_prediction_cloud, segmented_home_cloud, feature_batch); // target, prediction, query, without label...
		transform_.Update();
		// record data
		RecordData(loader, trend_array, 0, iteration_count, record_trend_interval, record_diagnosis_interval);

		/******************** just for display ******************/
		if(iteration_count % 10 == 0)
		{
			for(int i = 0; i < num_joints_; i++)
			{
				COLOUR c = GetColour(i, 0, num_joints_ - 1);
				pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_home_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>); 
				for(int j = 0; j < batch_size; j++)
				{
					int cloud_idx = aim_indices_batch.at<int>(j, 0);
					if(segmented_home_cloud[cloud_idx][i].rows != 0)
					{
						std::cout << "size: " << segmented_home_cloud[cloud_idx][i].rows << " joint: " << i << " frame: " << cloud_idx << std::endl;
						Mat2PCD(segmented_home_cloud[cloud_idx][i], segmented_home_cloud_pcd);
						break;
					}
				}
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> segmented_home_cloud_color(segmented_home_cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));
				char cloud_name[20];
				sprintf(cloud_name, "cloud_segment_%d", i);
				if(iteration_count == 0)
				{			
					viewer->addPointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);	
				}
				else
				{
					viewer->updatePointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);
				}
			}
			viewer->spinOnce(10.0); // ms as unit
		}
		/******************** just for display ******************/
		ShowLearningProgress(iteration_count);
	}	
}



void Explorer::Train()
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0; // sorted idx 
	int aim_frame_idx = 0; // actual frame idx pointed to
	int record_trend_interval = 1000;
	int record_diagnosis_interval = 1000;
	int query_cloud_size = 0;
	cv::Mat predicted_cloud_f; // matched_target_cloud, transformed_query_cloud, indices, min_dists;
	cv::Mat train_prop_f;
	std::mt19937 engine(rd_());		
	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));
	std::vector<cv::Mat> indices(num_joints_);
    std::vector<cv::Mat> min_dists(num_joints_);    
	std::vector<cv::Mat> segmented_target_cloud(num_joints_);
	std::vector<cv::Mat> segmented_home_cloud(num_joints_);
	std::vector<cv::Mat> segmented_prediction_cloud(num_joints_);
	cv::Mat cloud_f;
	std::vector<int> path(0);

	Loader loader(num_weights_, num_joints_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	cv::Mat home_cloud_label = cv::Mat::zeros(home_cloud_.rows, num_joints_, CV_64F);
	cv::Mat potential = cv::Mat::zeros(home_cloud_.rows, num_joints_, CV_64F);
	// some parameters need to be externalized
	cv::Mat home_cloud_neighbor_indices, home_cloud_neighbor_dists;
	BuildModelGraph(home_cloud_, num_joints_, home_cloud_neighbor_indices, home_cloud_neighbor_dists, neighborhood_range_, max_num_neighbors_);
	train_prop_.convertTo(train_prop_f, CV_32F);
	cv::flann::Index kd_trees(train_prop_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
	
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	home_prop_.copyTo(prev_explore_path_target_);
	// need to put this in configuration
	cv::Mat scale = cv::Mat::zeros(num_joints_, 2, CV_64F);
	for(int i = 0; i < num_joints_; i++)
	{
		scale.at<double>(i, 0) = joint_range_limit_.at<double>(i, 0) - home_prop_.at<double>(0, i);
		scale.at<double>(i, 1) = joint_range_limit_.at<double>(i, 1) - home_prop_.at<double>(0, i);
	}
	// main loop
	// always start from home pose

	/********************* just for display ************************/
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	
	/********************* just for display ************************/

	for(unsigned long iteration_count = 0; iteration_count < train_iteration_; iteration_count++)
	{		
		// feature
		aim_idx = iteration_count == 0 ? 0 : GenerateAimIndex(engine, kd_trees, path, iteration_count, scale);
		curr_prop_ = train_prop_.rowRange(aim_idx, aim_idx + 1);
		SetFeature(feature_, feature_home_, num_joints_, curr_prop_);
		// load cloud
		aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
		loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
		// calc transformation and transform cloud
		transform_.CalcTransformation(feature_);
		transform_.TransformCloud(home_cloud_, transform_.get_transform(), predicted_cloud_); // need to investigate home cloud issue
		if(iteration_count != 0)
		{
			// convert to 32 bit and build kd tree
			cloud_.convertTo(cloud_f, CV_32F);
			cv::flann::Index target_cloud_kd_trees(cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
			for(int i = 0; i < num_joints_; i++)
			{       
				indices[i] = cv::Mat::zeros(predicted_cloud_[i].rows, 1, CV_32S);
				min_dists[i] = cv::Mat::zeros(predicted_cloud_[i].rows, 1, CV_32F);
				predicted_cloud_[i].convertTo(predicted_cloud_f, CV_32F); 		       
				target_cloud_kd_trees.knnSearch(predicted_cloud_f, indices[i], min_dists[i], 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
			}
			if(iteration_count == 1)
				InitializeModelLabel(min_dists, num_joints_, home_cloud_label);
			else
				IteratedConditionalModes(home_cloud_neighbor_indices, min_dists, home_cloud_label, potential, num_joints_, icm_iteration_, max_num_neighbors_, icm_beta_, icm_sigma_); // update label, only one iteration first...
			// shuffle points according to label
			// Segment(matched_target_cloud, home_cloud_label, cloud_, indices, num_joints_); // output memory allocated inside function
			Segment(segmented_target_cloud, segmented_home_cloud, segmented_prediction_cloud, home_cloud_label, cloud_, home_cloud_, predicted_cloud_, indices, num_joints_);
			// update weights, matched target cloud should be a vector...
			// transform_.CalculateGradient(segmented_target_cloud, predicted_cloud_, home_cloud_, feature_); // target, prediction, query, without label...
			transform_.CalculateGradient(segmented_target_cloud, segmented_prediction_cloud, segmented_home_cloud, feature_); // target, prediction, query, without label...
			transform_.Update();
			// record data
			RecordData(loader, trend_array, aim_idx, iteration_count, record_trend_interval, record_diagnosis_interval);

			/******************** just for display ******************/
			if(iteration_count % 200 == 1)
			{
				
				for(int i = 0; i < num_joints_; i++)
				{
					COLOUR c = GetColour(i, 0, num_joints_ - 1);
					pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_home_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>); 
					Mat2PCD(segmented_home_cloud[i], segmented_home_cloud_pcd);
					pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> segmented_home_cloud_color(segmented_home_cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));
					char cloud_name[20];
					sprintf(cloud_name, "cloud_segment_%d", i);
					if(iteration_count == 1)
					{			
						viewer->addPointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);	
					}
					else
					{
						viewer->updatePointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);
					}
				}
				viewer->spinOnce(10.0); // ms as unit
			}
			/******************** just for display ******************/
		}
		ShowLearningProgress(iteration_count);
	}	
}


void Explorer::Test(bool single_frame, bool display, int test_idx)
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	unsigned long iteration_count = 0;
	cv::Mat target_cloud, matched_target_cloud, indices, min_dists;
	std::mt19937 engine(rd_());		
	int start_idx = 0;
	int end_idx = train_data_size_;
	if(single_frame)
	{
		start_idx = test_idx;
		end_idx = test_idx + 1;
	}
	Loader loader(num_weights_, num_joints_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	loader.LoadWeightsForTest(transform_);
	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(display)
	{		
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
		viewer->setBackgroundColor(0, 0, 0);	
		viewer->initCameraParameters();	
	}
	for(aim_idx = start_idx; aim_idx < end_idx; aim_idx++)
	{
		aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
		loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
		curr_prop_ = train_prop_.rowRange(aim_idx, aim_idx + 1);
		SetFeature(feature_, feature_home_, num_joints_, curr_prop_);
		transform_.CalcTransformation(feature_);
		transform_.TransformCloud(home_cloud_, transform_.get_transform(), predicted_cloud_); // need to investigate home cloud issue
		if(display)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr predicted_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			// currently only work for one joint...
			Mat2PCD(cloud_, target_cloud_pcd);
			Mat2PCD(predicted_cloud_[0], predicted_cloud_pcd);
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target_cloud_pcd, 0, 0, 255);			
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> predicted_cloud_color(predicted_cloud_pcd, 0, 255, 0);
			if(aim_idx - start_idx == 0)
			{			
				viewer->addPointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");			
				viewer->addPointCloud<pcl::PointXYZ>(predicted_cloud_pcd, predicted_cloud_color, "transformed_cloud");									
			}
			else
			{
				viewer->updatePointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");
				viewer->updatePointCloud<pcl::PointXYZ>(predicted_cloud_pcd, predicted_cloud_color, "transformed_cloud");
			}
			if(end_idx - start_idx == 1)
				viewer->spin();
			else
				viewer->spinOnce(50);
		}
	}
}

// build the neighbor graph of the home point cloud by radius search...
void Explorer::BuildModelGraph(const cv::Mat& home_cloud, int num_joints, cv::Mat& home_cloud_neighbor_indices, cv::Mat& home_cloud_neighbor_dists, double neighborhood_range, int max_num_neighbor)
{
	cv::Mat home_cloud_f; 
	home_cloud.convertTo(home_cloud_f, CV_32F);
	// initialize home cloud related matrices
	home_cloud_neighbor_indices = cv::Mat::zeros(home_cloud.rows, max_num_neighbor, CV_32S) - 1; // all initialized to -1, which is the marker of non-used cells...
	home_cloud_neighbor_dists = cv::Mat::zeros(home_cloud.rows, max_num_neighbor, CV_32F) - 1; // all initialized to -1, which is the marker
	cv::flann::Index home_cloud_kd_trees(home_cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
	for(int i = 0; i < home_cloud.rows; i++)
	{
		// the radius search thing only works for one row at a time...
		cv::Mat query_point = home_cloud_f.rowRange(i, i + 1);
		cv::Mat curr_neighbor_indices = cv::Mat::zeros(1, max_num_neighbor + 1, CV_32S) - 1;
		cv::Mat curr_neighbor_dists = cv::Mat::zeros(1, max_num_neighbor + 1, CV_32F) - 1;
		home_cloud_kd_trees.radiusSearch(query_point, curr_neighbor_indices, curr_neighbor_dists, neighborhood_range * neighborhood_range, max_num_neighbor + 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud, need to exclude self in the end
		curr_neighbor_indices.colRange(1, max_num_neighbor + 1).copyTo(home_cloud_neighbor_indices.rowRange(i, i + 1));
		curr_neighbor_dists.colRange(1, max_num_neighbor + 1).copyTo(home_cloud_neighbor_dists.rowRange(i, i + 1));
	}
	/*char output_dir[40];
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/home_cloud_indices.bin");
	FileIO::WriteMatInt(home_cloud_neighbor_indices, home_cloud.rows, max_num_neighbor, output_dir);
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/home_cloud_min_dists.bin");
	FileIO::WriteMatFloat(home_cloud_neighbor_dists, home_cloud.rows, max_num_neighbor, output_dir);*/
}

// todo: model class, refactoring later
void Explorer::InitializeModelLabel(const std::vector<cv::Mat>& min_dists, int num_joints, cv::Mat& home_cloud_label)
{
	// home_cloud_neighbor need to be initialized...
	int home_cloud_size = home_cloud_label.rows;
	double min_value = 0;
	double max_value = 0;
	// update according to maximum likelihood only, without neighborhood constraint...
	cv::Mat curr_point_min_dists = cv::Mat::zeros(num_joints, 1, CV_64F);
	cv::Point min_location, max_location;
	min_location.x = 0; min_location.y = 0; max_location.x = 0; max_location.y = 0;
	for(int i = 0; i < home_cloud_size; i++)
    {
		for(int j = 0; j < num_joints; j++)
		{
			curr_point_min_dists.at<double>(j, 0) = min_dists[j].at<float>(i, 0);
		}
		cv::minMaxLoc(curr_point_min_dists, &min_value, &max_value, &min_location, &max_location);
		home_cloud_label.at<double>(i, min_location.y) = 1;
	}
}

// iterated conditional convergence algorithm, beta: the weight of neighborhood, sigam: the standard deviation of error gaussian...
void Explorer::IteratedConditionalModes(const cv::Mat& home_cloud_neighbor_indices, const std::vector<cv::Mat>& min_dists, cv::Mat& home_cloud_label, cv::Mat& potential, int num_joints, int icm_iterations, int max_num_neighbors, double beta, double sigma) 
{
	int cloud_size = home_cloud_label.rows;
	double min_value = 0;
	double max_value = 0;
	cv::Point min_location, max_location;
	cv::Mat tmp_mat_1, tmp_mat_2, tmp_zeros;
	cv::Mat dist_likelihood = cv::Mat::zeros(1, num_joints, CV_64F);
	min_location.x = 0; min_location.y = 0; max_location.x = 0; max_location.y = 0;
	cv::Mat new_home_cloud_label; 
	for(int iter_idx = 0; iter_idx < icm_iterations; iter_idx++)
	{
		new_home_cloud_label = cv::Mat::zeros(home_cloud_label.rows, home_cloud_label.cols, CV_64F);
		for(int point_idx = 0; point_idx < cloud_size; point_idx++)
		{
			cv::Mat label_count = cv::Mat::zeros(1, num_joints, CV_64F);
			// exp(beta * sum_label(l)) / sum(exp(beta * sum_label(k)))
			for(int neighbor_idx = 0; neighbor_idx < max_num_neighbors; neighbor_idx++)
			{
				int neighbor_point_idx = home_cloud_neighbor_indices.at<int>(point_idx, neighbor_idx);
				if(neighbor_point_idx != -1)
				{
					label_count = label_count + home_cloud_label.rowRange(neighbor_point_idx, neighbor_point_idx + 1); 
				}
				else
				{
					break;
				}
			}
			label_count = label_count * beta;
			cv::exp(label_count, tmp_mat_1);
			cv::reduce(tmp_mat_1, tmp_mat_2, 1, CV_REDUCE_SUM); // row-wise reduce
			// neighborhood potential
			potential.rowRange(point_idx, point_idx + 1) = tmp_mat_1 / cv::repeat(tmp_mat_2, label_count.rows, label_count.cols);
			// error potential
			for(int joint_idx = 0; joint_idx < num_joints; joint_idx++)
			{
				// gaussian... zero mean, min dist is already squared...
				// dist_likelihood.at<double>(0, joint_idx) = (1 / (sigma * sqrt(2 * PI))) * exp(-0.5 * min_dists[joint_idx].at<float>(point_idx, 0) / (sigma * sigma));
				dist_likelihood.at<double>(0, joint_idx) = exp(-0.5 * min_dists[joint_idx].at<float>(point_idx, 0) / (sigma * sigma));
			}
			// multiply together...
			potential.rowRange(point_idx, point_idx + 1) = potential.rowRange(point_idx, point_idx + 1).mul(dist_likelihood);
			// dist_likelihood.copyTo(potential.rowRange(point_idx, point_idx + 1)); //  = potential.rowRange(point_idx, point_idx + 1).mul(dist_likelihood);
			cv::minMaxLoc(potential.rowRange(point_idx, point_idx + 1), &min_value, &max_value, &min_location, &max_location);
			tmp_zeros = cv::Mat::zeros(1, num_joints, CV_64F);
			// tmp_zeros.copyTo(home_cloud_label.rowRange(point_idx, point_idx + 1));
			// home_cloud_label.at<double>(point_idx, max_location.x) = 1;
			tmp_zeros.copyTo(new_home_cloud_label.rowRange(point_idx, point_idx + 1));
			new_home_cloud_label.at<double>(point_idx, max_location.x) = 1;
		}
		new_home_cloud_label.copyTo(home_cloud_label);
	}
}

void Explorer::LoadCloudInBatch(Loader& loader, int data_size, std::vector<cv::Mat>& cloud_batch)
{
	cloud_batch = std::vector<cv::Mat>(data_size);
	std::cout << "start to load cloud data, " << data_size << " cloud to load..." << std::endl;
	for(int i = 0; i < data_size; i++)
	{
		loader.LoadBinaryPointCloud(cloud_batch[i], i);
		if(i % 500 == 1)
		{
			std::cout << i << " cloud loaded..." << std::endl;
		}
	}
}

void Explorer::Segment(std::vector<cv::Mat>& segmented_target_cloud, 
					   std::vector<cv::Mat>& segmented_home_cloud, 
					   std::vector<cv::Mat>& segmented_prediction_cloud, 
					   const cv::Mat& home_cloud_label, 
					   const cv::Mat& target_cloud, 
					   const cv::Mat& home_cloud, 
					   const std::vector<cv::Mat>& prediction_cloud, 
					   const std::vector<cv::Mat>& indices, 
					   int num_joints)
{
	// shuffle the cloud to make it match with the template
	double min_value = 0;
	double max_value = 0;
	cv::Point min_location, max_location;
	cv::Mat count = cv::Mat::zeros(num_joints, 1, CV_32S);
	for(int i = 0; i < num_joints; i++)
	{
		segmented_target_cloud[i] = cv::Mat::zeros(indices[0].rows, target_cloud.cols, CV_64F);
		segmented_home_cloud[i] = cv::Mat::zeros(indices[0].rows, target_cloud.cols, CV_64F);
		segmented_prediction_cloud[i] = cv::Mat::zeros(indices[0].rows, target_cloud.cols, CV_64F);
	}
	for(int p = 0; p < indices[0].rows; p++)
	{
		
		cv::minMaxLoc(home_cloud_label.rowRange(p, p + 1), &min_value, &max_value, &min_location, &max_location);
		int label_idx = max_location.x;
		int curr_idx = indices[label_idx].at<int>(p, 0); // so the value range of index is the target cloud, but the size of index is the home cloud
		int curr_count = count.at<int>(label_idx, 0);
		target_cloud.rowRange(curr_idx, curr_idx + 1).copyTo(segmented_target_cloud[label_idx].rowRange(curr_count, curr_count + 1));
		prediction_cloud[label_idx].rowRange(p, p + 1).copyTo(segmented_prediction_cloud[label_idx].rowRange(curr_count, curr_count + 1));
		home_cloud.rowRange(p, p + 1).copyTo(segmented_home_cloud[label_idx].rowRange(curr_count, curr_count + 1));	
		count.at<int>(label_idx, 0) = count.at<int>(label_idx, 0) + 1;
	}
	for(int i = 0; i < num_joints; i++)
	{
		segmented_target_cloud[i] = segmented_target_cloud[i].rowRange(0, count.at<int>(i, 0));
		segmented_home_cloud[i] = segmented_home_cloud[i].rowRange(0, count.at<int>(i, 0));
		segmented_prediction_cloud[i] = segmented_prediction_cloud[i].rowRange(0, count.at<int>(i, 0));
	}
}

void Explorer::ShowLearningProgress(int iteration_count)
{
	if(iteration_count % 20 == 0)			
	{
		std::cout << "iteration: " << iteration_count << std::endl;			
	}
}

void Explorer::RecordData(Loader& loader, std::vector<std::vector<double>>& trend_array, int aim_idx, int iteration_count, int record_trend_interval, int record_diagnosis_interval)
{
	int trend_count = 0;
	for(int i = 0; i < num_joints_; i++)
	{
		for(int j = 0; j < num_weights_; j++)
		{
			trend_count = i * num_weights_ + j;
			trend_array[trend_count].push_back(cv::norm(transform_.get_w(i).rowRange(j, j + 1), cv::NORM_L2));
		}
	}
	trend_array[num_joints_ * num_weights_].push_back(aim_idx);
	// record trend
	if(iteration_count % record_trend_interval == 1)
	{			
		int append_flag = iteration_count == 1 ? 0 : 1;			
		loader.SaveTrend(trend_array, num_trend_, append_flag);								
		for(int i = 0; i < num_trend_; i++)
			trend_array[i].clear();
		// record testing weight
		loader.SaveWeightsForTest(transform_);
	}
	// record diagnosis
	if(iteration_count % record_diagnosis_interval == 1)
	{
		loader.SaveWeightsForDiagnosis(transform_, iteration_count / record_diagnosis_interval);	
	}
}

void Explorer::SetFeature(cv::Mat& feature, cv::Mat& feature_home, int num_joints, const cv::Mat& curr_prop)
{
	int sinusoidal_dim = 3;
	int feature_dim = feature.rows;
	cv::Mat count = cv::Mat::zeros(num_joints, 1, CV_64F);
	cv::Mat curr_prop_sinusoidal = cv::Mat::zeros(num_joints, sinusoidal_dim, CV_64F);
	// set the sinusoidal value
	for(int i = 0; i < num_joints; i++)
	{
		curr_prop_sinusoidal.at<double>(i, 0) = 1;
		curr_prop_sinusoidal.at<double>(i, 1) = sin(curr_prop.at<double>(0, i) / 180.0 * PI);
		curr_prop_sinusoidal.at<double>(i, 2) = cos(curr_prop.at<double>(0, i) / 180.0 * PI);
	}
	
	for(int idx = 0; idx <= feature_dim; idx++)
	{
		if(idx != 0)
		{
			feature.at<double>(idx - 1, 0) = 1;
			int factor = sinusoidal_dim;
			for(int joint_idx = num_joints - 1; joint_idx >= 0; joint_idx--)
			{
				if(joint_idx == num_joints - 1)
				{
					count.at<double>(joint_idx, 0) = idx % factor;	
				}
				else
				{
					count.at<double>(joint_idx, 0) = idx / factor % sinusoidal_dim;	
					factor *= sinusoidal_dim;
				}
				feature.at<double>(idx - 1, 0) *= curr_prop_sinusoidal.at<double>(joint_idx, count.at<double>(joint_idx, 0));	
			}
		}
		
	}
	feature = feature - feature_home;
}

int Explorer::GenerateAimIndex(std::mt19937& engine, cv::flann::Index& kd_trees, std::vector<int>& path, int iteration_count, const cv::Mat& scale)
{
	int aim_idx = 0;
	double current_range = 0;
	double max_speed = 4.0; // 0.6 * scale.at<double>(0, 0); // 0.4 * scale
	double path_length = 0;
	int num_frame_path = 0;
	if(path.size() == 0)
	{		
        // planar exploration range, starting from the center, range is 0 to 1
		current_range = ini_exploration_range_ + (max_exploration_range_ - ini_exploration_range_) * iteration_count / expand_iteration_;	
		current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
		for(int i = 0; i < num_joints_; i++)
		{
			std::uniform_real_distribution<double> uniform(scale.at<double>(i, 0) * current_range + home_prop_.at<double>(0, i), scale.at<double>(i, 1) * current_range + home_prop_.at<double>(0, i));
			explore_path_target_.at<double>(0, i) = uniform(engine); // row vector
		}
		path_length = cv::norm(explore_path_target_ - prev_explore_path_target_, cv::NORM_L2);
		num_frame_path = (int)(path_length / max_speed) + 1;
		path.clear();
		for(int i = 1; i <= num_frame_path; i++)
		{
			cv::Mat tmp_target = cv::Mat::zeros(1, num_joints_, CV_64F);
			tmp_target = prev_explore_path_target_ + (explore_path_target_ - prev_explore_path_target_) * i / num_frame_path;
			tmp_target.convertTo(tmp_target, CV_32F);
			kd_trees.knnSearch(tmp_target, explore_path_kdtree_indices_, explore_path_kdtree_dists_, 1, cv::flann::SearchParams(64));
			path.push_back(explore_path_kdtree_indices_.at<int>(0, 0));			
		}	
		explore_path_target_.copyTo(prev_explore_path_target_);
	}		
	aim_idx = path[0];
	path.erase(path.begin());		
	return aim_idx;
}

void Explorer::GenerateAimIndexBatch(std::mt19937& engine, cv::flann::Index& kd_trees, cv::Mat& aim_indices_batch, int batch_size, int iteration_count, const cv::Mat& scale)
{
	double current_range = 0;
	current_range = ini_exploration_range_ + (max_exploration_range_ - ini_exploration_range_) * iteration_count / expand_iteration_;	
	current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
	cv::Mat target_locations = cv::Mat::zeros(batch_size, num_joints_, CV_64F);
	aim_indices_batch = cv::Mat::zeros(batch_size, 1, CV_32S);
	for(int i = 0; i < num_joints_; i++)
	{
		std::uniform_real_distribution<double> uniform(scale.at<double>(i, 0) * current_range + home_prop_.at<double>(0, i), scale.at<double>(i, 1) * current_range + home_prop_.at<double>(0, i));
		for(int j = 0; j < batch_size; j++)
		{
			target_locations.at<double>(j, i) = uniform(engine);
		}
	}
	for(int i = 0; i < batch_size; i++)
	{
		cv::Mat tmp_target = cv::Mat::zeros(1, num_joints_, CV_64F);
		tmp_target = target_locations.rowRange(i, i + 1);
		tmp_target.convertTo(tmp_target, CV_32F);
		kd_trees.knnSearch(tmp_target, explore_path_kdtree_indices_, explore_path_kdtree_dists_, 1, cv::flann::SearchParams(64));
		aim_indices_batch.at<int>(i, 0) = explore_path_kdtree_indices_.at<int>(0, 0);
	}	
}		


//
//void Explorer::ShowTransformationGrid(int num_grid, int weight_idx)
//{
//	char input_dir[400];
//	char output_dir[400];	
//	int aim_idx = 0;	
//	int aim_frame_idx = 0;
//	int num_gradient_iteration = 10;
//	unsigned long iteration_count = 0;
//	double depth_threshold = 0.8;
//	double voxel_grid_size = 0.008;
//	int cost_flag = 0;
//	int write_trend_interval = 2000;
//	int cloud_scale = 1;
//	cv::Mat target_cloud, matched_target_cloud, transformed_cloud, indices, min_dists;
//	cv::Mat cost = cv::Mat::zeros(1, 1, CV_32F);
//	std::mt19937 engine(rd_());		
//	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));
//
//	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
//	loader.FormatWeightsForTestDirectory();
//	loader.FormatTrendDirectory();
//	loader.LoadLearningRates(transform_);			
//	// loader.LoadProprioception(num_train_data_, train_prop_, train_target_idx_, home_prop_);		
//	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_);	
//	loader.LoadWeightsForTest(transform_);
//	cv::Mat size = cv::Mat::zeros(1, 1, CV_64F);
//	
//	int home_frame_idx = train_target_idx_.at<double>(0, 0);
//	loader.LoadBinaryPointCloud(cloud_, home_frame_idx);
//	// cloud_ = cloud_ * cloud_scale_;
//	SetFeature(feature_, feature_home_, aim_idx, train_prop_);
//	// transform_.CalcTransformInv(feature_);
//	// transform_.TransformDataInv(cloud_, home_cloud_, 1);		
//
//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//	
//	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
//	viewer->setBackgroundColor(0, 0, 0);	
//	viewer->initCameraParameters();	
//	
//	for(int i = 0; i < num_grid; i++)
//	{
//		for(int j = 0; j < num_grid; j++)
//		{
//			cv::Mat curr_prop = cv::Mat::zeros(1, num_joints_, CV_64F);
//			curr_prop.at<double>(0, 0) = -1.0 + (0.5 + i) * 2.0 / num_grid;
//			curr_prop.at<double>(0, 1) = -1.0 + (0.5 + j) * 2.0 / num_grid;
//			prop_diff_ = train_prop_ - repeat(curr_prop, num_train_data_, 1);
//			prop_diff_ = prop_diff_.mul(prop_diff_);
//			reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
//			sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
//			aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);
//
//			aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
//			loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
//			// cloud_ = cloud_ * cloud_scale_;
//			// cloud_.colRange(0, dim_transform_ - 1) = cloud_.colRange(0, dim_transform_ - 1) * cloud_scale_;
//			SetFeature(feature_, feature_home_, aim_idx, train_prop_);
//			// transform_.CalcTransformInv(feature_);
//			// transform_.TransformDataInv(cloud_, home_cloud_, 1);				
//			// transform_.TransformData(home_cloud_, transformed_cloud, 1);
//			
//			pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);		
//			transformed_cloud = cv::Mat::zeros(home_cloud_[weight_idx].rows, dim_transform_, CV_64F);
//			home_cloud_[weight_idx].copyTo(transformed_cloud);
//			// transformed_cloud.colRange(0, dim_transform_ - 1) = home_cloud_[weight_idx].colRange(0, dim_transform_ - 1) / cloud_scale_;			
//			Mat2PCD(transformed_cloud, transformed_cloud_pcd);						
//			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);
//
//			char cloud_name[10];
//			sprintf(cloud_name, "%d_%d", i, j);			
//			viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, cloud_name);												
//		}
//	}
//	viewer->spin();
//
//}
//
//void Explorer::RecordingTrend(Transform& transform, Loader& loader, std::vector<std::vector<double>>& trend_array, int iter, int write_trend_interval, int aim_idx)
//{
//	// hack for the two joint case...
//	cv::Mat w_0 = transform_.w(0);
//	cv::Mat w_1 = transform_.w(1);
//	cv::Mat fisher_inv = transform_.fisher_inv(0);
//	cv::Mat natural_grad = transform_.natural_w_grad(0);
//	cv::Mat grad = transform_.w_grad(0);
//	int trend_number = 15; // 26;
//	trend_array[0].push_back(cv::norm(w_0.rowRange(0, 1), cv::NORM_L2)); trend_array[1].push_back(cv::norm(w_0.rowRange(1, 2), cv::NORM_L2));
//	trend_array[2].push_back(cv::norm(w_0.rowRange(2, 3), cv::NORM_L2)); trend_array[3].push_back(cv::norm(w_0.rowRange(3, 4), cv::NORM_L2));      
//	trend_array[4].push_back(cv::norm(w_0.rowRange(4, 5), cv::NORM_L2)); trend_array[5].push_back(cv::norm(w_0.rowRange(5, 6), cv::NORM_L2));		
//	//trend_array[6].push_back(cv::norm(w_0.rowRange(6, 7), cv::NORM_L2)); trend_array[7].push_back(cv::norm(w_0.rowRange(7, 8), cv::NORM_L2));	
//	//trend_array[8].push_back(cv::norm(w_0.rowRange(8, 9), cv::NORM_L2)); trend_array[9].push_back(cv::norm(w_0.rowRange(9, 10), cv::NORM_L2));
//	//trend_array[10].push_back(cv::norm(w_0.rowRange(10, 11), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w_0.rowRange(11, 12), cv::NORM_L2));
//
//	trend_array[6].push_back(cv::norm(w_1.rowRange(0, 1), cv::NORM_L2)); trend_array[7].push_back(cv::norm(w_1.rowRange(1, 2), cv::NORM_L2));
//	trend_array[8].push_back(cv::norm(w_1.rowRange(2, 3), cv::NORM_L2)); trend_array[9].push_back(cv::norm(w_1.rowRange(3, 4), cv::NORM_L2));      
//	trend_array[10].push_back(cv::norm(w_1.rowRange(4, 5), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w_1.rowRange(5, 6), cv::NORM_L2));		
//	/*trend_array[18].push_back(cv::norm(w_1.rowRange(6, 7), cv::NORM_L2)); trend_array[19].push_back(cv::norm(w_1.rowRange(7, 8), cv::NORM_L2));	
//	trend_array[20].push_back(cv::norm(w_1.rowRange(8, 9), cv::NORM_L2)); trend_array[21].push_back(cv::norm(w_1.rowRange(9, 10), cv::NORM_L2));
//	trend_array[22].push_back(cv::norm(w_1.rowRange(10, 11), cv::NORM_L2)); trend_array[23].push_back(cv::norm(w_1.rowRange(11, 12), cv::NORM_L2));*/
//
//	// trend_array[12].push_back(cv::norm(fisher_inv, cv::NORM_L2)); trend_array[13].push_back(cv::norm(natural_grad, cv::NORM_L2));
//	trend_array[12].push_back(cv::norm(natural_grad, cv::NORM_L2));
//	trend_array[13].push_back(cv::norm(fisher_inv, cv::NORM_L2)); 
//	trend_array[14].push_back(aim_idx);
//	/*trend_array[10].push_back(cv::norm(w.rowRange(10, 11), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w.rowRange(11, 12), cv::NORM_L2));	
//	trend_array[12].push_back(cv::norm(grad, cv::NORM_L2));*/
//	if(iter % write_trend_interval == 0)
//	{			
//		int append_flag = iter == 0 ? 0 : 1;			
//		loader.SaveTrend(trend_array, trend_number, append_flag);								
//		for(int i = 0; i < trend_number; i++)
//			trend_array[i].clear();
//	}
//	loader.SaveWeightsForTest(transform);
//	/*cv::Mat fisher_inv = transform_.fisher_inv();
//	cv::Mat natural_grad = transform_.natural_w_grad();*/
//}
//
//void Explorer::ReOrder(cv::Mat& input, cv::Mat& output, cv::Mat& input_indices)
//{
//	output = cv::Mat::zeros(input_indices.rows, input.cols, CV_64F);
//	for(int p = 0; p < input_indices.rows; p++)
//		for(int q = 0; q < input.cols; q++)
//			output.at<double>(p, q) = input.at<double>(input_indices.at<int>(p, 0), q);				
//}
//
void Explorer::PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat& cloud_mat)
{
	int size = cloud->points.size();
	int dim = 4;
	cloud_mat = cv::Mat::zeros(size, dim, CV_64F);
	for(int i = 0; i < size; i++)
	{
		cloud_mat.at<double>(i, 0) = cloud->points[i].x;
		cloud_mat.at<double>(i, 1) = cloud->points[i].y;
		cloud_mat.at<double>(i, 2) = cloud->points[i].z;
		cloud_mat.at<double>(i, 3) = 1.0;
	}
}
//
void Explorer::Mat2PCD(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	int size = cloud_mat.rows;
	std::vector<pcl::PointXYZ> points_vec(size);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < size; i++)
	{
		pcl::PointXYZ point;
		point.x = cloud_mat.at<double>(i, 0);
		point.y = cloud_mat.at<double>(i, 1);
		point.z = cloud_mat.at<double>(i, 2);
		cloud->push_back(point);
	}	
}
//
//void Explorer::PreprocessingAndSavePointCloud()
//{
//	char input_dir[400];
//	char output_dir[400];		
//	unsigned long iteration_count = 0;
//	double depth_threshold = 0.8;
//	double voxel_grid_size = 0.005; // 0.010;
//	int num_clouds = num_train_data_;
//	std::mt19937 engine(rd_());		
//
//	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
//	loader.FormatWeightsForTestDirectory();
//	loader.FormatTrendDirectory();
//	loader.LoadLearningRates(transform_);
//	// algorithms
//	pcl::PassThrough<pcl::PointXYZ> pass;
//	pcl::PCDReader reader;
//	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;	
//	
//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//	 /*pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//	 pcl::PointCloud<pcl::PointXYZ>::Ptr home_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//	 pcl::PointCloud<pcl::PointXYZ>::Ptr prev_home_cloud(new pcl::PointCloud<pcl::PointXYZ>);*/
//	 pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);	
//	// point clouds		
//	 /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
//	 viewer->setBackgroundColor(0, 0, 0);	
//	 viewer->initCameraParameters();*/	
//	for(iteration_count = 0; iteration_count < 17900; iteration_count++)
//	{					
//		loader.LoadPointCloud(cloud, reader, iteration_count); // load point cloud			
//		DepthFiltering(depth_threshold, pass, cloud, tmp_cloud);
//		DownSamplingPointCloud(voxel_grid_size, voxel_grid, tmp_cloud, cloud);
//		loader.SavePointCloudAsBinaryMat(cloud, iteration_count);
//		if(iteration_count % 100 == 1)
//			std::cout << "iteration: " << iteration_count << std::endl;
//	}
//}
//
//void Explorer::DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud)
//{
//	pass.setInputCloud(cloud);
//	pass.setFilterFieldName ("z");
//	pass.setFilterLimits(0.0, depth);
//	pass.filter(*filtered_cloud);
//}
//
//void Explorer::DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud)
//{
//	voxel_grid.setInputCloud(cloud);
//	voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
//	voxel_grid.filter(*down_sampled_cloud);
//}
//
//void Explorer::ShowCloudSequence()
//{
//	int num_cloud = 4000;
//	Loader loader(12, 3, 13, dir_id_, dir_);
//	pcl::PCDReader reader;
//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
//	viewer->setBackgroundColor(0, 0, 0);	
//	viewer->initCameraParameters();	
//	for(int i = 0; i < num_cloud; i++)
//	{
//		loader.LoadPointCloud(cloud, reader, i);
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, 0, 0, 255);	
//		if(i == 0)
//			viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "original_cloud");	
//		else
//			viewer->updatePointCloud(cloud, cloud_color, "original_cloud");
//		viewer->spinOnce(1);		
//	}
//	viewer->close();
//}
//
//void Explorer::GenerateLinePath(std::vector<std::vector<double>>& path, std::vector<double>& targets, std::vector<double>& prev_targets)
//{		
//	double max_speed = 8 / 20.0;
//	double path_length = 0;
//	for(int i = 0; i < num_joints_; i++)	
//		path_length += sqrt(pow(targets[i] - prev_targets[i], 2));	
//	
//	int num_frame_path = (int)(path_length / max_speed) + 1;
//
//	path = std::vector<std::vector<double>>(num_joints_, std::vector<double>(num_frame_path));
//
//	for(int i = 0; i < num_joints_; i++)	
//		for(int j = 0; j < num_frame_path; j++)		
//			path[i][j] = prev_targets[i] + (targets[i] - prev_targets[i]) * (j + 1) / num_frame_path;
//	
//
//}
//
//int Explorer::GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration)
//{
//	int aim_idx = 0;
//	double current_range = 0;
//	
//	current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
//	current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
//	std::uniform_real_distribution<double> uniform(-1.0 * current_range, 1.0 * current_range);	  	
//	
//	// generate path
//	if(path_count_ == 0)
//	{
//		for(int i = 0; i < num_joints_; i++)
//			targets_[i] = uniform(engine);
//		
//		GenerateLinePath(path_, targets_, prev_targets_);
//		for(int i = 0; i < num_joints_; i++)
//			prev_targets_[i] = targets_[i];
//		
//		path_count_ = path_[0].size();
//	}
//
//	for(int i = 0; i < num_joints_; i++)
//		curr_prop_.at<double>(0, i) = path_[i][path_[0].size() - path_count_];	
//	path_count_--;
//	
//	curr_prop_matrix_ = repeat(curr_prop_, num_train_data_, 1);
//	prop_diff_ = train_prop_ - curr_prop_matrix_;
//	prop_diff_ = prop_diff_.mul(prop_diff_);
//	reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
//	sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
//	aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);
//
//	return aim_idx;
//
//}
// home_cloud_kd_trees.radiusSearch(test_query, home_cloud_indices, home_cloud_min_dists, 0.1, 200, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
	// just for debugging
	/*char output_dir[40];
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/test/home_cloud_indices.bin");
	FileIO::WriteMatInt(home_cloud_neighbor_indices, home_cloud.rows, max_num_neighbor, output_dir);
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/test/home_cloud_min_dists.bin");
	FileIO::WriteMatFloat(home_cloud_neighbor_dists, home_cloud.rows, max_num_neighbor, output_dir);
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/test/home_cloud_label.bin");
	FileIO::WriteMatDouble(home_cloud_label, home_cloud.rows, num_joints, output_dir);*/