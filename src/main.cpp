#include "../inc/explorer.h"
#include "../inc/transform.h"
#include "../inc/fio.h"
#include <boost/program_options.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <iostream>

int main(int argc, char* argv[])
{
	// parameters...
	char data_set[400];
	char config_dir[100];
	static std::string TRAIN = "train";
	static std::string TEST = "test";
	// json keys
	static std::string DATA_SET = "dataset";
	static std::string NORMAL_LEARNING_RATE = "normal learning rate";
	static std::string TRAIN_ITERATIONS = "train iterations";
	static std::string EXPAND_ITERATIONS = "expand iterations";
	static std::string INI_EXPLORATION_RANGE = "initial exploration range";
	static std::string NUM_JOINTS = "joints number";
	static std::string TRAIN_DATA_SIZE = "train data size";
	static std::string TEST_DATA_SIZE = "test data size";
	static std::string JOINT_IDX = "joint idx";
	static std::string JOINT_RANGE_LIMIT = "joint range limit";
	static std::string RANGE_LOW_LIMIT = "low";
	static std::string RANGE_HIGH_LIMIT = "high";
	static std::string NEIGHBOR_HOOD_RANGE = "neighborhood range";
	static std::string MAX_NUM_NEIGHBORS = "max num neighbors";
	static std::string ICM_ITERATION = "icm iteration";
	static std::string ICM_BETA = "icm beta";
	static std::string ICM_SIGMA = "icm sigma";
	static int DIM_TRANSFORM = 3; // point cloud always 3D transform
	// running options
	int train_iteration, expand_iteration, train_data_size, test_data_size; 
	int num_joints, target_idx, test_idx, id, icm_iteration, max_num_neighbors;
	double normal_learning_rate, ini_exploration_range, neighborhood_range, icm_beta, icm_sigma;
	bool train, test, single_frame, test_display;
	cv::Mat joint_idx, joint_range_limit;
	namespace po = boost::program_options;
	namespace json = rapidjson;
	po::options_description desc("learning options");
	desc.add_options()
		("id,i", po::value<int>(), "directory id")
		("learning-option,l", po::value<std::string>(),"learning options")
		("single-frame,s", po::value<bool>(),"learning options")
		("display,d", po::value<bool>(),"display option")
		("test-index,t", po::value<int>(),"test frame index")
		("gtest_break_on_failure", "google test break on failure")
		("gtest_catch_exceptions", po::value<int>(), "google test catch expcetions")
		("gtest_filter", po::value<std::string>(), "google test filter");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if(vm.count("gtest_break_on_failure")) // if it is not learning related... 
	{
		testing::InitGoogleTest(&argc, argv); 
		RUN_ALL_TESTS(); 
		std::getchar(); // keep console window open until Return keystroke
	}
	else
	{
		if(!vm.count("id") || !vm.count("learning-option")) // if not all required options are there
		{
			std::cout << "incorrect command line argument lists..." << std::endl;
			std::getchar();
		}
		else
		{
			id = vm["id"].as<int>();
			std::string learning_option = vm["learning-option"].as<std::string>();
			
			bool train = TRAIN.compare(learning_option) == 0 ? true : false;
			bool test = TEST.compare(learning_option) == 0 ? true : false;
			
			if(train == false && test == false) // if not correct learning-option
			{
				std::cout << "incorrect learning-option, it should be set to either \"train\" or \"test\"" << std::endl;
				std::getchar();	
			}
			else
			{
				sprintf(config_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/config/config_%d.json", id);
				std::string config = FileIO::ReadFileString(config_dir);
				json::Document d;
				d.Parse<0>(config.c_str());
				sprintf(data_set, d[DATA_SET.c_str()].GetString());
				train_iteration = d[TRAIN_ITERATIONS.c_str()].GetInt();
				expand_iteration = d[EXPAND_ITERATIONS.c_str()].GetInt();
				ini_exploration_range = d[INI_EXPLORATION_RANGE.c_str()].GetDouble();
				num_joints = d[NUM_JOINTS.c_str()].GetInt();
				normal_learning_rate = d[NORMAL_LEARNING_RATE.c_str()].GetDouble();
				train_data_size = d[TRAIN_DATA_SIZE.c_str()].GetInt();
				test_data_size = d[TEST_DATA_SIZE.c_str()].GetInt();
				neighborhood_range = d[NEIGHBOR_HOOD_RANGE.c_str()].GetDouble();
				icm_iteration = d[ICM_ITERATION.c_str()].GetInt();
				icm_beta = d[ICM_BETA.c_str()].GetDouble();
				icm_sigma = d[ICM_SIGMA.c_str()].GetDouble();
				max_num_neighbors = d[MAX_NUM_NEIGHBORS.c_str()].GetInt();
				joint_idx = cv::Mat::zeros(num_joints, 1, CV_64F);
				joint_range_limit = cv::Mat::zeros(num_joints, 2, CV_64F);
				for(int i = 0; i < num_joints; i++)
				{
					joint_idx.at<double>(i, 0) = d[JOINT_IDX.c_str()][i].GetInt();
					joint_range_limit.at<double>(i, 0) = d[JOINT_RANGE_LIMIT.c_str()][i][RANGE_LOW_LIMIT.c_str()].GetDouble();
					joint_range_limit.at<double>(i, 1) = d[JOINT_RANGE_LIMIT.c_str()][i][RANGE_HIGH_LIMIT.c_str()].GetDouble();
				}
				// initialize explorer object
				Explorer explorer(id, data_set, train_iteration, expand_iteration, DIM_TRANSFORM + 1, num_joints, normal_learning_rate, ini_exploration_range, 
					train_data_size, test_data_size, joint_idx, joint_range_limit, neighborhood_range, icm_iteration, icm_beta, icm_sigma, max_num_neighbors);
				if(train == true)
				{
					explorer.Train();
				}
				else if(test == true)
				{
					if(!vm.count("single-frame") || !vm.count("display") || !vm.count("test-index"))
					{
						std::cout << "incorrect testing argument..." << std::endl;
						std::getchar();	
					}
					test_idx = vm["test-index"].as<int>();
					test_display = vm["display"].as<bool>();
					single_frame = vm["single-frame"].as<bool>();
					explorer.Test(single_frame, test_display, test_idx);
				}
				std::getchar();
			}
		}
	}
}	