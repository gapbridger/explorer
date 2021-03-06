#ifndef LOADER_H
#define LOADER_H 

#include "../inc/fio.h"
#include "../inc/transform.h"
#include "../inc/ellipse.h"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/highgui/highgui_c.h"

typedef std::vector<double> fL;

class Loader{

private:
	int action_number_;
	int diagnosis_number_;
	int trend_number_;
	int dir_id_;
	// trained weights for testing
	char** test_weights_dir_;
	char** diagnosis_weights_dir_;
	char** trend_dir_;
	char common_data_prefix_[400];
	char common_output_prefix_[400];
	char common_diagnosis_prefix_[400];

	double width_shift_;
	double height_shift_;
	double width_scale_;
	double height_scale_;

	

public:
	// Initialization
	Loader(int action_number, int trend_number, int dir_id, char* dir);	
	void FormatWeightsForTestDirectory();
	void FormatWeightsForDiagnosisDirectory();
	void FormatTrendDirectory();
	void LoadWeightsForTest(Transform& transform, int output_dim, int input_dim);
	void SaveWeightsForTest(Transform& transform, int output_dim, int input_dim);
	void SaveWeightsForDiagnosis(Transform& transform, Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx);
	void LoadWeightsForDiagnosis(Transform& transform, Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx);
	void SaveTrend(fL* trend_array, int trend_number, int append_flag);
	void SaveEllipse(Ellipse& ellipse);
	void SavePropSinusoidalFeature(cv::Mat& train_prop_feature, cv::Mat& test_prop_feature);
	void LoadPropSinusoidalFeature(cv::Mat& train_prop_feature, cv::Mat& test_prop_feature);
	void LoadEllipse(Ellipse& ellipse);
	void LoadImage(int frame_idx, cv::Mat& disp_img);
	void LoadSiftKeyPoint(cv::Mat& descriptors, cv::Mat& key_points, int frame_idx, cv::Mat& ini_transformation);
	void LoadAllSiftKeyPoint(MatL& descriptors, MatL& key_points, int start_idx, int end_idx, cv::Mat& ini_transformation);
	void LoadProprioception(int num_train_data, int num_test_data, cv::Mat& train_prop, cv::Mat& test_prop, cv::Mat& home_prop, cv::Mat& target_idx, cv::Mat& test_target_idx);
	void AppendTestWeightName(char** dir_str_array);	
	void AppendDiagnosisWeightName(char** dir_str_array);
	void AppendTrendName(char** dir_str_array);	

	void RecordSiftKeyPoints();
	void LoadLearningRates(Ellipse& ellipse);

	char** test_weight_dir();
	

};

#endif
