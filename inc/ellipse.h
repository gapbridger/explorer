#ifndef ELLIPSE_H
#define ELLIPSE_H
#define PI 3.14159265

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "transform.h"
typedef std::vector<cv::Mat> MatL;

class Ellipse
{
private:
	// initial parameters
	double ini_mu_x_;
	double ini_mu_y_;
	double ini_long_axis_;
	double ini_short_axis_;
	double ini_angle_;
	// current parameters
	double mu_x_;
	double mu_y_;
	double long_axis_;
	double short_axis_;
	double angle_;
	// previous parameters
	double prev_mu_x_;
	double prev_mu_y_;
	double prev_long_axis_;
	double prev_short_axis_;
	double prev_angle_;
	// reference parameters... in reference frame
	double ref_mu_x_;
	double ref_mu_y_;
	double ref_long_axis_;
	double ref_short_axis_;
	double ref_angle_;

	double ini_ref_mu_x_;
	double ini_ref_mu_y_;
	double ini_ref_long_axis_;
	double ini_ref_short_axis_;
	double ini_ref_angle_;	

	double radius_;

	/*cv::Mat ref_transform_;
	cv::Mat aim_transform_;*/
	// cv::Mat inv_aim_transform_;
	// current matrix structures
	cv::Mat mu_;
	cv::Mat cov_;
	cv::Mat cov_inv_;
	// previous matrix structures
	cv::Mat prev_mu_;
	cv::Mat prev_cov_;
	cv::Mat prev_cov_inv_;
	// ref matrix
	cv::Mat ref_mu_;
	cv::Mat ref_cov_;
	cv::Mat ref_cov_inv_;
	cv::Mat ref_eigen_value_;
	cv::Mat ref_eigen_vector_;

	cv::Mat eigen_value_;
	cv::Mat eigen_vector_;		

	cv::Mat ref_conic_;
	cv::Mat conic_;

	Transform ref_transform_;

	double eta_;

	// temporary variables
	/*cv::Mat translate_;
	cv::Mat scale_;
	cv::Mat rotate_;
	cv::Mat shear_;*/
	
	
	
public:
	Transform transform_;

	Ellipse(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle, double radius);		
	void UpdateEllipseByAction(const cv::Mat& action);	
	void UpdateRefEllipse(cv::Mat& points, cv::Mat& improvement);
	// void SetTransform(double x, double y, double long_axis, double short_axis, double angle, cv::Mat& transform);
	void SetCovariance(double long_axis, double short_axis, double angle, cv::Mat& cov);
	void SetMu(double mu_x, double mu_y, cv::Mat& mu);
	void CopyToPrev();
	void CalcInvCovar(const cv::Mat& cov, cv::Mat& cov_inv);
	double CalcMahaDist(const cv::Mat& data, const cv::Mat& mu, const cv::Mat& cov_inv);
	double CalcMahaDist(const cv::Mat& data);
	bool CheckInsideEllipse(const cv::Mat& point, const cv::Mat& mu, const cv::Mat& cov_inv);
	void DrawEllipse(cv::Mat& disp_img);
	void DrawEllipse(cv::Mat& disp_img, double radius); // with external passed radius...
	void GetKeyPointInEllipse(const cv::Mat& descriptor, const cv::Mat& key_point, cv::Mat& elips_descriptor, cv::Mat& elips_key_point, int curr_frame_flag);
	void ClassifyPointsForReferenceFrame(MatL match_data, cv::Mat& classified_points, cv::Mat& improvement);
	void SetRefEllipseParameters();
	void SetConic(double x, double y, double long_axis, double short_axis, double angle, cv::Mat& conic);
	cv::Mat ref_mu();
	cv::Mat ref_cov();
	void set_ref_mu(const cv::Mat& mu);
	void set_ref_cov(const cv::Mat& cov);
	void set_eta(double eta);
	
};

#endif
// cv::Mat ref_conic_;
// cv::Mat conic_;
// cv::Mat prev_conic_;
// void SetConic(double x, double y, double long_axis, double short_axis, double angle, cv::Mat& conic);
// void InitializeEllipse();
// initial parameters
// double ini_x_; // initial x position
// double ini_y_; // initial y position
// double ini_lx_; // initial long axis 
// double ini_sx_; // initial short axis
// double ini_ang_; // initial angle