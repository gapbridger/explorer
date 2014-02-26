#include "../inc/ellipse.h"

Ellipse::Ellipse(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle, double radius) :
	ini_mu_x_(initial_x),
	ini_mu_y_(initial_y),
	ini_long_axis_(initial_long_axis),
	ini_short_axis_(initial_short_axis),
	ini_angle_(initial_angle),
	mu_x_(initial_x),
	mu_y_(initial_y),
	long_axis_(initial_long_axis),
	short_axis_(initial_short_axis),
	angle_(initial_angle),
	radius_(radius),
	transform_(Transform(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle)),
	ref_transform_(Transform(0, 0, 1, 1, 0))
{
	mu_ = cv::Mat::zeros(2, 1, CV_64F);
	cov_ = cv::Mat::eye(2, 2, CV_64F);
	cov_inv_ = cv::Mat::eye(2, 2, CV_64F);

	prev_mu_ = cv::Mat::zeros(2, 1, CV_64F);
	prev_cov_ = cv::Mat::eye(2, 2, CV_64F);
	prev_cov_inv_ = cv::Mat::eye(2, 2, CV_64F);

	ref_mu_ = cv::Mat::zeros(2, 1, CV_64F);
	ref_cov_ = cv::Mat::eye(2, 2, CV_64F);
	ref_cov_inv_ = cv::Mat::eye(2, 2, CV_64F);	

	eigen_value_ = cv::Mat::zeros(2, 2, CV_64F);
	eigen_vector_ = cv::Mat::zeros(2, 2, CV_64F);	
	ref_eigen_value_ = cv::Mat::zeros(2, 2, CV_64F);
	ref_eigen_vector_ = cv::Mat::zeros(2, 2, CV_64F);
	ref_conic_ = cv::Mat::zeros(3, 3, CV_64F);
	conic_ = cv::Mat::zeros(3, 3, CV_64F);

	// ref
	ref_mu_x_ = 0;
	ref_mu_y_ = 0;
	ref_long_axis_ = 1.0;
	ref_short_axis_ = 1.0;
	ref_angle_ = 0;	
	// initialize ref conic...
	SetConic(ref_mu_x_, ref_mu_y_, ref_long_axis_, ref_short_axis_, ref_angle_, ref_conic_);

	ini_ref_mu_x_ = 0;
	ini_ref_mu_y_ = 0;
	ini_ref_long_axis_ = 1.0;
	ini_ref_short_axis_ = 1.0;
	ini_ref_angle_ = 0;	
		
	SetMu(mu_x_, mu_y_, mu_);
	SetCovariance(long_axis_, short_axis_, angle_, cov_);
	CalcInvCovar(cov_, cov_inv_);

	eta_ = 0;
}

void Ellipse::UpdateEllipseByAction(const cv::Mat& action)
{	
	// the actions are ordered in the sequence of x, y, angle, long axis and short axis
	
	// transform ellipse in reference frame
	cv::Mat inv_transform = transform_.transform_inv();
	conic_ = inv_transform.t() * ref_conic_ * inv_transform;

	double a = conic_.at<double>(0, 0); double b = conic_.at<double>(0, 1);	
	double c = conic_.at<double>(1, 1); double d = conic_.at<double>(0, 2);	
	double f = conic_.at<double>(1, 2); double g = conic_.at<double>(2, 2);
	
	// miu 
	mu_x_ = (c * d - b * f) / (b * b - a * c); // x
	mu_y_ = (a * f - b * d) / (b * b - a * c); // y
	// long and short axes
	long_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
	short_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (-sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
	// angle
	if(b == 0 && a < c)
		angle_ = 0;
	else if(b == 0 && a >= c)
		angle_ = PI / 2;
	else if(a < c)
		angle_ = 1.0 / 2.0 * atan(2 * b / (a - c));
	else if(a >= c)
		angle_ = PI / 2 + 1.0 / 2.0 * atan(2 * b / (a - c));

	SetMu(mu_x_, mu_y_, mu_);
	SetCovariance(long_axis_, short_axis_, angle_, cov_);
	CalcInvCovar(cov_, cov_inv_);
}

void Ellipse::SetMu(double mu_x, double mu_y, cv::Mat& mu)
{
	mu.at<double>(0, 0) = mu_x;
	mu.at<double>(1, 0) = mu_y;
}

void Ellipse::SetCovariance(double long_axis, double short_axis, double angle, cv::Mat& cov)
{
	eigen_value_.at<double>(0, 0) = pow(long_axis, 2); eigen_value_.at<double>(0, 1) = 0.0;
	eigen_value_.at<double>(1, 1) = pow(short_axis, 2); eigen_value_.at<double>(1, 0) = 0.0;
	eigen_vector_.at<double>(0, 0) = cos(angle); eigen_vector_.at<double>(0, 1) = -sin(angle);
	eigen_vector_.at<double>(1, 0) = sin(angle); eigen_vector_.at<double>(1, 1) = cos(angle);
	// aim covar
	cov = eigen_vector_ * eigen_value_ * eigen_vector_.t();
}

void Ellipse::CalcInvCovar(const cv::Mat& cov, cv::Mat& cov_inv)
{
	cv::invert(cov, cov_inv);	
}

// calculate mahalanobis distance
double Ellipse::CalcMahaDist(const cv::Mat& data, const cv::Mat& mu, const cv::Mat& cov_inv)
{
	// calculate mahalanobis distance  
	cv::Mat dist_mat = (data - mu).t() * cov_inv * (data - mu);  
	return dist_mat.at<double>(0, 0);
}

double Ellipse::CalcMahaDist(const cv::Mat& data)
{
	cv::Mat dist_mat = (data - mu_).t() * cov_inv_ * (data - mu_);  
	return dist_mat.at<double>(0, 0);
}

// judge whether inside an ellipse
bool Ellipse::CheckInsideEllipse(const cv::Mat& point, const cv::Mat& mu, const cv::Mat& cov_inv)
{
	// radius is in the sense of standard deviation
	if(CalcMahaDist(point, mu, cov_inv) <= pow(radius_, 2))
		return true;
	else
		return false;
}

void Ellipse::CopyToPrev()
{
	prev_mu_x_ = mu_x_;
	prev_mu_y_ = mu_y_;
	prev_long_axis_ = long_axis_;
	prev_short_axis_ = short_axis_;
	prev_angle_ = angle_;

	mu_.copyTo(prev_mu_);
	cov_.copyTo(prev_cov_);

	transform_.CopyToPrev();
	ref_transform_.CopyToPrev();
}
  
// draw the ellipses with previous or other ones...
void Ellipse::DrawEllipse(cv::Mat& disp_img)
{
	cv::ellipse(disp_img, cv::Point2f(mu_x_, mu_y_), cv::Size(long_axis_ * radius_, short_axis_ * radius_), angle_ / PI * 180.0, 0.0, 360.0, cv::Scalar(190, 0, 190), 2);
}

void Ellipse::DrawEllipse(cv::Mat& disp_img, double radius)
{
	cv::ellipse(disp_img, cv::Point2f(mu_x_, mu_y_), cv::Size(long_axis_ * radius, short_axis_ * radius), angle_ / PI * 180.0, 0.0, 360.0, cv::Scalar(190, 0, 190), 2);
}

void Ellipse::GetKeyPointInEllipse
	(
	const cv::Mat& descriptor, 
	const cv::Mat& key_point, 
	cv::Mat& elips_descriptor, 
	cv::Mat& elips_key_point, 
    int curr_frame_flag
	)
{
	std::vector<int> idx_list;
	int num_key_point, num_elips_key_point, num_cols; 
	cv::Mat pt = cv::Mat::zeros(2, 1, CV_64F);
	num_key_point = key_point.rows;
	num_cols = descriptor.cols;
	for(int i = 0; i < num_key_point; i++){		
		pt.at<double>(0, 0) = key_point.at<double>(i, 0); // x
		pt.at<double>(1, 0) = key_point.at<double>(i, 1); // y		
		if(curr_frame_flag)
		{
			if(CheckInsideEllipse(pt, mu_, cov_inv_))
				idx_list.push_back(i);
		}
		else
		{
			if(CheckInsideEllipse(pt, prev_mu_, prev_cov_inv_))
				idx_list.push_back(i);
		}
	}
	// allocate memory
	num_elips_key_point = idx_list.size();
	elips_descriptor.create(num_elips_key_point, num_cols, CV_64F);
	elips_key_point.create(num_elips_key_point, 2, CV_64F);
	// assign value
	for(int i = 0; i < num_elips_key_point; i++)
	{
		key_point.rowRange(idx_list[i], idx_list[i] + 1).copyTo(elips_key_point.rowRange(i, i + 1));
		descriptor.rowRange(idx_list[i], idx_list[i] + 1).copyTo(elips_descriptor.rowRange(i, i + 1));
	}
}

void Ellipse::ClassifyPointsForReferenceFrame(MatL match_data, cv::Mat& classified_points, cv::Mat& improvement)
{
	int num_classified_data = 0;
	double classification_dist_threshold =  0.08;
	double ref_dist = 0; // distance in the reference frame...
	double improve_percent = 0;
	
	std::vector<int> idx_list;
	if(match_data[0].cols != 14)
	{
		std::cout << "match data size incorrect..." << std::endl;
		exit(0);
	}
	for(int i = 0; i < match_data.size(); i++)
	{
		ref_dist = sqrt(pow(match_data[i].at<double>(0, 7) - match_data[i].at<double>(0, 9), 2) + pow(match_data[i].at<double>(0, 8) - match_data[i].at<double>(0, 10), 2));
		if(ref_dist < classification_dist_threshold)
			idx_list.push_back(i);
	}
	num_classified_data = idx_list.size();
	classified_points.create(num_classified_data, 2, CV_64F);
	improvement.create(num_classified_data, 1, CV_64F);
	for(int i = 0; i < num_classified_data; i++)
	{
		classified_points.at<double>(i, 0) = match_data[idx_list[i]].at<double>(0, 9); // current ref point x...
		classified_points.at<double>(i, 1) = match_data[idx_list[i]].at<double>(0, 10); // current ref point y...
		if(match_data[idx_list[i]].at<double>(0, 13) >= match_data[idx_list[i]].at<double>(0, 12))
			improve_percent = 0;
		else
			improve_percent = (match_data[idx_list[i]].at<double>(0, 12) - match_data[idx_list[i]].at<double>(0, 13)) / match_data[idx_list[i]].at<double>(0, 12);
		// assign improve percentage...
		improvement.at<double>(i, 0) = improve_percent;
	}		
}

void Ellipse::UpdateRefEllipse(cv::Mat& points, cv::Mat& improvement)
{
	if(points.rows != 0)
	{
		// previous update rule is wrong
		// double eta = 1e-6;		
		cv::Mat tmp_cov = cv::Mat::zeros(2, 2, CV_64F);		
		cv::Mat tmp_mu = cv::Mat::zeros(1, 2, CV_64F);
		cv::Mat improvement_sum;
		cv::reduce(improvement, improvement_sum, 0, CV_REDUCE_SUM);
				
		if(improvement_sum.at<double>(0, 0) != 0)
		{
			improvement = improvement / improvement_sum.at<double>(0, 0);			
			// tmp mu
			for(int i = 0; i < points.rows; i++)
				tmp_mu = tmp_mu + improvement.at<double>(i, 0) * points.rowRange(i, i + 1);									
			// update mu...
			ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
			// tmp covariance
			for(int i = 0; i < points.rows; i++)		
				tmp_cov = tmp_cov + improvement.at<double>(i, 0) * (points.rowRange(i, i + 1).t() - ref_mu_) * (points.rowRange(i, i + 1) - ref_mu_.t());										
			// update ref cov
			ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);
		}		
		SetRefEllipseParameters();
	}

	/********* original update mu *********/
	// cv::reduce(points, tmp_mu, 0, CV_REDUCE_AVG);
	// ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
		
	/********* original update cov *********/
	// update ref mu
	/*for(int i = 0; i < points.rows; i++)
	{
		tmp_cov = (points.rowRange(i, i + 1).t() - ref_mu_) * (points.rowRange(i, i + 1) - ref_mu_.t());
		ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);				
	}*/
}

// need to test...
void Ellipse::SetRefEllipseParameters()
{	
	// for testing...

	cv::eigen(ref_cov_, ref_eigen_value_, ref_eigen_vector_);	
	ref_mu_x_ = ref_mu_.at<double>(0, 0);
	ref_mu_y_ = ref_mu_.at<double>(1, 0);
	ref_long_axis_ = sqrt(ref_eigen_value_.at<double>(0, 0));
	ref_short_axis_ = sqrt(ref_eigen_value_.at<double>(1, 0));
	ref_angle_ = atan2(ref_eigen_vector_.at<double>(0, 1), ref_eigen_vector_.at<double>(0, 0)); // sin cos
	SetConic(ref_mu_x_, ref_mu_y_, ref_long_axis_, ref_short_axis_, ref_angle_, ref_conic_);
}


void Ellipse::SetConic(double x, double y, double long_axis, double short_axis, double angle, cv::Mat& conic)
{

	double a = pow(cos(angle), 2) / pow(long_axis, 2) + pow(sin(angle), 2) / pow(short_axis, 2);
	double b = 2 * cos(angle) * sin(angle) * (1 / pow(long_axis, 2) - 1 / pow(short_axis, 2));
	double c = pow(sin(angle), 2) / pow(long_axis, 2) + pow(cos(angle), 2) / pow(short_axis, 2);
	double d = -(2 * a * x + b * y);
	double e = -(2 * c * y + b * x);
	double f = a * pow(x, 2) + b * x * y + c * pow(y, 2) - 1;
	// assignment...
	conic.at<double>(0, 0) = a;         conic.at<double>(0, 1) = b / 2.0;
	conic.at<double>(0, 2) = d / 2.0;   conic.at<double>(1, 0) = b / 2.0;
	conic.at<double>(1, 1) = c;         conic.at<double>(1, 2) = e / 2.0;
	conic.at<double>(2, 0) = d / 2.0;   conic.at<double>(2, 1) = e / 2.0;
	conic.at<double>(2, 2) = f;
}

cv::Mat Ellipse::ref_mu()
{
	return ref_mu_;
}

cv::Mat Ellipse::ref_cov()
{
	return ref_cov_;
}

void Ellipse::set_ref_mu(const cv::Mat& mu)
{
	mu.copyTo(ref_mu_);
}

void Ellipse::set_ref_cov(const cv::Mat& cov)
{
	cov.copyTo(ref_cov_);
}

void Ellipse::set_eta(double eta)
{
	eta_ = eta;
}
/*	
	mu_x_ = (ini_mu_x_ - ini_ref_mu_x_) + ref_mu_x_ + action.at<double>(0, 0);
	mu_y_ = (ini_mu_y_ - ini_ref_mu_y_) + ref_mu_y_ + action.at<double>(1, 0);
	angle_ = (ini_angle_ - ini_ref_angle_) + ref_angle_ + action.at<double>(2, 0);
	long_axis_ = (ini_long_axis_ - ini_ref_long_axis_) + ref_long_axis_ + action.at<double>(3, 0);
	short_axis_ = (ini_short_axis_ - ini_ref_short_axis_) + ref_short_axis_ + action.at<double>(4, 0);
*/

/*
	mu_x_ = ini_mu_x_ + action.at<double>(0, 0);
	mu_y_ = ini_mu_y_ + action.at<double>(1, 0);
	angle_ = ini_angle_ + action.at<double>(2, 0);
	long_axis_ = ini_long_axis_ + action.at<double>(3, 0);
	short_axis_ = ini_short_axis_ + action.at<double>(4, 0);
*/

/*	
	mu_x_ = ini_mu_x_ + action.at<double>(0, 0);
	mu_y_ = ini_mu_y_ + action.at<double>(1, 0);
	angle_ = ini_angle_ + action.at<double>(2, 0);
	long_axis_ = ini_long_axis_ + action.at<double>(3, 0);
	short_axis_ = ini_short_axis_ + action.at<double>(4, 0);
*/

// eigen values stored in descending order, eigen vectors stored in rows with the order corresponding to eigen values
/*	
	ref_cov_.at<double>(0, 0) = 7.000000000000002;
	ref_cov_.at<double>(0, 1) = 5.196152422706633;
	ref_cov_.at<double>(1, 0) = 5.196152422706633;
	ref_cov_.at<double>(1, 1) = 12.999999999999998;
*/

//ref_mu_.at<double>(0, 0) = 0;
//ref_mu_.at<double>(1, 0) = 0;
//ref_cov_.at<double>(0, 0) = 1.0;
//ref_cov_.at<double>(0, 1) = 0.0;
//ref_cov_.at<double>(1, 0) = 0.0;
//ref_cov_.at<double>(1, 1) = 1.0;