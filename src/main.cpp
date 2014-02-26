#include "../inc/explorer.h"
#include "../inc/transform.h"

int main(int argc, char* argv[])
{
	// parameters...
	char input_dir[400];
	int num_iteration = 100000; 
	int single_frame_flag = 0;
	int display_flag = 0;
	int test_idx = 0;
	int target_idx = 0;	
	int directory_id = atoi(argv[1]);
	int train_flag = atoi(argv[2]);	
	int expanding_period = num_iteration / 2;
	
	//Transform transform(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle);
	//transform.CheckInvGradient();
	
	if(argc == 4)
	{
		if(train_flag == 1)
			num_iteration = atoi(argv[3]);			
		else if(train_flag == 0)
			single_frame_flag = atoi(argv[3]);
		else
			return 0;
	}
	if(argc == 5)
	{
		if(train_flag == 1)
		{
			num_iteration = atoi(argv[3]);
			expanding_period = atof(argv[4]);
		}
		else if(train_flag == 0)
		{
			single_frame_flag = atoi(argv[3]);
			display_flag = atoi(argv[4]);
		}
		else
			return 0;
		
	}
	if(argc == 6){
		if(train_flag == 0)
		{
			single_frame_flag = atoi(argv[3]);
			display_flag = atoi(argv[4]);
			test_idx = atoi(argv[5]);
		}
		else
			return 0;
	}

	// initialization
	std::cout << "training iteration: " << num_iteration << std::endl;
	std::cout << "expanding period: " << expanding_period << std::endl;
	Explorer explorer(directory_id, num_iteration, expanding_period);
	
	if(train_flag == 1)
		explorer.Train();	
	else if(train_flag == 0)
		explorer.Test(display_flag, single_frame_flag, 0, 2319, test_idx, 0, 0);
	else if(train_flag == 2)
	{
		Loader loader(5, 8, directory_id);
		loader.RecordSiftKeyPoints();
	}

	else
		exit(0);
		
}
