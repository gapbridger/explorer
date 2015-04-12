#include "gtest/gtest.h"
#include "explorer.h"
#include "loader.h"
#include "fio.h"

class UnifiedLearningTest:public ::testing::Test
{
protected:
    virtual void SetUp();
    virtual void TearDown();
public:
	char* test_data_dir_prefix_;
	double double_epsilon_;
};


