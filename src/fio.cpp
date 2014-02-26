#include "../inc/fio.h"

using namespace cv;
using namespace std;

// the additional declaration for separating template function in .h and .cpp files
void FileIO::ReadMatInt(Mat& dst, int h, int w, string name){
    ifstream file_pt((char*)name.c_str(), ios::in|ios::binary);
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            file_pt.read(reinterpret_cast<char*>(&(dst.at<int>(i, j))), sizeof(int));
        }
    }
    file_pt.close();
}

void FileIO::ReadMatFloat(Mat& dst, int h, int w, string name){
    ifstream file_pt((char*)name.c_str(), ios::in|ios::binary);
	float a = 0;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            file_pt.read(reinterpret_cast<char*>(&a), sizeof(float));
			dst.at<double>(i, j) = (double)a;
			// file_pt.read(reinterpret_cast<char*>(&(dst.at<double>(i, j))), sizeof(float));
        }
    }
    file_pt.close();
}

void FileIO::ReadMatDouble(Mat& dst, int h, int w, string name){
    ifstream file_pt((char*)name.c_str(), ios::in|ios::binary);
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            file_pt.read(reinterpret_cast<char*>(&(dst.at<double>(i, j))), sizeof(double));
        }
    }
    file_pt.close();
}

void FileIO::WriteMatInt(Mat& src, int h, int w, string name){

    FILE* file_pt = fopen((char*)name.c_str(), "wb");
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            fwrite(&(src.at<int>(i, j)), sizeof(int), 1, file_pt);
        }
    }
    fclose(file_pt);
}

void FileIO::WriteMatFloat(Mat& src, int h, int w, string name){
    FILE* file_pt = fopen((char*)name.c_str(), "wb");
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            fwrite(&(src.at<float>(i, j)), sizeof(float), 1, file_pt);
        }
    }
    fclose(file_pt);
}

void FileIO::RecordMatDouble(Mat& src, int h, int w, string name, int append_flag){
  FILE* file_pt;
  if(append_flag)
    file_pt = fopen((char*)name.c_str(), "ab");
  else
    file_pt = fopen((char*)name.c_str(), "wb");
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      fwrite(&(src.at<double>(i, j)), sizeof(double), 1, file_pt);
    }
  }
  fclose(file_pt);
}

void FileIO::AppendMatFloat(Mat& src, int h, int w, string name){
    FILE* file_pt = fopen((char*)name.c_str(), "ab");
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            fwrite(&(src.at<double>(i, j)), sizeof(double), 1, file_pt);
        }
    }
    fclose(file_pt);
}

void FileIO::WriteMatDouble(Mat& src, int h, int w, string name){

    FILE* file_pt = fopen((char*)name.c_str(), "wb");
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            fwrite(&(src.at<double>(i, j)), sizeof(double), 1, file_pt);
        }
    }
    fclose(file_pt);
}
