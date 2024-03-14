#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <tuple>
#include <opencv2/opencv.hpp>

class Dataset
{
private:
    float image_height;
    float image_width;
    float fx;
    float fy;
    float cx;
    float cy;
    float crop_edge;
    float png_depth_scale;
    std::string base_dir;

    template<typename T>
    T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return T();
            }

        }
        else{
            found = true;
            return (T) node;
        }
    }
public:
    ~Dataset(const std::string & configFile, const std::string & basedir);

    virtual std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> get_filepaths();
}

#endif // DATASET_H