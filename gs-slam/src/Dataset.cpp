#include <Dataset.hpp>

template<>
float Dataset::readParameter<float>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0.0f;
        }
    }
    else if(!node.isReal()){
        std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.real();
    }
}

template<>
int Dataset::readParameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0;
        }
    }
    else if(!node.isInt()){
        std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.operator int();
    }
}

template<>
std::string Dataset::readParameter<string>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return string();
        }
    }
    else if(!node.isString()){
        std::cerr << name << " parameter must be a string, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.string();
    }
}

template<>
cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return cv::Mat();
        }
    }
    else{
        found = true;
        return node.mat();
    }
}

Dataset::Dataset(const std::string & configFile, const std::string & basedir)
    : base_dir(basedir)
{
    cv::FileStorage fSettings(configFile, cv::FileStorage::READ);

    bool found;

    fx = readParameter<float>(fSettings, "Camera1.fx", found);
    fy = readParameter<float>(fSettings, "Camera1.fy", found);
    cx = readParameter<float>(fSettings, "Camera1.cx", found);
    cy = readParameter<float>(fSettings, "Camera1.cy", found);
}