#include <Tum.hpp>

std::vector<std::string> parse_list(const std::string & path)
{
    std::fstream readFile;
    readFile.open(path);

    if(!readFile)
    {
        std::cerr << "ERROR: failed to open file " << std::endl;  //if the file cannot be opened an error is displayed
        exit(0); //if it cannot open the console terminates
    }
    else
    {
        std::cerr << "File successfully opened" << std::endl;
    }

    while(readFile >> storeFile)
    {
        if(readFile.bad())
        {
            std::cerr << "File failed to read " << std::endl;
            break; //loop terminates
        } 
        else
        {
            for (int i = 0; i < sizeof(myWord)/sizeof(myWord[0]); i++)
            {
                readFile >> myWord[i]; 
                count++;
            }
        } 
    }

    readFile.close();
}


std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> Tum::get_filepaths()
{
    namespace fs = std::filesystem;

    std::string pose_list_path;
    if (fs::exists(base_dir + "/groundtruth.txt")) 
    {
        pose_list_path = base_dir + "/groundtruth.txt";
    }
    else if (fs::exists(base_dir + "/pose.txt")) 
    {
        pose_list_path = base_dir + "/pose.txt";
    }

    std::vector<std::string> image_list_paths = parse_list(base_dir + "/rgb.txt");
    std::vector<std::string> depth_list_paths = parse_list(base_dir + "/depth.txt");



    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        std::string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}