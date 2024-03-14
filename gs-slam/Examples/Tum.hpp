#ifndef TUM_H
#define TUM_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <Dataset.hpp>

class Tum : public Dataset
{
private:
    std::vector<std::string> parse_list(const std::string & path);
public:
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> get_filepaths();
}

#endif // TUM_H