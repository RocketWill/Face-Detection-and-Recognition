#include <iostream>
#include <fstream>
#include <algorithm> 
#include <utility>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <map>

std::vector<cv::String> get_subdirs(std::string folder) {
    std::vector<cv::String> filenames;
    std::vector<cv::String> results;
    cv::utils::fs::glob(folder, "*", filenames, false, true);
    for (auto path: filenames) {
        if (cv::utils::fs::isDirectory(path)) {
            results.push_back(path);
        }
    }
    return results;
}

std::string basename(std::string path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

std::map<std::string, std::vector<cv::String>> get_person_img_map(std::vector<cv::String> dirs) {
    std::map<std::string, std::vector<cv::String>> mapping;
    for (auto dir_: dirs) {
        std::vector<cv::String> filenames;
        cv::utils::fs::glob(dir_, "*", filenames, false, false);
        mapping[basename(dir_)] = filenames;
    }
    return mapping;
}

std::map<std::string, std::vector<cv::String>> prepare_facebank_raw_data(std::string folder) {
    auto subdirs = get_subdirs(folder);
    auto mapping = get_person_img_map(subdirs);
    return mapping;
}
