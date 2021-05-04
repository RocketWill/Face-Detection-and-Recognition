#include <iostream>
#include <fstream>
#include <algorithm> 
#include <utility>
#include <vector>
#include "RetinaFace.h"
#include "read_facebank.hpp"
#include "arcface.h"
#include "yaml-cpp/yaml.h"
#include "utils.hpp"

using namespace std;

int main (int argc, char *argv[]) { 
    if (argc < 4)
    {
        std::cout << "Please create config file and folder name!" << std::endl;
        return -1;
    }

    std::string r50_config_file = argv[1];
    std::string arcface_config_file = argv[2];
    std::string image_path = argv[3];

    RetinaFace RetinaFace(r50_config_file);
    RetinaFace.LoadEngine();

    ArcFace ArcFace(arcface_config_file);
    ArcFace.LoadEngine();    

    YAML::Node root = YAML::LoadFile(arcface_config_file);
    YAML::Node config = root["arcface"];
    std::string facebank_file = config["facebank_file"].as<std::string>();
    bool update = config["update"].as<bool>();
    if (update) {
        std::cout << "preparing face bank..." << std::endl;
        std::string facebank_dir = config["facebank_dir"].as<std::string>();
        auto facebank_mapping = prepare_facebank_raw_data(facebank_dir);
        ArcFace.PrepareFacebank(facebank_mapping, facebank_file);
    }
    int emb_num = config["EMB_NUM"].as<int>();
    int num_faces = config["NUM_FACES"].as<int>();
    cv::Mat facebank_features = read_facebank(facebank_file, emb_num, num_faces);
    cout << facebank_features.cols << " " << facebank_features.rows << endl; // (10, 512)


    std::vector<std::vector<RetinaFace::FaceRes>> all_dets = RetinaFace.InferenceImage(image_path);
    cv::Mat img = cv::imread(image_path);
    for (size_t i=0; i<all_dets[0].size(); ++i) {
        cv::Mat aligned_face = RetinaFace.align_face(img, all_dets[0][i].keypoints, 112);
        all_dets[0][i].aligned_face = aligned_face;
        cv::Mat face_feature = ArcFace.InferenceImage(aligned_face);

        std::cout << face_feature * facebank_features << std::endl;
        cv::imwrite(std::to_string(i)+"_align.jpg", aligned_face);
    }

    
    return 0;

}