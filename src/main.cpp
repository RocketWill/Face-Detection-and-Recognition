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
    std::string name_file = config["names_file"].as<std::string>();
    bool update = config["update"].as<bool>();
    if (update) {
        std::cout << "preparing face bank..." << std::endl;
        std::string facebank_dir = config["facebank_dir"].as<std::string>();
        auto facebank_mapping = prepare_facebank_raw_data(facebank_dir);
        ArcFace.PrepareFacebank(facebank_mapping, facebank_file, name_file);
    }
    int emb_num = config["EMB_NUM"].as<int>();
    int num_faces = config["NUM_FACES"].as<int>();
    cv::Mat facebank_features = read_facebank(facebank_file, emb_num, num_faces);
    cout << facebank_features.cols << " " << facebank_features.rows << endl; // (10, 512)

    std::vector<std::string> name_ids = read_name_file(name_file);

    std::vector<std::vector<RetinaFace::FaceRes>> all_dets = RetinaFace.InferenceImage(image_path);

    cv::Mat img = cv::imread(image_path);
    cv::Mat vis = img.clone();
    for (size_t i=0; i<all_dets[0].size(); ++i) {
        cv::Mat aligned_face = RetinaFace.align_face(img, all_dets[0][i].keypoints, 112);
        all_dets[0][i].aligned_face = aligned_face;
        cv::Mat face_feature = ArcFace.InferenceImage(aligned_face);
        cv::Mat exp_face_feature = expand_face_feature(face_feature, num_faces, emb_num);
        cv::Mat dist = exp_face_feature - facebank_features.t();
        cv::Mat dist_pow;
        cv::pow(dist, 2, dist_pow);
        cv::Mat dist_sum;

        // calculate distance
        cv::reduce(dist_pow, dist_sum, 1, CV_REDUCE_SUM, CV_32F);
        cout << dist_sum << endl;

        // calculate score
        auto score_matrix = face_feature * facebank_features;
        std::cout << score_matrix << std::endl;
        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;
        std::string name = "Unknown";
        cv::minMaxLoc(score_matrix, &minVal, &maxVal, &minLoc, &maxLoc);
        if (maxVal < 0.2) {
            std::cout << "Cannot recognize your face." << std::endl;
        } else {
            name = name_ids[maxLoc.x];
            std::cout << "Guess you are " << name << "!" << std::endl;
        }
        
        // visualize
        int x = all_dets[0][i].face_box.x - all_dets[0][i].face_box.w / 2;
        int y = all_dets[0][i].face_box.y - all_dets[0][i].face_box.h / 2;
        cv::Rect box(x, y, all_dets[0][i].face_box.w, all_dets[0][i].face_box.h);
        cv::rectangle(vis, box, cv::Scalar(0, 255, 255), 2, cv::LINE_8, 0);
        cv::putText(vis, name, cv::Point(x, y - 5),
                            cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    }

    cv::imwrite("./rec.jpg", vis);
    return 0;
}