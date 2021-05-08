#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <cassert>

using namespace std;
using namespace cv;

vector<float> read_bin(string file_path) {
    ifstream rf(file_path, ios::out | ios::binary);
    long begin, end;
    begin = rf.tellg();
    rf.seekg (0, ios::end);
    end = rf.tellg();
    rf.seekg (0, ios::beg);
    vector<float> pts((end-begin)/sizeof(float));
    int idx = 0;
    while(!rf.eof()) {
        rf.read((char *) &pts[idx++], sizeof(float));
    }
    rf.close();
    return pts;
}

vector<std::string> read_name_file(string file_path) {
    ifstream file;
    string line;
    file.open(file_path);
    vector<string> names;
    while (getline(file, line)) {
        names.push_back(line);
    }
    file.close();
    return(names);
}

cv::Mat read_facebank(string bin_path, int emb_num, int num_faces) {
    vector<float> face_emb = read_bin(bin_path);
    // for (size_t i = 0; i<100; ++i) {
    //     std::cout << face_emb[i] << std::endl;
    // }
    float *face_emb_data = face_emb.data();
    // cv::Mat feature(num_faces, emb_num, CV_32FC1);
    cv::Mat feature(num_faces, emb_num, CV_32FC1, face_emb_data);
    // assert(num_faces == (face_emb.size() / emb_num) and "facebank size doesn't match!");
    // for (size_t i=0; i<num_faces; ++i) {
    //     cv::Mat onefeature(1, emb_num, CV_32FC1, face_emb_data + i * emb_num);
    //     cv::normalize(onefeature, onefeature);
    //     onefeature.copyTo(feature.row(i));
    // }
    return feature.t(); // (facebank_feature * num_faces)
}

