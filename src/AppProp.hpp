#ifndef _APPPROP_H_
#define _APPPROP_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <algorithm>
#include <chrono>

class AppProp {
private:
    int const m_ = 200;
    double const alpha_a_ = 50;
    double const alpha_s_ = 100;
//    double const alpha_a_ = 138;
//    double const alpha_s_ = 18;
//    double const alpha_a_ = 0.2;
//    double const alpha_s_ = 0.2;

    cv::Mat src_img_;
    cv::Mat final_mask_;
    cv::Mat final_img_;
    int height_;
    int width_;
    int n_;

    std::vector<cv::Vec<double, 9>> features;


    void render_final(){
        for(int i = 0; i < height_; i++){
            for(int j = 0; j < width_; j++){
                final_img_.at<cv::Vec3b>(i, j) = src_img_.at<cv::Vec3b>(i, j);
                double l = final_img_.at<cv::Vec3b>(i, j)[0] + final_mask_.at<uint8_t>(i, j);
                if(l>255) l = 255;
                if(l<0) l = 0;
                final_img_.at<cv::Vec3b>(i, j)[0] = uint8_t(l);
            }
        }
    }

    void convert_vec_to_mask(Eigen::VectorXd const & vector, cv::Mat &mask) const {
        mask = cv::Mat(src_img_.size(), CV_8UC1);
        for(int i = 0; i < height_; i++){
            for(int j = 0; j < width_; j++){
                mask.at<uint8_t>(i, j) = (uint8_t)vector(i*width_ + j);
            }
        }
    }

    void get_feature(){
        for(int i=0;i<height_;i++){
            for(int j=0;j<width_;j++){
                cv::Vec<double, 9> ret;
                cv::Vec3b lab = src_img_.at<cv::Vec3b>(i, j);
                ret[0] = lab[0];
                ret[1] = lab[1];
                ret[2] = lab[2];

                int sz = 3 / 2;
                int si = std::max(i - sz, 0);
                int ei = std::min(i + sz, height_-1);
                int sj = std::max(j - sz, 0);
                int ej = std::min(j + sz, width_-1);

                cv::Mat neighbor = src_img_(cv::Rect(sj, si,
                                                     ej - sj + 1, ei - si + 1));
                cv::Scalar mean;
                cv::Scalar std_dev;
                cv::meanStdDev(neighbor, mean, std_dev);

                ret[3] = mean[0];
                ret[4] = mean[1];
                ret[5] = mean[2];
                ret[6] = std_dev[0];
                ret[7] = std_dev[1];
                ret[8] = std_dev[2];

                features[i*width_+j] = ret;
            }
        }
    }

public:
    AppProp(cv::Mat const & source_img){
        cv::cvtColor(source_img, src_img_, CV_BGR2Lab);
        cv::cvtColor(source_img, final_img_, CV_BGR2Lab);

        width_ = src_img_.cols;
        height_ = src_img_.rows;
        n_ = width_*height_;

        features.resize(n_);

        get_feature();
    }

    void propagating(cv::Mat const & init_mask_g, cv::Mat const & init_mask_w){
        Eigen::VectorXd g(n_), w(n_), one_n(n_);
        Eigen::MatrixXd U(n_, m_);
        double lambda = 1.0;
        double sum_w = 0.0;

        g.setZero();
        w.setZero();
        one_n.setOnes();
        U.setZero();

        for(int i=0;i<height_;i++){
            for(int j=0;j<width_;j++){
                int idx = i*(width_)+j;
                g(idx) = init_mask_g.at<double>(i, j);
                w(idx) = init_mask_w.at<double>(i, j);
                sum_w += w(idx);
            }
        }

        lambda = sum_w * 1.0 / n_;

        std::vector<int> random_idx(n_);
        std::vector<int> random_idx_r(n_);
        for (int i = 0; i < n_; ++i) {
            random_idx[i] = i;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(random_idx.begin(), random_idx.end(), gen);
        for (int i = 0; i < n_; ++i) {
            random_idx_r[random_idx[i]] = i;
        }

        Eigen::VectorXd tmp(n_);
        for (int i = 0; i < n_; ++i) {
            tmp[i] = g[random_idx[i]];
        }
        g = tmp;
        for (int i = 0; i < n_; ++i) {
            tmp[i] = w[random_idx[i]];
        }
        w = tmp;

        for(int i = 0; i < n_; i++){
            for(int j = 0; j < m_; j++){
                int idx_i = random_idx[i];
                int idx_j = random_idx[j];
                int idx_i_x = idx_i / width_;
                int idx_i_y = idx_i % width_;
                int idx_j_x = idx_j / width_;
                int idx_j_y = idx_j % width_;
                cv::Vec<double, 9> feature_i = features[idx_i];
                cv::Vec<double, 9> feature_j = features[idx_j];
                cv::Vec2i location_i = cv::Vec2i(idx_i_x, idx_i_y);
                cv::Vec2i location_j = cv::Vec2i(idx_j_x, idx_j_y);
                double z_ij = cv::exp(-cv::norm(feature_i-feature_j, cv::NORM_L2SQR)/alpha_a_)*cv::exp(-cv::norm(location_i-location_j, cv::NORM_L2SQR)/alpha_s_);
                U(i, j) = z_ij;
            }
        }

        Eigen::MatrixXd U_tran = U.transpose();
        Eigen::MatrixXd A = U.block(0, 0, m_, m_);
        double regularizationParameter = 1e-3;
        A = A + regularizationParameter * Eigen::MatrixXd::Identity(m_, m_);
        Eigen::MatrixXd A_inv = A.inverse();


        Eigen::VectorXd a = U * (A_inv * (U_tran * (w.asDiagonal() * one_n))); // U A^-1 U^T M W 1_n
        Eigen::VectorXd b = U * (A_inv * (U_tran * one_n));                    // U A^-1 U^T M 1_n

        a /= lambda*2.0;
        Eigen::VectorXd d_inv = a + b;
        for (int i = 0; i < n_; i++){
            d_inv(i) = 1.0 / d_inv(i);
        }

        Eigen::VectorXd c = U * (A_inv * (U_tran * (w.asDiagonal() * g))); // U A^-1 U^T M W g
        Eigen::MatrixXd U_TD_1 = U_tran * d_inv.asDiagonal();                                  // U^T D^-1
        Eigen::MatrixXd D_1U = d_inv.asDiagonal() * U;                                      // D^-1 U
        Eigen::MatrixXd mid = (U_TD_1 * U - A).inverse();                                 // (-A + U^T D^-1 U)^-1

        Eigen::VectorXd e = d_inv.asDiagonal() * c - D_1U * (mid * (U_TD_1 * c));
        e /= lambda * 2;

        for (int i = 0; i < n_; ++i) {
            tmp[i] = e[random_idx_r[i]];
        }
        e = tmp;

        convert_vec_to_mask(e, final_mask_);
        cv::imwrite("final_mask.png", final_mask_);
        render_final();
    }

    void get_edit_result(cv::Mat &result){
        result = final_img_;
        cv::cvtColor(result, result, CV_Lab2BGR);
    }
};

namespace APGUI{

enum{
    MASK_NEGATIVE = 0,
    MASK_POSITIVE = 1
};

class GUI{
public:
    cv::Vec3b const GREEN = cv::Vec3b(0, 255, 0);
    cv::Vec3b const RED = cv::Vec3b(0, 0, 255);
\
    bool mouse_down_ = 0;
    int brightness_increase_ = 30;
    cv::Mat mask_w_;
    cv::Mat mask_g_;
    cv::Mat src_img_;
    cv::Mat editing_img_;
    cv::Mat final_img_;
    std::string editing_window_name_;
    std::string final_window_name_;
    int tool_type_ = 0;

public:
    auto show_img()->void{
        cv::imshow(editing_window_name_, editing_img_);
        cv::imshow(final_window_name_, final_img_);
    }

    auto mouse_handle(int event, int x, int y, int flags, void* param)->void{
        if(tool_type_==1){
            positive_tool(event, x, y, flags, param);
        }
        if(tool_type_==2){
            negative_tool(event, x, y, flags, param);
        }
    }

    auto point_to_valid(int x, int y)->cv::Point{
        x = std::max(0, x);
        y = std::max(0, y);
        x = std::min(src_img_.size().width-1, x);
        y = std::min(src_img_.size().height-1, y);
        return cv::Point(x, y);
    }

    auto point_to_valid(cv::Point po)->cv::Point{
        return point_to_valid(po.x, po.y);
    }

    auto brushAssign(int x, int y, uint8_t msk)->void{
        int const sz = 10;
        for(int i=-sz;i<=sz;i++){
            for(int j=-sz;j<=sz;j++){
                cv::Point po = point_to_valid(x+i, y+j);
                if(msk == MASK_POSITIVE){
                    mask_w_.at<double>(po) = 0.8;
                    mask_g_.at<double>(po) = brightness_increase_;
                    editing_img_.at<cv::Vec3b>(po) = GREEN;
                }
                else{
                    mask_w_.at<double>(po) = 1.0;
                    mask_g_.at<double>(po) = 0;
                    editing_img_.at<cv::Vec3b>(po) = RED;
                }
            }
        }
    }

    auto positive_tool(int event, int x, int y, int flags, void* param)->void{
        if(event==CV_EVENT_LBUTTONDOWN){
            mouse_down_ = 1;
            brushAssign(x, y, MASK_POSITIVE);
        }
        else if(event==CV_EVENT_LBUTTONUP && mouse_down_){
            mouse_down_ = 0;
            brushAssign(x, y, MASK_POSITIVE);
        }
        else if(event==CV_EVENT_MOUSEMOVE && mouse_down_){
            brushAssign(x, y, MASK_POSITIVE);
            show_img();
        }
    }

    auto negative_tool(int event, int x, int y, int flags, void* param)->void{
        if(event==CV_EVENT_LBUTTONDOWN){
            mouse_down_ = 1;
            brushAssign(x, y, MASK_NEGATIVE);
        }
        else if(event==CV_EVENT_LBUTTONUP && mouse_down_){
            mouse_down_ = 0;
            brushAssign(x, y, MASK_NEGATIVE);
        }
        else if(event==CV_EVENT_MOUSEMOVE && mouse_down_){
            brushAssign(x, y, MASK_NEGATIVE);
            show_img();
        }
    }

}gui;

auto mouse_handle(int event, int x, int y, int flags, void* param)->void{
    gui.mouse_handle(event, x, y, flags, param);
}

void gui_main(cv::Mat const & img, bool extra = false){
//    std::cout<<"工具：1加强、2不加强\n";
//    std::cout<<"操作：q退出并保存，s进行图像处理，r重置\n";

    gui.editing_window_name_ = "AppProp";
    cv::namedWindow(gui.editing_window_name_, CV_WINDOW_AUTOSIZE);
    gui.final_window_name_ = "result";
    cv::namedWindow(gui.final_window_name_, CV_WINDOW_AUTOSIZE);

    gui.src_img_.create(img.size(), CV_8UC3);
    img.copyTo(gui.src_img_);
    gui.editing_img_.create(img.size(), CV_8UC3);
    img.copyTo(gui.editing_img_);
    gui.final_img_.create(img.size(), CV_8UC3);
    img.copyTo(gui.final_img_);

    gui.mask_g_ = cv::Mat::zeros(img.size(), CV_64F);
    gui.mask_w_ = cv::Mat::zeros(img.size(), CV_64F);

    cv::setMouseCallback(gui.editing_window_name_, mouse_handle);

    gui.show_img();
    AppProp app_prop(img.clone());

    while(true){
        int key = cv::waitKey();
        if(key=='1'){
            gui.tool_type_ = 1;
        }
        else if(key=='2'){
            gui.tool_type_ = 2;
        }
        else if(key=='q'){
            cv::imwrite("res.png", gui.final_img_);
            break;
        }
        else if(key=='s'){
            cv::imwrite("init_mask.png", gui.editing_img_);
            auto start = std::chrono::high_resolution_clock::now();

            app_prop.propagating(gui.mask_g_.clone(), gui.mask_w_.clone());
            app_prop.get_edit_result(gui.final_img_);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            if(extra) std::cout << duration << "ms" << std::endl;
        }
        else if(key=='r'){
            img.copyTo(gui.editing_img_);
            img.copyTo(gui.final_img_);
            gui.mask_g_ = cv::Mat::zeros(img.size(), CV_64F);
            gui.mask_w_ = cv::Mat::zeros(img.size(), CV_64F);
        }
        gui.show_img();
    }

    cv::destroyWindow(gui.editing_window_name_);
    cv::destroyWindow(gui.final_window_name_);
}

} // namespace GCGUI

#endif //_APPPROP_H_
