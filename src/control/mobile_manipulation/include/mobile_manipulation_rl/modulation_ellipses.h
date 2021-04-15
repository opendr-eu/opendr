#ifndef MODULATION
#define MODULATION

#include <mobile_manipulation_rl/ellipse.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <sstream>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>

// GP for ellipse regression from IRM
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <gp/gp.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/bind.hpp>
#include "gp/gp_utils.h"
#include "gp/rprop.h"
#include "tf/transform_datatypes.h"

namespace modulation_ellipses {

  class Modulation {
  private:
    std::vector<ellipse::Ellipse> ellipses_;
    Eigen::Vector3d position_;
    Eigen::VectorXd gripper_position_;
    double gripper_yaw_speed_;
    Eigen::VectorXf speed_;
    bool do_ir_modulation_ = true;
    int first_ellipse_ = 0;

    std::vector<double> lambda_;
    std::vector<double> gamma_;
    std::vector<double> real_gamma_;
    double gamma_alpha_;
    std::vector<std::vector<double>> xi_wave_;

    void computeXiWave();
    void computeGamma();
    void computeGammaAlpha(int ellipseNr);
    double computeWeight(int k);
    std::vector<double> computeEigenvalue(int k);
    std::vector<double> computeHyperplane(int k);
    std::vector<std::vector<double>> computeEBase(int k, std::vector<double> normal);
    Eigen::MatrixXf assembleD_k(int k);
    Eigen::MatrixXf assembleE_k(int k);

    // KNN for the lookup for base orientation
    cv::Ptr<cv::ml::KNearest> knnAperture_;
    cv::Ptr<cv::ml::KNearest> knnAngle_;
    cv::ml::KNearest *knnCosAngle_;
    cv::ml::KNearest *knnSinAngle_;
    // UBUNTU 14 VERSION:
    // cv::KNearest knnAperture_;
    // cv::KNearest knnAngle_;
    // cv::KNearest knnCosAngle_;
    // cv::KNearest knnSinAngle_;

    // GP Regression stuff for the ellipses
    boost::shared_ptr<libgp::GaussianProcess> gp_radiiX_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_radiiY_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_centerX_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_centerY_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_phi_cos_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_phi_sin_outer;
    boost::shared_ptr<libgp::GaussianProcess> gp_radiiX_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_radiiY_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_centerX_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_centerY_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_phi_cos_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_phi_sin_inner;
    boost::shared_ptr<libgp::GaussianProcess> gp_phi_inner;

    bool existsTest(const std::string &name);

  public:
    Modulation(Eigen::Vector3d &curr_position, Eigen::VectorXf &curr_speed);
    Modulation();
    ~Modulation();

    Eigen::MatrixXf modulation_;
    Eigen::MatrixXf modulation_gripper_;

    void updateSpeedAndPosition(Eigen::Vector3d &curr_pose, Eigen::VectorXf &curr_speed, Eigen::VectorXd &curr_gripper_pose);
    void computeModulationMatrix();

    std::vector<ellipse::Ellipse> &getEllipses();
    std::vector<ellipse::Ellipse> &getEllipses(Eigen::Vector3d &curr_pose, Eigen::VectorXf &curr_speed,
                                               Eigen::VectorXd &curr_gripper_pose);

    void setEllipses();

    static double computeL2Norm(std::vector<double> v);

    Eigen::VectorXf compModulation();
    void run(Eigen::VectorXf &curr_pose, Eigen::VectorXf &curr_speed);

    visualization_msgs::MarkerArray getEllipsesVisMarker(Eigen::VectorXf &curr_pose, Eigen::VectorXf &curr_speed);
  };

}  // namespace modulation_ellipses

#endif
