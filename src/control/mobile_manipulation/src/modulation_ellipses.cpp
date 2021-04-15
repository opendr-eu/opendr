#include <mobile_manipulation_rl/modulation_ellipses.h>

using namespace std;
namespace modulation_ellipses {
  bool Modulation::existsTest(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
  }

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

  Modulation::Modulation(Eigen::Vector3d &curr_position, Eigen::VectorXf &curr_speed) :
    modulation_(2, 2),
    modulation_gripper_(2, 2),
    speed_(curr_speed),
    position_(curr_position),
    gripper_position_(7) {
    modulation_ << 1, 0, 0, 1;
    gripper_position_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  }

  Modulation::Modulation() : modulation_(2, 2), modulation_gripper_(2, 2), speed_(3), position_(3), gripper_position_(7) {
    speed_ << 1, 1, 1;
    position_ << 1, 1, 1;
    modulation_ << 1, 0, 0, 1;
    gripper_position_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  }

  Modulation::~Modulation() {}

  std::vector<ellipse::Ellipse> &Modulation::getEllipses() { return ellipses_; }

  std::vector<ellipse::Ellipse> &Modulation::getEllipses(Eigen::Vector3d &curr_pose, Eigen::VectorXf &curr_speed,
                                                         Eigen::VectorXd &curr_gripper_pose) {
    // updateSpeedAndPosition(curr_pose,curr_speed,curr_gripper_pose);
    return ellipses_;
  }

  void Modulation::setEllipses() {
    std::vector<ellipse::Ellipse> ellipses;

    ellipses.push_back(ellipse::Ellipse(gripper_position_[0], gripper_position_[1], "inner"));
    ellipses.push_back(ellipse::Ellipse(gripper_position_[0], gripper_position_[1], "outter"));
    // ellipses.push_back(ellipse::Ellipse(1.565,-1.3,M_PI/2,1.18));// (1.515,-1.3,M_PI/2,1.18));
    // ellipses.push_back(ellipse::Ellipse(1.54,1.237,M_PI/2,1.20814));//(1.49,1.187,M_PI/2,1.20814));
    // ellipses.push_back(ellipse::Ellipse(4.69,0.67,-0.7,0.7)); // Kallax
    // ellipses.push_back(ellipse::Ellipse(4.1,0.8,-0.7,0.7)); // pose1
    // ellipses.push_back(ellipse::Ellipse(4.3,-0.6,-0.7,0.7)); //pose2
    // ellipses.push_back(ellipse::Ellipse(4.1,-0.5,-0.7,0.7)); //pose2
    ellipses_ = ellipses;

    // load data for the knn lookup for base orientation:
    std::string fpath;
    if (existsTest("Ellipse_modulation_models/knnDataAngle.csv")) {
      fpath = "Ellipse_modulation_models/";
    } else if (existsTest("../Ellipse_modulation_models/")) {
      fpath = "../Ellipse_modulation_models/";
    } else {
      throw std::runtime_error("Ellipse_modulation_models folder not found. Please run from project root.");
    }
    cv::Ptr<cv::ml::TrainData> mlDataTrain = cv::ml::TrainData::loadFromCSV(fpath + "knnDataAngle.csv", 0, -1, -1);
    cv::Mat Mat_Train_Data = mlDataTrain->getTrainSamples();
    cv::Mat Mat_Label_Data = mlDataTrain->getTrainResponses();

    knnAngle_ = cv::ml::KNearest::create();
    knnAngle_->train(Mat_Train_Data, cv::ml::ROW_SAMPLE, Mat_Label_Data);

    cv::Ptr<cv::ml::TrainData> mlDataTrainAperture = cv::ml::TrainData::loadFromCSV(fpath + "knnDataAperture.csv", 0, -1, -1);
    cv::Mat Mat_Label_Data_Aperture = mlDataTrainAperture->getTrainResponses();
    knnAperture_ = cv::ml::KNearest::create();
    knnAperture_->train(Mat_Train_Data, cv::ml::ROW_SAMPLE, Mat_Label_Data_Aperture);

    // UBUNTU 14 VERSION:
    // CV_EXPORTS::CvMLData mlDataTrain;
    // mlDataTrain.read_csv("/home/twelsche/Desktop/IROS_18/trainData.csv");
    // const CvMat* Mat_Train_Data = mlDataTrain.get_values();
    // // Data for mean angle
    // CV_EXPORTS::CvMLData mlDataAngleResponse;
    // mlDataAngleResponse.read_csv("/home/twelsche/Desktop/IROS_18/responseAngle0Data.csv");
    // const CvMat* Mat_ResponseAngle_Data = mlDataAngleResponse.get_values();
    // // Data for angle aperture
    // // ROS_INFO("Samples for k-NN loaded");
    // CV_EXPORTS::CvMLData mlDataResponse;
    // mlDataResponse.read_csv("/home/twelsche/Desktop/IROS_18/responseApertureData.csv");
    // const CvMat* Mat_Response_Data = mlDataResponse.get_values();
    // // ROS_INFO("Aperture Data for k-NN loaded");

    // const Mat& sampleIdx=Mat();
    // knnAperture_.train(Mat_Train_Data, Mat_Response_Data,sampleIdx,true,32,false);
    // knnAngle_.train(Mat_Train_Data, Mat_ResponseAngle_Data,sampleIdx,true,32,false);

    // Load the trained GP models for the modulation ellipses
    gp_radiiX_outer.reset(new libgp::GaussianProcess((fpath + "gp_radiiX_outer").c_str()));
    gp_radiiY_outer.reset(new libgp::GaussianProcess((fpath + "gp_radiiY_outer").c_str()));
    gp_centerX_outer.reset(new libgp::GaussianProcess((fpath + "gp_centerX_outer").c_str()));
    gp_centerY_outer.reset(new libgp::GaussianProcess((fpath + "gp_centerY_outer").c_str()));
    gp_phi_cos_outer.reset(new libgp::GaussianProcess((fpath + "gp_phi_cos_outer").c_str()));
    gp_phi_sin_outer.reset(new libgp::GaussianProcess((fpath + "gp_phi_sin_outer").c_str()));

    gp_radiiX_inner.reset(new libgp::GaussianProcess((fpath + "gp_radiiX_inner").c_str()));
    gp_radiiY_inner.reset(new libgp::GaussianProcess((fpath + "gp_radiiY_inner").c_str()));
    gp_centerX_inner.reset(new libgp::GaussianProcess((fpath + "gp_centerX_inner").c_str()));
    gp_centerY_inner.reset(new libgp::GaussianProcess((fpath + "gp_centerY_inner").c_str()));
    gp_phi_cos_inner.reset(new libgp::GaussianProcess((fpath + "gp_phi_cos_inner").c_str()));
    gp_phi_sin_inner.reset(new libgp::GaussianProcess((fpath + "gp_phi_sin_inner").c_str()));
  }

  void Modulation::updateSpeedAndPosition(Eigen::Vector3d &curr_pose, Eigen::VectorXf &curr_speed,
                                          Eigen::VectorXd &curr_gripper_pose) {
    position_ = curr_pose;
    speed_ = curr_speed;
    gripper_position_ = curr_gripper_pose;
    Eigen::Matrix2f R(2, 2);

    Eigen::Isometry3d gripperPose;
    gripperPose.setIdentity();
    Eigen::Quaterniond Q =
      Eigen::Quaterniond(curr_gripper_pose(6), curr_gripper_pose(3), curr_gripper_pose(4), curr_gripper_pose(5));
    gripperPose.linear() = Q.matrix();
    double gripper_pitch;
    Eigen::Vector3d euler = Q.toRotationMatrix().eulerAngles(2, 1, 0);
    gripper_pitch = euler[1];
    if (gripper_pitch > M_PI / 2)
      gripper_pitch = M_PI - gripper_pitch;
    else if (gripper_pitch < -M_PI / 2)
      gripper_pitch = -M_PI - gripper_pitch;

    for (int k = 0; k < ellipses_.size(); k++) {
      if (ellipses_[k].getType() == "outter" || ellipses_[k].getType() == "inner") {
        // retrieve parameters for ir ellipses from gripper (x, pitch)
        Eigen::Vector3d x_Offset_gripper;
        x_Offset_gripper << -0.18, 0.0, 0.0;
        x_Offset_gripper = gripperPose.linear() * x_Offset_gripper;
        Eigen::Vector3d wrist_pose;
        wrist_pose << gripper_position_[0] + x_Offset_gripper[0], gripper_position_[1] + x_Offset_gripper[1],
          gripper_position_[2] + x_Offset_gripper[2] - 0.1;
        double x_test[] = {wrist_pose(2), gripper_pitch};

        // update speed and position of irm ellipses
        Eigen::Vector3d radial_velocity;
        Eigen::Vector3d angle_velocity;
        angle_velocity << curr_speed[3], curr_speed[4], curr_speed[5];
        std::vector<double> ellipse_speed;

        if (ellipses_[k].getType() == "inner") {
          ellipses_[k].setHeight(gp_radiiY_inner->f(x_test) + 0.05);
          ellipses_[k].setWidth(gp_radiiX_inner->f(x_test) + 0.05);
          Eigen::Vector3d xOffset_inner;
          xOffset_inner << gp_centerX_inner->f(x_test), gp_centerY_inner->f(x_test), 0.0;
          xOffset_inner = gripperPose.linear() * xOffset_inner;
          double alpha = atan2(xOffset_inner[1], xOffset_inner[0]);
          double cos_alpha_inner = gp_phi_cos_inner->f(x_test);
          double sin_alpha_inner = gp_phi_sin_inner->f(x_test);
          double alpha_inner = atan2(sin_alpha_inner, cos_alpha_inner);
          double cosangle = cos(alpha - alpha_inner);
          double sinangle = sin(alpha - alpha_inner);
          R << cosangle, -sinangle, sinangle, cosangle;
          ellipses_[k].setR(R);
          ellipses_[k].setPPoint(wrist_pose[0] + xOffset_inner[0], wrist_pose[1] + xOffset_inner[1]);
          radial_velocity = angle_velocity.cross(x_Offset_gripper);
          ellipse_speed.push_back(curr_speed[0] + radial_velocity[0]);
          ellipse_speed.push_back(curr_speed[1] + radial_velocity[1]);
        } else {
          ellipses_[k].setHeight(gp_radiiX_outer->f(x_test) + 0.0);
          ellipses_[k].setWidth(gp_radiiY_outer->f(x_test) + 0.0);
          Eigen::Vector3d xOffset_outer;
          xOffset_outer << gp_centerX_outer->f(x_test), gp_centerY_outer->f(x_test), 0.0;
          xOffset_outer = gripperPose.linear() * xOffset_outer;
          double alpha = atan2(xOffset_outer[1], xOffset_outer[0]);
          double cos_alpha_outer = gp_phi_cos_outer->f(x_test);
          double sin_alpha_outer = gp_phi_sin_outer->f(x_test);
          double alpha_outer = atan2(sin_alpha_outer, cos_alpha_outer);
          double cosangle = cos(alpha - alpha_outer);
          double sinangle = sin(alpha - alpha_outer);
          R << cosangle, -sinangle, sinangle, cosangle;
          ellipses_[k].setR(R);
          ellipses_[k].setPPoint(wrist_pose[0] + xOffset_outer[0], wrist_pose[1] + xOffset_outer[1]);
          radial_velocity = angle_velocity.cross(x_Offset_gripper);
          ellipse_speed.push_back(curr_speed[0] + radial_velocity[0]);
          ellipse_speed.push_back(curr_speed[1] + radial_velocity[1]);

          // update orientation part of positioning for irm ellipses
          int nr_neighbors = 19;
          Eigen::Vector2f pos_ell_frame;
          pos_ell_frame << position_[0] - ellipses_[k].getPPoint()[0], position_[1] - ellipses_[k].getPPoint()[1];
          pos_ell_frame = ellipses_[k].getR().transpose() * pos_ell_frame;
          float _sample[4];
          CvMat sample_beta0 = cvMat(1, 4, CV_32FC1, _sample);
          sample_beta0.data.fl[0] = (float)curr_gripper_pose(2);
          sample_beta0.data.fl[1] = (float)Q.toRotationMatrix().eulerAngles(2, 1, 0)[1];  // 0.0;
          if (sample_beta0.data.fl[1] > M_PI / 2)
            sample_beta0.data.fl[1] = M_PI - sample_beta0.data.fl[1];
          else if (sample_beta0.data.fl[1] < -M_PI / 2)
            sample_beta0.data.fl[1] = -M_PI - sample_beta0.data.fl[1];
          sample_beta0.data.fl[2] = (float)pos_ell_frame[0];
          sample_beta0.data.fl[3] = (float)pos_ell_frame[1];
          float _response[nr_neighbors];
          float _neighbors[1];

          CvMat resultMat0 = cvMat(1, 1, CV_32FC1, _neighbors);
          const CvMat *resultMat = &resultMat0;
          CvMat neighborResponses0 = cvMat(1, nr_neighbors, CV_32FC1, _response);
          const CvMat *neighborResponses = &neighborResponses0;
          // UBUNTU 14 VERSION
          // CvMat resultMat = cvMat(1,1,CV_32FC1,_neighbors);
          // CvMat neighborResponses = cvMat(1,nr_neighbors,CV_32F,_response);
          ///

          cv::Mat Mat_result = cv::cvarrToMat(resultMat);
          cv::Mat Mat_neighborResponses = cv::cvarrToMat(neighborResponses);
          const CvMat *sample_beta = &sample_beta0;
          cv::Mat M1 = cv::cvarrToMat(sample_beta);

          float result_beta0 = knnAngle_->findNearest(M1, nr_neighbors, Mat_result, Mat_neighborResponses);
          // UBUNTU 14 VERSION
          // const float **neighbors=0;
          // float result_beta0 = knnAngle_.find_nearest(&sample_beta0, nr_neighbors,&resultMat,neighbors,&neighborResponses);

          Mat_neighborResponses.convertTo(Mat_neighborResponses, CV_64FC1);
          double sum_sin = 0.0;
          double sum_cos = 0.0;
          for (int s = 0; s < nr_neighbors; s++) {
            double neighbor_i = Mat_neighborResponses.at<double>(s);
            // UBUNTU 14 VERSION
            // double neighbor_i = (double) neighborResponses.data.fl[s];

            sum_sin += sin(neighbor_i);
            sum_cos += cos(neighbor_i);
          }
          result_beta0 = atan2(sum_sin, sum_cos);

          // find aperture for legal orientation with knn regression
          float result_beta_ap = knnAperture_->findNearest(M1, nr_neighbors, Mat_result, Mat_neighborResponses);
          // UBTUNTU 14VERSION
          // float result_beta_ap = knnAperture_.find_nearest(&sample_beta0, nr_neighbors);

          ellipses_[k].setPPointAlpha(result_beta0);
          ellipses_[k].setAlphaAp(result_beta_ap);
        }
        ellipses_[k].setSpeed(ellipse_speed);
      }
    }
  }

  void Modulation::computeXiWave() {
    xi_wave_.clear();
    for (int i = 0; i < ellipses_.size(); i++) {
      Eigen::Vector2f pos_ell_frame;
      pos_ell_frame << position_[0] - ellipses_[i].getPPoint()[0], position_[1] - ellipses_[i].getPPoint()[1];
      pos_ell_frame = ellipses_[i].getR().transpose() * pos_ell_frame;
      std::vector<double> xi_wave_i = {pos_ell_frame[0], pos_ell_frame[1]};
      xi_wave_.push_back(xi_wave_i);
    }
  }

  void Modulation::computeGammaAlpha(int ellipseNr) {
    // robot base orientation
    Eigen::Matrix2f R;
    R << std::cos(position_[2]), -std::sin(position_[2]), std::sin(position_[2]), std::cos(position_[2]);
    Eigen::Vector2f or_x;
    or_x << 1.0, 0.0;
    or_x = ellipses_[ellipseNr].getR().transpose() * R * or_x;  //
    // angle between robot base orientation and connection to gripper
    double dot =
      -xi_wave_[ellipseNr][0] * or_x[0] - xi_wave_[ellipseNr][1] * or_x[1];  // dot product between [x1, y1] and [x2, y2]
    double det = -xi_wave_[ellipseNr][1] * or_x[0] +
                 xi_wave_[ellipseNr][0] * or_x[1];  // x1*y2 - y1*x2      # determinant, angle = atan2(det, dot)
    double current_alpha = atan2(det, dot);

    // base orientation in ellipseframe
    double base_angle = position_[2] + ellipses_[ellipseNr].getAlpha();  // atan2(or_x[1], or_x[0]);
    double alpha_dist = ellipses_[ellipseNr].getPPointAlpha() - base_angle;
    if (alpha_dist < -M_PI) {
      alpha_dist += 2.0 * M_PI;
    } else if (alpha_dist > M_PI)
      alpha_dist -= 2.0 * M_PI;

    double powerF = 2.0;
    gamma_alpha_ = pow((alpha_dist / ellipses_[ellipseNr].getAlphaAp() / 2.0), powerF);
    double speed_before = speed_(12);  //-gripper_yaw_speed_;

    if (gamma_alpha_ >= 0.1) {
      if (alpha_dist * speed_before >= 0.0)
        speed_before = speed_before * gamma_alpha_ * 10;
      else
        speed_before = -speed_before * gamma_alpha_ * 10;
    } else {
      // ROS_INFO("Inside  Angle Bound. Angledist to opt: %g, measured angle: %g and speed:
      // %g",alpha_dist*180.0/M_PI,position_[2]*180.0/M_PI,speed_(12));
      if (alpha_dist * speed_before < 0.0)
        speed_before = speed_before * (1.0 - gamma_alpha_);
    }
    speed_(12) = speed_before;  // + gripper_yaw_speed_;
  }

  void Modulation::computeGamma() {
    gamma_.clear();
    computeXiWave();
    int i = 0;
    for (ellipse::Ellipse ellipse : ellipses_) {
      double gamma_i = pow(pow((xi_wave_[i][0] / ellipse.getHeight()), 2 * ellipse.getP1()) +
                             pow((xi_wave_[i][1] / ellipse.getWidth()), 2 * ellipse.getP2()),
                           1.0 / ellipse.getP2());
      ellipses_[i].setInCollision(false);
      if (ellipse.getType() == "outter") {
        gamma_i = 1.0 / gamma_i;
        computeGammaAlpha(i);
      }

      real_gamma_.push_back(gamma_i);
      if (gamma_i < 1.0) {
        ellipses_[i].setInCollision(true);
        gamma_i = 1.0;
      }
      gamma_.push_back(gamma_i);

      ellipse.setGamma(gamma_i);
      i++;
    }
  }

  double Modulation::computeWeight(int k) {
    double w = 1;
    for (int i = first_ellipse_; i < ellipses_.size(); i++) {
      if (i != k) {
        w = w * ((gamma_[i] - 1) / ((gamma_[k] - 1) + (gamma_[i] - 1)));
      }
    }
    if (w != w) {
      w = 1.0;
      for (int i = first_ellipse_; i < ellipses_.size(); i++) {
        if (i != k) {
          w = w * ((real_gamma_[i] - 1) / ((real_gamma_[k] - 1) + (real_gamma_[i] - 1)));
        }
      }
    }
    if (!do_ir_modulation_ & first_ellipse_ > k)
      w = 0;
    return w;
  }

  std::vector<double> Modulation::computeEigenvalue(int k) {
    std::vector<double> lambda;
    double w = computeWeight(k);
    double collision_repulsion = -50.0;

    Eigen::Vector2f speed;
    speed << speed_(7) - ellipses_[k].getSpeed()[0], speed_(8) - ellipses_[k].getSpeed()[1];
    Eigen::Vector2f e_k1;
    e_k1 << assembleE_k(k)(0, 0), assembleE_k(k)(0, 1);
    bool passed_object = false;
    if (speed.transpose().dot(ellipses_[k].getR() * e_k1) > 0.0)
      passed_object = true;  // use this to conmtroll tail effekt, stop modulation if object already passed

    // OUTER IRM Bound
    if (ellipses_[k].getType() == "outter") {
      if (passed_object && !ellipses_[k].getInCollision())
        lambda = {1.0 - (w / pow(gamma_[k], 1.0 / ellipses_[k].getRho())), 1.0};
      else if (passed_object && ellipses_[k].getInCollision())
        lambda = {collision_repulsion, 1.0};
      else
        lambda = {1.0, 1.0};
    }
    // INNER IRM Bound
    else if (ellipses_[k].getType() == "inner") {
      lambda = {1.0 - (w / pow(gamma_[k], 1.0 / ellipses_[k].getRho())),
                1.0 + 1.0 / 500000.0 * (w / pow(gamma_[k], 1.0 / ellipses_[k].getRho()))};
      if (passed_object && !ellipses_[k].getInCollision()) {
        lambda[0] = 1.0;
        lambda[1] = 1.0;
      }
      // case of colliding with bound
      else if (ellipses_[k].getInCollision()) {
        if (passed_object)
          lambda[0] = 2 - lambda[0];  // inside bound but moving out -> accelerate
        else
          lambda[0] = collision_repulsion;  // inside bound and moving further in -> mirror and increase velocity
        lambda[1] = 1.0 + (w / pow(gamma_[k], 1.0 / ellipses_[k].getRho()));
      }
    }

    return lambda;
  }

  Eigen::MatrixXf Modulation::assembleD_k(int k) {
    Eigen::MatrixXf d_k(2, 2);
    d_k.setIdentity();
    std::vector<double> lambda = computeEigenvalue(k);

    for (int i = 0; i < 2; i++) {
      d_k(i, i) = lambda[i];
    }
    return d_k;
  }

  std::vector<double> Modulation::computeHyperplane(int k) {
    // Derivation of Gamma in ~Xi_i direction
    std::vector<double> n = {(pow(xi_wave_[k][0] / ellipses_[k].getHeight(), 2.0 * ellipses_[k].getP1() - 1)) * 2 *
                               ellipses_[k].getP1() / ellipses_[k].getHeight(),
                             (pow(xi_wave_[k][1] / ellipses_[k].getWidth(), 2.0 * ellipses_[k].getP2() - 1)) * 2 *
                               ellipses_[k].getP2() / ellipses_[k].getWidth()};

    Eigen::Vector2f n_rot;
    n_rot << n[0], n[1];
    n_rot = ellipses_[k].getR() * n_rot;
    ellipses_[k].setHyperNormal(std::vector<double>{n_rot(0), n_rot(1)});
    return n;
  };

  std::vector<std::vector<double>> Modulation::computeEBase(int k, std::vector<double> normal) {
    int d = 2;
    std::vector<std::vector<double>> base = {{0, 0}};
    for (int i = 1; i <= d - 1; i++) {
      for (int j = 1; j <= d; j++) {
        if (j == 1) {
          base[i - 1][j - 1] = -normal[i - 1];
        } else if (j == i && i != 1) {
          base[i - 1][j - 1] = normal[0];
        } else {
          base[i - 1][j - 1] = 0;
        }
      }
    }
    return base;
  };

  Eigen::MatrixXf Modulation::assembleE_k(int k) {
    Eigen::MatrixXf e_k(2, 2);
    std::vector<double> norm = computeHyperplane(k);
    std::vector<std::vector<double>> base = computeEBase(k, norm);
    for (int i = 0; i < 2; i++) {
      e_k(i, 0) = norm[i];
      if (i == 0)
        e_k(i, 1) = norm[1];
      else
        e_k(i, 1) = -norm[0];
    }
    return e_k;
  }

  void Modulation::computeModulationMatrix() {
    modulation_ << 1, 0, 0, 1;
    std::stringstream mes;
    computeGamma();
    bool out_mod = false;
    for (int k = 0; k < ellipses_.size(); k++) {
      Eigen::MatrixXf d_k = assembleD_k(k);
      Eigen::MatrixXf e_k = assembleE_k(k);
      Eigen::MatrixXf res = (ellipses_[k].getR() * e_k * d_k * e_k.inverse() * ellipses_[k].getR().transpose());
      modulation_ = (res * modulation_);
    }
  }

  Eigen::VectorXf Modulation::compModulation() {
    if (ellipses_.size() == 0)
      return speed_;
    computeModulationMatrix();
    Eigen::VectorXf d2(2);
    // find weighted relative speed with respect to obstacles
    double meanVelX = 0.0;
    double meanVelY = 0.0;
    double weightSum = 0.0;
    for (int k = 0; k < ellipses_.size(); k++) {
      double weight_k = computeWeight(k);
      // if (weight_k < 0.1)
      //   continue;
      weightSum += weight_k;
      meanVelX += weight_k * ellipses_[k].getSpeed()[0];
      meanVelY += weight_k * ellipses_[k].getSpeed()[1];
    }
    d2 << speed_[7] - meanVelX / weightSum, speed_[8] - meanVelY / weightSum;
    d2 = modulation_ * d2;
    speed_(7) = d2[0] + meanVelX / weightSum;
    speed_(8) = d2[1] + meanVelY / weightSum;

    return speed_;
  }

  void Modulation::run(Eigen::VectorXf &curr_pose, Eigen::VectorXf &curr_speed) {
    Eigen::Vector3d trans;
    trans[0] = curr_pose(7);
    trans[1] = curr_pose(8);
    // tf::Quaternion q(curr_pose(10),curr_pose(11),curr_pose(12),curr_pose(13));
    // trans[2] = tf::getYaw(q);
    Eigen::Quaterniond Q2 = Eigen::Quaterniond(curr_pose(13), curr_pose(10), curr_pose(11), curr_pose(12));
    auto euler = Q2.toRotationMatrix().eulerAngles(0, 1, 2);
    trans[2] = euler[2];
    Eigen::VectorXd curr_gripper_pose(7);
    curr_gripper_pose[0] = curr_pose(0);
    curr_gripper_pose[1] = curr_pose(1);
    curr_gripper_pose[2] = curr_pose(2);
    curr_gripper_pose[3] = curr_pose(3);
    curr_gripper_pose[4] = curr_pose(4);
    curr_gripper_pose[5] = curr_pose(5);
    curr_gripper_pose[6] = curr_pose(6);
    // Update speed and position for the irm objects
    updateSpeedAndPosition(trans, curr_speed, curr_gripper_pose);
    // compute and return modulated velocity

    compModulation();
    ;
    curr_speed(7) = speed_(7);
    curr_speed(8) = speed_(8);
    curr_speed(12) = speed_(12);
  }

  double computeL2Norm(std::vector<double> v) {
    double res = 0;
    for (double entry : v) {
      res += entry * entry;
    }
    return sqrt(res);
  }

  visualization_msgs::MarkerArray Modulation::getEllipsesVisMarker(Eigen::VectorXf &curr_pose, Eigen::VectorXf &curr_speed) {
    Eigen::Vector3d current_base_pose;
    current_base_pose[0] = curr_pose(7);
    current_base_pose[1] = curr_pose(8);
    current_base_pose[2] = curr_pose(9);
    Eigen::VectorXd curr_gripper_pose(7);
    curr_gripper_pose[0] = curr_pose(0);
    curr_gripper_pose[1] = curr_pose(1);
    curr_gripper_pose[2] = curr_pose(2);
    curr_gripper_pose[3] = curr_pose(3);
    curr_gripper_pose[4] = curr_pose(4);
    curr_gripper_pose[5] = curr_pose(5);
    curr_gripper_pose[6] = curr_pose(6);
    // std::vector<ellipse_extraction::Ellipse> ellipses = manager_->getEllipses(current_base_pose,
    // curr_speed,curr_gripper_pose,dt); std::vector<ellipse::Ellipse> ellipses = modulation_.getEllipses(current_base_pose,
    // curr_speed,curr_gripper_pose);
    visualization_msgs::MarkerArray ma;
    int index = 0;
    for (ellipse::Ellipse ellipse : ellipses_) {
      visualization_msgs::Marker line_strip;
      line_strip.type = visualization_msgs::Marker::LINE_STRIP;
      line_strip.action = visualization_msgs::Marker::ADD;
      line_strip.scale.x = 0.01;
      line_strip.header.frame_id = "odom_combined";
      line_strip.id = index + 330000;
      line_strip.ns = ellipse.getType();
      line_strip.color.a = 1.0;
      line_strip.lifetime = ros::Duration(0.0);

      index++;
      geometry_msgs::Point p;
      int max_i = 100;
      for (int i = 0; i < max_i; i++) {
        if (ellipse.getInCollision())
          line_strip.color.r = 1.0;
        else if (ellipse.getType() == "obstacle") {
          line_strip.color.r = 0.0;
          line_strip.color.g = 0.0;
          line_strip.color.b = 0.0;
        } else
          line_strip.color.b = 1.0;

        if (ellipse.getType() == "outter") {
          line_strip.lifetime = ros::Duration(0.0);
          line_strip.id = 2000;
        } else if (ellipse.getType() == "inner") {
          line_strip.lifetime = ros::Duration(0.0);
          line_strip.id = 2001;
        }

        double theta = -M_PI + 2.0 * M_PI * ((double)i / (double)max_i);
        // x_obs(1,:,n) = a(1,:).*cos(theta);
        // x_obs(2,:,n) = a(2,:).*sign(theta).*(1 - cos(theta).^(2.*p(1,:))).^(1./(2.*p(2,:)));
        // x_obs(:,:,n) = R*x_obs(:,:,n) + repmat(obs{n}.x0,1,np);

        double sign;
        if (i < max_i / 2)
          sign = -1.0;
        else
          sign = 1.0;

        Eigen::Vector2f margin_point;
        margin_point << ellipse.getHeight() * cos(theta),
          ellipse.getWidth() * sign * pow((1.0 - pow(cos(theta), 2.0 * ellipse.getP1())), 1.0 / (2.0 * ellipse.getP2()));

        Eigen::Vector2f e_center(ellipse.getPPoint()[0], ellipse.getPPoint()[1]);
        margin_point = ellipse.getR() * margin_point + e_center;

        p.x = margin_point(0);
        p.y = margin_point(1);
        p.z = current_base_pose[2];  // 0;
        line_strip.points.push_back(p);
      }
      line_strip.points.push_back(line_strip.points[0]);
      ma.markers.push_back(line_strip);

      if (ellipse.getType() == "outter") {
        // Proposed Orientation from IR ellipse
        visualization_msgs::Marker meanOrientation;
        meanOrientation.type = visualization_msgs::Marker::ARROW;
        meanOrientation.ns = "IR mean orientation";
        meanOrientation.header.frame_id = "odom_combined";
        meanOrientation.scale.x = 0.1;
        meanOrientation.scale.y = 0.01;
        meanOrientation.scale.z = 0.01;
        meanOrientation.color.b = 1.0;
        meanOrientation.color.a = 1.0;
        meanOrientation.id = 1;
        meanOrientation.pose.position.x = current_base_pose[0];
        meanOrientation.pose.position.y = current_base_pose[1];
        meanOrientation.pose.position.z = current_base_pose[2];
        // set orientation
        double angle0 = ellipse.getPPointAlpha() - ellipse.getAlpha();
        meanOrientation.pose.orientation.x = 0.0;
        meanOrientation.pose.orientation.y = 0.0;
        meanOrientation.pose.orientation.z = sin(angle0 / 2);
        meanOrientation.pose.orientation.w = cos(angle0 / 2);
        ma.markers.push_back(meanOrientation);

        // Proposed range from IR ellipse / KNN regression
        visualization_msgs::Marker alpha_aperture;
        alpha_aperture.type = visualization_msgs::Marker::LINE_STRIP;
        alpha_aperture.action = visualization_msgs::Marker::ADD;
        alpha_aperture.scale.x = 0.015;
        alpha_aperture.header.frame_id = "odom_combined";
        alpha_aperture.id = 184375;
        alpha_aperture.ns = "alpha_aperture";
        tf::Quaternion baseQ(curr_pose(10), curr_pose(11), curr_pose(12), curr_pose(13));
        double base_angle = tf::getYaw(baseQ) + ellipse.getAlpha();
        double alpha_dist = ellipse.getPPointAlpha() - base_angle;
        if (alpha_dist < -M_PI) {
          alpha_dist += 2.0 * M_PI;
        } else if (alpha_dist > M_PI)
          alpha_dist -= 2.0 * M_PI;
        alpha_dist = std::abs(alpha_dist);
        if (alpha_dist > ellipse.getAlphaAp()) {
          alpha_aperture.color.r = 1.0;
        } else {
          alpha_aperture.color.b = 0.7;
          alpha_aperture.color.r = 1.0;
          alpha_aperture.color.g = 0.1;
        }
        alpha_aperture.color.a = 1.0;
        alpha_aperture.lifetime = ros::Duration(0.0);
        geometry_msgs::Point point;
        int max_points = 20;
        point.x = current_base_pose[0];
        point.y = current_base_pose[1];
        point.z = current_base_pose[2];
        alpha_aperture.points.push_back(point);
        for (int i = -max_points; i < max_points + 1; i++) {
          point.x = current_base_pose[0] + 0.15 * cos(angle0 + i * ellipse.getAlphaAp() / max_points);
          point.y = current_base_pose[1] + 0.15 * sin(angle0 + i * ellipse.getAlphaAp() / max_points);
          alpha_aperture.points.push_back(point);
        }
        point.x = current_base_pose[0];
        point.y = current_base_pose[1];
        point.z = current_base_pose[2];
        alpha_aperture.points.push_back(point);
        ma.markers.push_back(alpha_aperture);
      }
    }
    int traj_lenght = 1;

    visualization_msgs::Marker base_marker;
    base_marker.type = visualization_msgs::Marker::CUBE;
    base_marker.action = visualization_msgs::Marker::ADD;
    base_marker.scale.x = 0.03;
    base_marker.scale.y = 0.03;
    base_marker.scale.z = 0.03;
    base_marker.header.frame_id = "odom_combined";
    base_marker.id = traj_lenght;
    base_marker.ns = "base";
    base_marker.color.g = 1.0;
    base_marker.color.a = 1.0;
    base_marker.pose.position.x = current_base_pose[0];
    base_marker.pose.position.y = current_base_pose[1];
    base_marker.pose.position.z = current_base_pose[2];
    base_marker.pose.orientation.w = 1.0;
    ma.markers.push_back(base_marker);

    visualization_msgs::Marker gripper_marker;
    gripper_marker.type = visualization_msgs::Marker::SPHERE;
    gripper_marker.action = visualization_msgs::Marker::ADD;
    gripper_marker.scale.x = 0.03;
    gripper_marker.scale.y = 0.03;
    gripper_marker.scale.z = 0.03;
    gripper_marker.header.frame_id = "odom_combined";
    gripper_marker.id = traj_lenght;
    gripper_marker.ns = "gripper";
    gripper_marker.color.b = 1.0;
    gripper_marker.color.r = 1.0;
    gripper_marker.color.a = 1.0;
    gripper_marker.pose.position.x = curr_gripper_pose[0];
    gripper_marker.pose.position.y = curr_gripper_pose[1];
    gripper_marker.pose.position.z = curr_gripper_pose[2];
    gripper_marker.pose.orientation.w = 1.0;
    ma.markers.push_back(gripper_marker);

    // Marker for the Frame
    std_msgs::ColorRGBA red;
    red.r = 1.0;
    red.a = 1.0;
    std_msgs::ColorRGBA blue;
    blue.b = 1.0;
    blue.a = 1.0;
    std_msgs::ColorRGBA green;
    green.g = 1.0;
    green.a = 1.0;
    // Base
    visualization_msgs::Marker mark;
    mark.header.frame_id = "/odom_combined";
    mark.ns = "Base_Frame";
    mark.id = 3;
    mark.type = visualization_msgs::Marker::LINE_LIST;
    mark.action = visualization_msgs::Marker::ADD;
    mark.pose.orientation.w = 1.0;
    mark.scale.x = 0.015;
    mark.color = blue;
    blue.a = 1.0;
    const double poseCoordsLength = 0.15;
    geometry_msgs::Point ptOrig;
    Eigen::Isometry3d basePose;
    basePose.translation() = Eigen::Vector3d(curr_pose(7), curr_pose(8), curr_pose(9));
    Eigen::Quaterniond Q = Eigen::Quaterniond(curr_pose(13), 0.0, 0.0, curr_pose(12));
    basePose.linear() = Q.matrix();
    ptOrig.x = basePose.translation().x();
    ptOrig.y = basePose.translation().y();
    ptOrig.z = basePose.translation().z();
    Eigen::Vector3d xCoord(poseCoordsLength, 0.0, 0.0);
    geometry_msgs::Point ptX;
    tf::pointEigenToMsg(basePose * xCoord, ptX);

    mark.points.push_back(ptOrig);
    mark.points.push_back(ptX);
    mark.colors.push_back(red);
    mark.colors.push_back(red);
    Eigen::Vector3d yCoord(0.0, poseCoordsLength, 0.0);
    geometry_msgs::Point ptY;
    tf::pointEigenToMsg(basePose * yCoord, ptY);

    mark.points.push_back(ptOrig);
    mark.points.push_back(ptY);
    mark.colors.push_back(green);
    mark.colors.push_back(green);
    Eigen::Vector3d zCoord(0.0, 0.0, poseCoordsLength);
    geometry_msgs::Point ptZ;
    tf::pointEigenToMsg(basePose * zCoord, ptZ);

    mark.points.push_back(ptOrig);
    mark.points.push_back(ptZ);
    mark.colors.push_back(blue);
    mark.colors.push_back(blue);
    // Gripper
    visualization_msgs::Marker mark_gripper_frame;
    mark_gripper_frame.header.frame_id = "/odom_combined";
    mark_gripper_frame.ns = "Gripper_Frame";
    mark_gripper_frame.id = 4;
    mark_gripper_frame.type = visualization_msgs::Marker::LINE_LIST;
    mark_gripper_frame.action = visualization_msgs::Marker::ADD;
    mark_gripper_frame.pose.orientation.w = 1.0;
    mark_gripper_frame.scale.x = 0.015;
    mark_gripper_frame.color = blue;
    geometry_msgs::Point ptOrig2;
    Eigen::Isometry3d gripperPose;
    gripperPose.translation() = Eigen::Vector3d(curr_pose(0), curr_pose(1), curr_pose(2));
    Eigen::Quaterniond Q2 = Eigen::Quaterniond(curr_pose(6), curr_pose(3), curr_pose(4), curr_pose(5));
    gripperPose.linear() = Q2.matrix();
    ptOrig2.x = gripperPose.translation().x();
    ptOrig2.y = gripperPose.translation().y();
    ptOrig2.z = gripperPose.translation().z();

    tf::pointEigenToMsg(gripperPose * xCoord, ptX);
    mark_gripper_frame.points.push_back(ptOrig2);
    mark_gripper_frame.points.push_back(ptX);
    mark_gripper_frame.colors.push_back(red);
    mark_gripper_frame.colors.push_back(red);

    tf::pointEigenToMsg(gripperPose * yCoord, ptY);
    mark_gripper_frame.points.push_back(ptOrig2);
    mark_gripper_frame.points.push_back(ptY);
    mark_gripper_frame.colors.push_back(green);
    mark_gripper_frame.colors.push_back(green);

    tf::pointEigenToMsg(gripperPose * zCoord, ptZ);
    mark_gripper_frame.points.push_back(ptOrig2);
    mark_gripper_frame.points.push_back(ptZ);
    mark_gripper_frame.colors.push_back(blue);
    mark_gripper_frame.colors.push_back(blue);
    ma.markers.push_back(mark);
    ma.markers.push_back(mark_gripper_frame);

    // return the visualization marker
    return ma;
  }

}  // namespace modulation_ellipses
