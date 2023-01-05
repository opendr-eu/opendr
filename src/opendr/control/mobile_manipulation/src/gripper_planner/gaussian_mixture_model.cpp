// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gripper_planner/gaussian_mixture_model.hpp>

GaussianMixtureModel::GaussianMixtureModel(double max_speed_gripper_rot, double max_speed_base_rot) :
  _max_speed_gripper_rot{max_speed_gripper_rot},
  _max_speed_base_rot{max_speed_base_rot},
  _nr_modes{0},
  _kP{1},
  _kV{1},
  // duration in real time per time step in the fitted model
  _motion_duration{30} {
}

GaussianMixtureModel::~GaussianMixtureModel() {
}

void GaussianMixtureModel::adaptModel(tf::Transform obj_origin_goal, tf::Vector3 gmm_base_offset) {
  // Matrix for translation to new start
  tf::Transform attractor_transformer = obj_origin_goal * _related_object_pose.inverse();

  // original version. T still needed to adapt sigma
  tf::Transform T1;
  T1.setIdentity();
  T1.setOrigin(obj_origin_goal.getOrigin() - _related_object_pose.getOrigin());
  // Rotation
  tf::Transform GR;
  GR.setIdentity();
  GR.setRotation(obj_origin_goal.getRotation());
  // if grasp object
  tf::Transform MGR;
  MGR.setIdentity();
  MGR.setRotation(_related_object_pose.getRotation());
  tf::Transform T;
  T = GR * MGR.inverse();
  T.setOrigin(-(T * obj_origin_goal.getOrigin()) + obj_origin_goal.getOrigin());

  // loop over gaussians
  _MuEigen.clear();
  for (int i = 0; i < _nr_modes; i++) {
    tf::Transform tf_Mu_gr_i;
    tf_Mu_gr_i.setOrigin(tf::Vector3(_MuEigenBck[i](1), _MuEigenBck[i](2), _MuEigenBck[i](3)));
    tf_Mu_gr_i.setRotation(tf::Quaternion(_MuEigenBck[i](4), _MuEigenBck[i](5), _MuEigenBck[i](6), _MuEigenBck[i](7)));
    tf::Transform tf_Mu_base_i;
    tf_Mu_base_i.setOrigin(tf::Vector3(_MuEigenBck[i](8), _MuEigenBck[i](9), _MuEigenBck[i](10)));
    tf_Mu_base_i.setRotation(tf::Quaternion(_MuEigenBck[i](11), _MuEigenBck[i](12), _MuEigenBck[i](13), _MuEigenBck[i](14)));

    tf_Mu_gr_i = attractor_transformer * tf_Mu_gr_i;
    tf_Mu_base_i = attractor_transformer * tf_Mu_base_i;

    // check if base is in feasible height:
    if (tf_Mu_base_i.getOrigin().z() > 1.2)
      tf_Mu_base_i.setOrigin(tf::Vector3(tf_Mu_base_i.getOrigin().x(), tf_Mu_base_i.getOrigin().y(), 1.2));
    else if (tf_Mu_base_i.getOrigin().z() < 0.7)
      tf_Mu_base_i.setOrigin(tf::Vector3(tf_Mu_base_i.getOrigin().x(), tf_Mu_base_i.getOrigin().y(), 0.7));
    // make sure base model is in upright orientation
    tf_Mu_base_i.setRotation(
      tf::Quaternion(0.0, 0.0,
                     tf_Mu_base_i.getRotation().z() / sqrt((tf_Mu_base_i.getRotation().z() * tf_Mu_base_i.getRotation().z()) +
                                                           (tf_Mu_base_i.getRotation().w() * tf_Mu_base_i.getRotation().w())),
                     tf_Mu_base_i.getRotation().w() / sqrt((tf_Mu_base_i.getRotation().z() * tf_Mu_base_i.getRotation().z()) +
                                                           (tf_Mu_base_i.getRotation().w() * tf_Mu_base_i.getRotation().w()))));

    // base goal was not necessarily learned for this robot. So use an offset to ensure robot moves close enough to the final
    // goal
    tf_Mu_base_i = utils::tipToGripperGoal(tf_Mu_base_i, -gmm_base_offset, tf::Quaternion(0, 0, 0, 1));

    // set _MuEigen
    double time_i = _MuEigenBck[i](0);
    Eigen::VectorXf Mu_i_eigen(15);
    Mu_i_eigen << time_i, tf_Mu_gr_i.getOrigin().x(), tf_Mu_gr_i.getOrigin().y(), tf_Mu_gr_i.getOrigin().z(),
      tf_Mu_gr_i.getRotation().x(), tf_Mu_gr_i.getRotation().y(), tf_Mu_gr_i.getRotation().z(), tf_Mu_gr_i.getRotation().w(),
      tf_Mu_base_i.getOrigin().x(), tf_Mu_base_i.getOrigin().y(), tf_Mu_base_i.getOrigin().z(), tf_Mu_base_i.getRotation().x(),
      tf_Mu_base_i.getRotation().y(), tf_Mu_base_i.getRotation().z(), tf_Mu_base_i.getRotation().w();

    // keep model height
    Mu_i_eigen(10) = 1.0;  //_MuEigenBck[i](10);
    _MuEigen.push_back(Mu_i_eigen);

    // transform Sigma
    Eigen::MatrixXf TS(4, 4);
    TS.setIdentity();
    Eigen::Matrix3d T_h(3, 3);
    tf::matrixTFToEigen(T.getBasis(), T_h);
    TS.block(1, 1, 3, 3) << T_h(0, 0), T_h(0, 1), T_h(0, 2), T_h(1, 0), T_h(1, 1), T_h(1, 2), T_h(2, 0), T_h(2, 1), T_h(2, 2);
    Eigen::MatrixXf Sigma_gripper_t(4, 4);
    Sigma_gripper_t = _Sigma[i].block(0, 0, 4, 4);
    Sigma_gripper_t = TS * Sigma_gripper_t * TS;
    _Sigma[i].block(0, 0, 4, 4) = Sigma_gripper_t;

    Eigen::MatrixXf Sigma_base_t(4, 4);
    Sigma_base_t(0, 0) = _Sigma[i](0, 0);
    Sigma_base_t.block(1, 0, 3, 1) = _Sigma[i].block(7, 0, 3, 1);
    Sigma_base_t.block(0, 1, 1, 3) = _Sigma[i].block(0, 7, 1, 3);
    Sigma_base_t.block(1, 1, 3, 3) = _Sigma[i].block(7, 7, 3, 3);
    Sigma_base_t = TS * Sigma_base_t * TS;
    _Sigma[i](0, 0) = Sigma_base_t(0, 0);
    _Sigma[i].block(7, 0, 3, 1) = Sigma_base_t.block(1, 0, 3, 1);
    _Sigma[i].block(0, 7, 1, 3) = Sigma_base_t.block(0, 1, 1, 3);
    _Sigma[i].block(7, 7, 3, 3) = Sigma_base_t.block(1, 1, 3, 3);
  }

  // _MuEigenBck = _MuEigen;
}

double GaussianMixtureModel::gaussPDF(double &current_time, int mode_nr) {
  double t_m = current_time - _MuEigen[mode_nr](0);
  double prob = t_m * t_m / _Sigma[mode_nr](0, 0);
  prob = exp(-0.5 * prob) / sqrt(2.0 * 3.141592 * _Sigma[mode_nr](0, 0));
  return prob;
}

template<typename T1, typename T2> T1 GaussianMixtureModel::extract(const T2 &full, const T1 &ind) {
  int num_indices = ind.innerSize();
  T1 target(num_indices);
  for (int i = 0; i < num_indices; i++) {
    target[i] = full[ind[i]];
  }
  return target;
}

void GaussianMixtureModel::integrateModel(double current_time, double dt_real, Eigen::VectorXf *current_pose,
                                          Eigen::VectorXf *current_speed, const double &min_velocity,
                                          const double &max_velocity, bool do_update) {
  // ROS_INFO("gmm time orig: %f, dt_real: %f", current_time, dt_real);
  double current_time_gmm = current_time / _motion_duration;
  double dt_gmm = dt_real / _motion_duration;

  Eigen::VectorXf ind_vec_out(14);
  ind_vec_out << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0;
  // activation weights
  std::vector<double> H;

  // part of the 'hack' below
  current_time_gmm -= gmm_time_offset_;

  // ROS_INFO("current_time_gmm %g",current_time_gmm);
  if (current_time_gmm > _MuEigen[_nr_modes - 1](0)) {
    current_time_gmm = _MuEigen[_nr_modes - 1](0);
  }
  // ROS_INFO("current_time_gmm: %f, dt_gmm: %f", current_time_gmm, dt_gmm);

  double sumH = 0.0;
  for (int i = 0; i < _nr_modes; i++) {
    double hi = _Priors[i] * gaussPDF(current_time_gmm, i);
    H.push_back(hi);
    sumH += hi;
  }

  tf::Quaternion current_gripper_q((*current_pose)(3), (*current_pose)(4), (*current_pose)(5), (*current_pose)(6));
  tf::Quaternion current_base_q(0.0, 0.0, (*current_pose)(12), (*current_pose)(13));

  // acceleration
  Eigen::VectorXf currF(14);
  currF << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  int highest_i = 0, secHighest_i = 0;
  double highest_h = 0.0, secHighest_h = 0.0;
  for (int i = 0; i < _nr_modes; i++) {
    Eigen::VectorXf FTmp(14);
    Eigen::MatrixXf Sigma_out_in(14, 1);
    Sigma_out_in = _Sigma[i].block(1, 0, 14, 1);
    FTmp = extract(_MuEigen[i], ind_vec_out) + Sigma_out_in * (1.0 / _Sigma[i](0, 0)) * (current_time_gmm - _MuEigen[i](0));
    currF += FTmp * H[i] / sumH;
    // for Rotation part
    if (H[i] > highest_h) {
      secHighest_i = highest_i;
      secHighest_h = highest_h;
      highest_i = i;
      highest_h = H[i];
    } else if (H[i] > secHighest_h) {
      secHighest_i = i;
      secHighest_h = H[i];
    }
  }
  // Hack for stopping time if starting to far away from demonstrations
  Eigen::Vector3f Mu3;
  Mu3 << _MuEigen[highest_i](1), _MuEigen[highest_i](2), _MuEigen[highest_i](3);
  Eigen::Vector3f Pos3;
  Pos3 << (*current_pose)(0), (*current_pose)(1), (*current_pose)(2);
  if (do_update && ((Mu3 - Pos3).norm() > 2.5)) {
    gmm_time_offset_ += dt_gmm;
  }
  // END Hack

  // adapt model for base motion acording to current gripper pose
  // adaptBaseModel(current_pose);
  double h_ratio = secHighest_h / (secHighest_h + highest_h);
  if (current_time_gmm >= _MuEigen[_nr_modes - 1](0)) {
    h_ratio = 0.0;
    currF = extract(_MuEigen[_nr_modes - 1], ind_vec_out);
  }
  tf::Quaternion desired_gripper_q =
    tf::Quaternion(_MuEigen[highest_i](4), _MuEigen[highest_i](5), _MuEigen[highest_i](6), _MuEigen[highest_i](7))
      .slerp(tf::Quaternion(_MuEigen[secHighest_i](4), _MuEigen[secHighest_i](5), _MuEigen[secHighest_i](6),
                            _MuEigen[secHighest_i](7)),
             h_ratio);

  tf::Quaternion relative = desired_gripper_q * current_gripper_q.inverse();
  double angle = relative.getAngle();
  if (angle > M_PI) {
    angle -= 2 * M_PI;
  }
  tf::Vector3 v_rot_gripper = relative.getAxis() * angle;
  tf::Vector3 a_rot_gripper =
    _kP * v_rot_gripper - _kV * tf::Vector3((*current_speed)(3), (*current_speed)(4), (*current_speed)(5));

  tf::Quaternion current_desired_base_q =
    tf::Quaternion(0.0, 0.0, _MuEigen[highest_i](13), _MuEigen[highest_i](14))
      .slerp(tf::Quaternion(0.0, 0.0, _MuEigen[secHighest_i](13), _MuEigen[secHighest_i](14)), h_ratio);
  tf::Quaternion relative_base = current_desired_base_q * current_base_q.inverse();
  // tf::Vector3 v_rot_base =	relative_base.getAxis () * relative_base.getAngle();
  double base_angle = tf::getYaw(relative_base);  // relative_base.getAngle();
  if (base_angle > M_PI) {
    base_angle -= 2 * M_PI;
  }
  double a_rot_base = _kP * base_angle - _kV * (*current_speed)(12);

  Eigen::VectorXf currAcc(14);
  currAcc = _kP * (currF - *current_pose) - _kV * (*current_speed);
  currAcc(3) = a_rot_gripper.x();
  currAcc(4) = a_rot_gripper.y();
  currAcc(5) = a_rot_gripper.z();
  currAcc(6) = 0;
  currAcc(10) = 0;
  currAcc(11) = 0;
  currAcc(12) = a_rot_base;
  currAcc(13) = 0;
  // update velocity
  *current_speed = *current_speed + 1 * currAcc;

  // limit gripper speed to max_speed
  tf::Vector3 gripper_vel(current_speed->coeffRef(0), current_speed->coeffRef(1), current_speed->coeffRef(2));
  double unscaled_gripper_z = current_speed->coeffRef(2);
  gripper_vel = utils::normScaleVel(gripper_vel, min_velocity * dt_real, max_velocity * dt_real);
  current_speed->coeffRef(0) = gripper_vel.x();
  current_speed->coeffRef(1) = gripper_vel.y();
  current_speed->coeffRef(2) = gripper_vel.z();

  // limit rot speed gripper
  tf::Vector3 gripper_rot_vel(current_speed->coeffRef(3), current_speed->coeffRef(4), current_speed->coeffRef(5));
  gripper_rot_vel = utils::normScaleVel(gripper_rot_vel, 0.0, _max_speed_gripper_rot * dt_real);
  current_speed->coeffRef(3) = gripper_rot_vel.x();
  current_speed->coeffRef(4) = gripper_rot_vel.y();
  current_speed->coeffRef(5) = gripper_rot_vel.z();

  // limit robotbase speed
  // 'hack' to ensure that if starting far away the base won't out-run the gripper (due to z always being 0 in the
  // norm-scaling): scale with the same z as the gripper, then set z to zero
  tf::Vector3 base_vel(current_speed->coeffRef(7), current_speed->coeffRef(8), unscaled_gripper_z);
  base_vel = utils::normScaleVel(base_vel, min_velocity * dt_real, max_velocity * dt_real);
  base_vel.setZ(0.0);
  current_speed->coeffRef(7) = base_vel.x();
  current_speed->coeffRef(8) = base_vel.y();

  // limit robotbase rot speed
  double lim = _max_speed_base_rot * dt_real;
  current_speed->coeffRef(12) = utils::clampDouble(current_speed->coeffRef(12), -lim, lim);

  // update position
  *current_pose = *current_pose + *current_speed;  //+ currAcc*(dt*dt*0.5);
  // restrict torso height to bounds
  if (current_pose->coeffRef(9) < 0.746)
    (*current_pose)(9) = 0.746;
  else if (current_pose->coeffRef(9) > 1.06)
    (*current_pose)(9) = 1.06;
  // update rotation
  tf::Vector3 gripper_rot_speed((*current_speed)(3), (*current_speed)(4), (*current_speed)(5));
  double alpha_N = gripper_rot_speed.length();
  tf::Quaternion Q2;
  if (alpha_N != 0.0) {
    tf::Quaternion Q(gripper_rot_speed.normalized(), alpha_N);
    Q2 = Q * current_gripper_q;
  } else {
    Q2 = current_gripper_q;
  }
  Q2.normalize();

  (*current_pose)(3) = Q2.x();  // desired_gripper_q.x();//
  (*current_pose)(4) = Q2.y();  // desired_gripper_q.y();//
  (*current_pose)(5) = Q2.z();  // desired_gripper_q.z();//
  (*current_pose)(6) = Q2.w();  // desired_gripper_q.w();//

  tf::Quaternion Q3(tf::Vector3(0.0, 0.0, 1.0), (*current_speed)(12));
  tf::Quaternion Q4 = Q3 * current_base_q;
  Q4.normalize();
  (*current_pose)(10) = 0.0;
  (*current_pose)(11) = 0.0;
  (*current_pose)(12) = Q4.z();  // current_desired_base_q.z();//
  (*current_pose)(13) = Q4.w();  // current_desired_base_q.w();//
}

template<typename T> bool GaussianMixtureModel::parseVector(std::ifstream &is, std::vector<T> &pts, const std::string &name) {
  std::string line;
  getline(is, line);
  if (!is.good()) {
    ROS_ERROR("getline error for %s pts", name.c_str());
    return false;
  }

  std::stringstream ss(line);
  std::string nameStr;
  ss >> nameStr;
  nameStr = utils::trim(nameStr);
  if (utils::startsWith(nameStr, "\"")) {
    nameStr = nameStr.substr(1);
    if (utils::endsWith(nameStr, "\"")) {
      nameStr = nameStr.substr(0, nameStr.size() - 1);
    } else {
      // keep first word, skip rest until "
      while (ss.good()) {
        std::string discard;
        ss >> discard;
        discard = utils::trim(discard);
        if (utils::endsWith(discard, "\""))
          break;
      }
    }
  }
  if (!utils::startsWith(nameStr, name)) {
    ROS_ERROR("parseVector name mismatch: %s - %s", name.c_str(), nameStr.c_str());
    return false;
  }
  while (ss.good()) {
    T data(0);
    if (ss >> data)
      pts.push_back(data);
  }
  return true;
}

bool GaussianMixtureModel::loadFromFile(std::string &filename) {
  // ROS_INFO("Loading GMM from file...");
  std::ifstream is(filename.c_str());
  if (!is.good()) {
    ROS_INFO("GMM loading Error");
    return false;
  }

  std::string line;
  // skip empty lines until we're at a good one
  do {
    getline(is, line);
    if (!is.good()) {
      ROS_ERROR("Data error");
      return false;
    }
    line = utils::trim(line);
  } while (line.empty());

  if (!utils::startsWith(line, "\"")) {
    ROS_ERROR("Name start error in %s", line.c_str());
    return false;
  }
  if (!utils::endsWith(line, "\"")) {
    ROS_ERROR("Name end error in %s", line.c_str());
    return false;
  }
  std::string name = line.substr(1, line.size() - 2);
  // ROS_INFO("Data for  %s - parsing data", name.c_str());

  // Read in Priors
  _Priors.clear();
  std::vector<double> Priors;
  if (!parseVector(is, Priors, "p"))
    return false;
  _Priors = Priors;
  _nr_modes = Priors.size();
  // Read in Mu
  line.clear();
  do {
    getline(is, line);
    if (!is.good()) {
      ROS_ERROR("Traj name error");
      return false;
    }
    line = utils::trim(line);
  } while (line.empty());

  if (!utils::startsWith(line, "\"")) {
    ROS_ERROR("Traj name start error in %s", line.c_str());
    return false;
  }
  if (!utils::endsWith(line, "\"")) {
    ROS_ERROR("Traj name end error in %s", line.c_str());
    return false;
  }
  name = line.substr(1, line.size() - 2);
  // ROS_INFO("Data for  %s - parsing data", name.c_str());

  std::vector<double> Mu_t;
  if (!parseVector(is, Mu_t, "t"))
    return false;
  std::vector<double> Mu_x;
  if (!parseVector(is, Mu_x, "x"))
    return false;
  std::vector<double> Mu_y;
  if (!parseVector(is, Mu_y, "y"))
    return false;
  if (!(Mu_x.size() == Mu_y.size())) {
    ROS_ERROR("Size mismatch for yPts");
    return false;
  }
  std::vector<double> Mu_z;
  if (!parseVector(is, Mu_z, "z"))
    return false;
  if (!(Mu_x.size() == Mu_z.size())) {
    ROS_ERROR("Size mismatch for zPts");
    return false;
  }
  std::vector<double> Mu_qx;
  if (!parseVector(is, Mu_qx, "qx"))
    return false;
  if (!(Mu_x.size() == Mu_qx.size())) {
    ROS_ERROR("Size mismatch for qxPts");
    return false;
  }
  std::vector<double> Mu_qy;
  if (!parseVector(is, Mu_qy, "qy"))
    return false;
  if (!(Mu_x.size() == Mu_qy.size())) {
    ROS_ERROR("Size mismatch for qyPts");
    return false;
  }
  std::vector<double> Mu_qz;
  if (!parseVector(is, Mu_qz, "qz"))
    return false;
  if (!(Mu_x.size() == Mu_qz.size())) {
    ROS_ERROR("Size mismatch for qzPts");
    return false;
  }
  std::vector<double> Mu_qw;
  if (!parseVector(is, Mu_qw, "qw"))
    return false;
  if (!(Mu_x.size() == Mu_qw.size())) {
    ROS_ERROR("Size mismatch for qwPts");
    return false;
  }
  std::vector<double> Mub_x;
  if (!parseVector(is, Mub_x, "x"))
    return false;
  std::vector<double> Mub_y;
  if (!parseVector(is, Mub_y, "y"))
    return false;
  if (!(Mub_x.size() == Mub_y.size())) {
    ROS_ERROR("Size mismatch for yPts");
    return false;
  }
  std::vector<double> Mub_z;
  if (!parseVector(is, Mub_z, "z"))
    return false;
  if (!(Mub_x.size() == Mub_z.size())) {
    ROS_ERROR("Size mismatch for zPts");
    return false;
  }
  std::vector<double> Mub_qx;
  if (!parseVector(is, Mub_qx, "qx"))
    return false;
  if (!(Mub_x.size() == Mub_qx.size())) {
    ROS_ERROR("Size mismatch for qxPts");
    return false;
  }
  std::vector<double> Mub_qy;
  if (!parseVector(is, Mub_qy, "qy"))
    return false;
  if (!(Mub_x.size() == Mub_qy.size())) {
    ROS_ERROR("Size mismatch for qyPts");
    return false;
  }
  std::vector<double> Mub_qz;
  if (!parseVector(is, Mub_qz, "qz"))
    return false;
  if (!(Mub_x.size() == Mub_qz.size())) {
    ROS_ERROR("Size mismatch for qzPts");
    return false;
  }
  std::vector<double> Mub_qw;
  if (!parseVector(is, Mub_qw, "qw"))
    return false;
  if (!(Mub_x.size() == Mub_qw.size())) {
    ROS_ERROR("Size mismatch for qwPts");
    return false;
  }

  _Mu.clear();
  _MuEigen.clear();
  for (int i = 0; i < _nr_modes; i++) {
    std::vector<double> Mu_i;
    Mu_i.push_back(Mu_t[i]);
    Mu_i.push_back(Mu_x[i]);
    Mu_i.push_back(Mu_y[i]);
    Mu_i.push_back(Mu_z[i]);
    Mu_i.push_back(Mu_qx[i]);
    Mu_i.push_back(Mu_qy[i]);
    Mu_i.push_back(Mu_qz[i]);
    Mu_i.push_back(Mu_qw[i]);
    Mu_i.push_back(Mub_x[i]);
    Mu_i.push_back(Mub_y[i]);
    Mu_i.push_back(Mub_z[i]);
    Mu_i.push_back(Mub_qx[i]);
    Mu_i.push_back(Mub_qy[i]);
    Mu_i.push_back(Mub_qz[i]);
    Mu_i.push_back(Mub_qw[i]);

    _Mu.push_back(Mu_i);
    Eigen::VectorXf Mu_i_eigen(15);
    Mu_i_eigen << Mu_i[0], Mu_i[1], Mu_i[2], Mu_i[3], Mu_i[4], Mu_i[5], Mu_i[6], Mu_i[7], Mu_i[8], Mu_i[9], Mu_i[10], Mu_i[11],
      Mu_i[12], Mu_i[13], Mu_i[14];
    _MuEigen.push_back(Mu_i_eigen);
  }
  _MuEigenBck = _MuEigen;
  // Read in Sigma
  _Sigma.clear();
  for (int j = 0; j < _nr_modes; j++) {
    line.clear();
    do {
      getline(is, line);
      if (!is.good()) {
        ROS_ERROR("Traj name error");
        return false;
      }
      line = utils::trim(line);
    } while (line.empty());

    if (!utils::startsWith(line, "\"")) {
      ROS_ERROR("Traj name start error in %s", line.c_str());
      return false;
    }
    if (!utils::endsWith(line, "\"")) {
      ROS_ERROR("Traj name end error in %s", line.c_str());
      return false;
    }
    name = line.substr(1, line.size() - 2);
    // ROS_INFO("Data for  %s - parsing data", name.c_str());

    std::vector<double> Sigma_i0;
    if (!parseVector(is, Sigma_i0, "s"))
      return false;
    std::vector<double> Sigma_i1;
    if (!parseVector(is, Sigma_i1, "s"))
      return false;
    std::vector<double> Sigma_i2;
    if (!parseVector(is, Sigma_i2, "s"))
      return false;
    std::vector<double> Sigma_i3;
    if (!parseVector(is, Sigma_i3, "s"))
      return false;
    std::vector<double> Sigma_i4;
    if (!parseVector(is, Sigma_i4, "s"))
      return false;
    std::vector<double> Sigma_i5;
    if (!parseVector(is, Sigma_i5, "s"))
      return false;
    std::vector<double> Sigma_i6;
    if (!parseVector(is, Sigma_i6, "s"))
      return false;
    std::vector<double> Sigma_i7;
    if (!parseVector(is, Sigma_i7, "s"))
      return false;
    std::vector<double> Sigma_i8;
    if (!parseVector(is, Sigma_i8, "s"))
      return false;
    std::vector<double> Sigma_i9;
    if (!parseVector(is, Sigma_i9, "s"))
      return false;
    std::vector<double> Sigma_i10;
    if (!parseVector(is, Sigma_i10, "s"))
      return false;
    std::vector<double> Sigma_i11;
    if (!parseVector(is, Sigma_i11, "s"))
      return false;
    std::vector<double> Sigma_i12;
    if (!parseVector(is, Sigma_i12, "s"))
      return false;
    std::vector<double> Sigma_i13;
    if (!parseVector(is, Sigma_i13, "s"))
      return false;
    std::vector<double> Sigma_i14;
    if (!parseVector(is, Sigma_i14, "s"))
      return false;

    Eigen::MatrixXf Sigma_i(15, 15);
    for (int i = 0; i < 15; i++) {
      Sigma_i(i, 0) = Sigma_i0[i];
      Sigma_i(i, 1) = Sigma_i1[i];
      Sigma_i(i, 2) = Sigma_i2[i];
      Sigma_i(i, 3) = Sigma_i3[i];
      Sigma_i(i, 4) = Sigma_i4[i];
      Sigma_i(i, 5) = Sigma_i5[i];
      Sigma_i(i, 6) = Sigma_i6[i];
      Sigma_i(i, 7) = Sigma_i7[i];
      Sigma_i(i, 8) = Sigma_i8[i];
      Sigma_i(i, 9) = Sigma_i9[i];
      Sigma_i(i, 10) = Sigma_i10[i];
      Sigma_i(i, 11) = Sigma_i11[i];
      Sigma_i(i, 12) = Sigma_i12[i];
      Sigma_i(i, 13) = Sigma_i13[i];
      Sigma_i(i, 14) = Sigma_i14[i];
    }
    _Sigma.push_back(Sigma_i);
  }

  // set goal state
  _goalState.setOrigin(tf::Vector3(_Mu[_nr_modes - 1][1], _Mu[_nr_modes - 1][2], _Mu[_nr_modes - 1][3]));
  _goalState.setRotation(
    tf::Quaternion(_Mu[_nr_modes - 1][4], _Mu[_nr_modes - 1][5], _Mu[_nr_modes - 1][6], _Mu[_nr_modes - 1][7]));
  _goalState.stamp_ = ros::Time(_Mu[_nr_modes - 1][0]);

  // set related object parameters
  line.clear();
  do {
    getline(is, line);
    if (!is.good()) {
      ROS_ERROR("Traj name error");
      return false;
    }
    line = utils::trim(line);
  } while (line.empty());

  if (!utils::startsWith(line, "\"")) {
    ROS_ERROR("Traj name start error in %s", line.c_str());
    return false;
  }
  if (!utils::endsWith(line, "\"")) {
    ROS_ERROR("Traj name end error in %s", line.c_str());
    return false;
  }
  name = line.substr(1, line.size() - 2);
  // ROS_INFO("Data for  %s - parsing data", name.c_str());

  std::vector<double> object_pose;
  if (!parseVector(is, object_pose, "pose"))
    return false;
  _related_object_pose.setOrigin(tf::Vector3(object_pose[0], object_pose[1], object_pose[2]));
  _related_object_pose.setRotation(tf::Quaternion(object_pose[3], object_pose[4], object_pose[5], object_pose[6]));
  std::vector<double> object_grasp;
  if (!parseVector(is, object_grasp, "grasp"))
    return false;
  _related_object_grasp_pose.setOrigin(tf::Vector3(object_grasp[0], object_grasp[1], object_grasp[2]));
  _related_object_grasp_pose.setRotation(tf::Quaternion(object_pose[3], object_pose[4], object_pose[5], object_pose[6]));
  std::vector<char> object_name;
  if (!parseVector(is, object_name, "name"))
    return false;
  std::string objstr(object_name.begin(), object_name.end());
  _related_object_name = objstr;

  gmm_time_offset_ = 0.0;

  ROS_DEBUG("Successfully loaded GMM!");
  return true;
}

tf::Transform GaussianMixtureModel::getLastMuEigenBckGripper() const {
  int i = _nr_modes - 1;
  tf::Vector3 vec(_MuEigenBck[i](1), _MuEigenBck[i](2), _MuEigenBck[i](3));
  tf::Quaternion q(_MuEigenBck[i](4), _MuEigenBck[i](5), _MuEigenBck[i](6), _MuEigenBck[i](7));
  return tf::Transform(q, vec);
};
