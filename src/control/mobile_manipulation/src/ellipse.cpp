#include "mobile_manipulation_rl/ellipse.h"

namespace ellipse {

  double Ellipse::sBuffer = 0.50;

  // Ellipse build to represent the IR of the Robot
  Ellipse::Ellipse(double posx, double posy, std::string type) : _p1(1.0), _p2(1.0) {
    _p_point = {{posx, posy}};
    _type = type;
    _in_collision = false;
    _R << 1.0, 0.0, 0.0, 1.0;
    std::vector<double> n;
    n.push_back(0.0);
    n.push_back(1.0);
    setHyperNormal(n);
    if (type == "inner") {
      _width = 0.6;
      _height = 0.5;
      _rho = 1.88;
    } else {
      _width = 0.85;
      _height = 0.8;
      _rho = 0.08;
      _p_alpha = 0.0;
      _alpha_ap = M_PI / 8.0;
    }
    _speed.push_back(0.0);
    _speed.push_back(0.0);
  }

  // Ellipse build from position and alpha
  Ellipse::Ellipse(double posx, double posy, double alpha, double height) : _width(sBuffer), _p1(4.0), _p2(4.0), _rho(0.03) {
    _p_point = {{posx, posy}};
    _alpha = alpha;
    _height = height;
    _in_collision = false;
    double cosangle = cos(alpha);
    double sinangle = sin(alpha);
    // _R << cosangle ,-sinangle , sinangle,cosangle;
    _R << cosangle, sinangle, -sinangle, cosangle;
    _type = "obstacle";
    _speed.push_back(0.0);
    _speed.push_back(0.0);
    std::vector<double> n;
    n.push_back(0.0);
    n.push_back(1.0);
    setHyperNormal(n);
  }

  Ellipse::~Ellipse() {}

  double Ellipse::getWidth() { return _width; }

  double Ellipse::getHeight() { return _height; }

  double Ellipse::getP1() { return _p1; }

  double Ellipse::getP2() { return _p2; }

  double Ellipse::getAlpha() {
    return atan2(-_R(1, 0), _R(0, 0));  //_alpha;
  }

  double Ellipse::getRho() { return _rho; }

  Eigen::Matrix2f Ellipse::getR() { return _R; }

  std::array<double, 2> &Ellipse::getPPoint() { return _p_point; }

  void Ellipse::setPPoint(double x, double y) { _p_point = {{x, y}}; }

  std::string Ellipse::getType() { return _type; }

  void Ellipse::setGamma(double gamma) { _gamma = gamma; }

  double Ellipse::getGamma() { return _gamma; }

}  // namespace ellipse
