#ifndef ELLIPSE
#define ELLIPSE

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <vector>
namespace ellipse {

    class Ellipse {
      private:
        // line_extraction::Line _line;
        double _height;
        double _width;
        double _alpha_ap;
        double _p1;
        double _p2;
        double _rho;
        double _alpha;
        double _gamma;
        std::string _type;
        Eigen::Matrix2f _R;
        bool _in_collision;
        std::vector<double> _hyper_normal;

        std::array<double, 2> _p_point;
        double _p_alpha;
        std::vector<double> _speed;

      public:
        Ellipse(double posx, double posy, std::string type);
        Ellipse(double posx, double posy, double alpha, double height);
        ~Ellipse();
        std::vector<double[]> calc_points();
        static double sBuffer;

        double getAngle();
        double getWidth();
        void setWidth(double width) { _width = width; };
        double getHeight();
        void setHeight(double height) { _height = height; };
        double getP1();
        double getP2();
        double getAlpha();
        double getRho();
        std::string getType();
        Eigen::Matrix2f getR();
        void setR(Eigen::Matrix2f R) { _R = R; };

        std::array<double, 2> &getPPoint();
        double getAlphaAp() { return _alpha_ap; };
        void setAlphaAp(double alpha_ap) { _alpha_ap = alpha_ap; };
        double getPPointAlpha() { return _p_alpha; };
        void setPPointAlpha(double p_alpha) { _p_alpha = p_alpha; };
        std::vector<double> getSpeed() { return _speed; };
        void setSpeed(std::vector<double> speed) { _speed = speed; };
        void setPPoint(double x, double y);
        void setGamma(double gamma);
        double getGamma();
        bool onLine(std::array<double, 2> &point);
        bool getInCollision() { return _in_collision; };
        void setInCollision(bool b) { _in_collision = b; };
        void setHyperNormal(std::vector<double> n) { _hyper_normal = n; };
        std::vector<double> getHyperNormal() { return _hyper_normal; };
    };

}  // namespace ellipse
#endif
