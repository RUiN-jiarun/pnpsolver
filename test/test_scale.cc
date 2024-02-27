#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "interface/pnp_solver.h"

int main(int argc, char** argv) {

    std::ifstream point2Ds_file("../point2Ds.txt");
    std::ifstream point3Ds_file("../point3Ds.txt");
    std::ifstream priors_file("../priors.txt");

    std::vector<Eigen::Vector2d> points2D;
    std::vector<Eigen::Vector3d> points3D;
    std::vector<double> priors;

    double a, b, x, y, z, prior;
    while (point2Ds_file >> a >> b && point3Ds_file >> x >> y >> z && priors_file >> prior)
    {
        points2D.emplace_back(a, b);
        points3D.emplace_back(x, y, z);
        priors.push_back(prior);
    }

    point2Ds_file.close();
    point3Ds_file.close();
    priors_file.close();

    std::string model_name = "PINHOLE";
    std::vector<double> params = {1727.69, 1727.69, 388, 521};

    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    std::vector<char> mask;

    size_t num_inliers = 0;
    if (sovle_pnp_ransac(points2D, points3D, model_name, params, qvec, tvec,
                            num_inliers, 12.0, 0.01, 0.9999, 10000, &mask,
                            colpnp::LORANSAC, colpnp::WEIGHT_SAMPLE,
                            &priors)) {
        std::cout << "qvec: " << qvec.transpose() << std::endl;
        std::cout << "tvec: " << tvec.transpose() << std::endl;
        std::cout << "inlier: " << num_inliers << std::endl;
    } else {
        std::cout << "Failed" << std::endl;
    }
    return 0;
}
