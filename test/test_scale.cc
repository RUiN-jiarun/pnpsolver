#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "interface/pnp_solver.h"

int main(int argc, char** argv) {

    std::ifstream point2Ds_file("../example/D/point2Ds.txt");
    std::ifstream point3Ds_file("../example/D/point3Ds.txt");
    std::ifstream priors_file("../example/D/priors.txt");

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

    Eigen::Vector4d gt_qvec(0.998926, 0.0453978, -0.0091101, 0.00184613);
    Eigen::Vector3d gt_tvec(1.26049, 1.14126, 5.99852);
    std::cout << "GT qvec: " << gt_qvec.transpose() << std::endl;
    std::cout << "GT tvec: " << gt_tvec.transpose() << std::endl;
    std::cout << "=========================================================" << std::endl;

    Eigen::Vector3d scale_factors(1.0, 1.0, 1.0);
    std::vector<bool> fix_scale{1, 1, 0};
    // Eigen::Vector3d center(-1.217529, 1.931424, 10.653617);
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    Eigen::Matrix4d Tow; 
    // Tow <<  -1.14216520e+00,  4.29204776e+00,  1.23306060e+01, -1.61736221e+02,
    //         -1.30553074e+01, -2.88119428e-01, -1.10894186e+00, -1.05676075e+00,
    //         -9.24277981e-02, -1.23787525e+01,  4.30071730e+00,  1.33246706e+00,
    //          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00;
    Tow <<  -1.04581324e+00,  4.36154647e+00,  1.19901674e+01, -1.48835425e+02,
            -1.27560466e+01, -1.37276147e-01, -1.06420439e+00, -5.03243888e+00,
            -2.33959876e-01, -1.20349300e+01,  4.35579721e+00,  1.53103024e+01,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00;
    // Eigen::Vector4d center_homo(-1.217529, 1.931424, 10.653617, 1.0);
    // Eigen::Vector4d trans_center = Tow * center_homo;
    // std::cout << trans_center.transpose() << std::endl;
    size_t num_inliers = 0;
    if (solve_pnp_ransac(points2D, points3D, model_name, params, Tow, center, qvec, tvec, scale_factors, fix_scale,
                            num_inliers, 12.0, 0.01, 0.9999, 10000, &mask,
                            colpnp::LORANSAC, colpnp::WEIGHT_SAMPLE,
                            &priors)) {
        std::cout << "Scale factors: " << scale_factors.transpose() << std::endl;
        std::cout << "Final qvec: " << qvec.transpose() << std::endl;
        std::cout << "Final tvec: " << tvec.transpose() << std::endl;
        std::cout << "Inliers: " << num_inliers << std::endl;
    } else {
        std::cout << "Failed" << std::endl;
    }
    return 0;
}
