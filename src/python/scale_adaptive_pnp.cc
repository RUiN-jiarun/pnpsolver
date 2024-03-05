#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include "interface/pnp_solver.h"

namespace py = pybind11;

py::dict scale_adaptive_pnp(
    const Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>>
        points2D,
    const Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor>>
        points3D,
    const Eigen::Ref<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>
        priors, 
    const Eigen::Ref<Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>>
        Tow,
    const Eigen::Ref<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>
        center,
    const py::dict camera,
    const py::dict pnp_option) {

    assert(points2D.cols() == points3D.cols());
    assert(points3D.cols() == priors.cols());
    assert(Tow.cols() == 4);
    assert(center.cols() == 3);

    std::string camera_model_name = camera["model_name"].cast<std::string>();
    std::vector<double> params = camera["params"].cast<std::vector<double>>();

    std::vector<Eigen::Vector2d> point2D_vec(points2D.cols());
    std::vector<Eigen::Vector3d> point3D_vec(points3D.cols());
    std::vector<double> priors_vec(priors.cols());
    for (size_t i = 0; i != point2D_vec.size(); ++i) {
        point2D_vec[i][0] = static_cast<double>(points2D(0, i));
        point2D_vec[i][1] = static_cast<double>(points2D(1, i));
        point3D_vec[i][0] = static_cast<double>(points3D(0, i));
        point3D_vec[i][1] = static_cast<double>(points3D(1, i));
        point3D_vec[i][2] = static_cast<double>(points3D(2, i));
        priors_vec[i] = static_cast<double>(priors(0, i));
    }

    Eigen::Matrix4d Tow_mat;
    Tow_mat << static_cast<double>(Tow(0, 0)), static_cast<double>(Tow(0, 1)), static_cast<double>(Tow(0, 2)), static_cast<double>(Tow(0, 3)),
               static_cast<double>(Tow(1, 0)), static_cast<double>(Tow(1, 1)), static_cast<double>(Tow(1, 2)), static_cast<double>(Tow(1, 3)),
               static_cast<double>(Tow(2, 0)), static_cast<double>(Tow(2, 1)), static_cast<double>(Tow(2, 2)), static_cast<double>(Tow(2, 3)),
               static_cast<double>(Tow(3, 0)), static_cast<double>(Tow(3, 1)), static_cast<double>(Tow(3, 2)), static_cast<double>(Tow(3, 3));
    // std::cout << Tow_mat << std::endl;

    Eigen::Vector3d center_vec;
    center_vec[0] = static_cast<double>(center(0, 0));
    center_vec[1] = static_cast<double>(center(0, 1));
    center_vec[2] = static_cast<double>(center(0, 2));

    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    Eigen::Vector3d scale_factors{1.0, 1.0, 1.0};
    double error_thres = pnp_option["error_thres"].cast<double>();
    double inlier_ratio = pnp_option["inlier_ratio"].cast<double>();
    double confidence = pnp_option["confidence"].cast<double>();
    double max_iter = pnp_option["max_iter"].cast<double>();
    bool fix_x = pnp_option["fix_x"].cast<bool>();
    bool fix_y = pnp_option["fix_y"].cast<bool>();
    bool fix_z = pnp_option["fix_z"].cast<bool>();
    std::vector<bool> fix_scale{fix_x, fix_y, fix_z};
    std::vector<char> mask;

    colpnp::Robustor robustor = colpnp::RANSAC;
    bool lo = pnp_option["local_optimal"].cast<bool>();
    if (lo) {
        robustor = colpnp::LORANSAC;
    }

    py::dict result;
    result["ninlier"] = 0;
    result["mask"] = mask;
    result["qvec"] = qvec;
    result["tvec"] = tvec;
    result["scale"] = scale_factors;

    size_t num_inliers = 0;
    bool success = colpnp::solve_pnp_ransac(
        point2D_vec, point3D_vec, camera_model_name, params, Tow_mat, center_vec, qvec, tvec, scale_factors, fix_scale, 
        num_inliers, error_thres, inlier_ratio, confidence, max_iter, &mask,
        robustor, colpnp::WEIGHT_SAMPLE, &priors_vec);
    if (success) {
        result["ninlier"] = num_inliers;
        result["mask"] = mask;
        result["qvec"] = qvec;
        result["tvec"] = tvec;
        result["scale"] = scale_factors;
    }

    return result;
}

PYBIND11_MODULE(mypnp, m) {
    m.def("scale_adaptive_pnp", &scale_adaptive_pnp);
}