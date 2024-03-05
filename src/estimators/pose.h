// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_ESTIMATORS_POSE_H_
#define COLMAP_SRC_ESTIMATORS_POSE_H_

#include <vector>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "base/camera.h"
#include "base/camera_models.h"
#include "optim/loransac.h"
#include "util/alignment.h"

#include "util/threading.h"
#include "util/types.h"

namespace colmap {

enum RansacSampler {
    RANDOM_SAMPLE = 1,
    WEIGHT_SAMPLE = 2,
};

enum RansacRobustor { ROBUSTRER_RANSAC = 1, ROBUSTER_LORANSAC = 2 };

struct AbsolutePoseEstimationOptions {
    // Whether to estimate the focal length.
    bool estimate_focal_length = false;

    // Number of discrete samples for focal length estimation.
    size_t num_focal_length_samples = 30;

    // Minimum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double min_focal_length_ratio = 0.2;

    // Maximum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double max_focal_length_ratio = 5;

    // Number of threads for parallel estimation of focal length.
    int num_threads = ThreadPool::kMaxNumThreads;

    // Options used for P3P RANSAC.
    RANSACOptions ransac_options;

    void Check() const {
        // CHECK_GT(num_focal_length_samples, 0);
        // CHECK_GT(min_focal_length_ratio, 0);
        // CHECK_GT(max_focal_length_ratio, 0);
        // CHECK_LT(min_focal_length_ratio, max_focal_length_ratio);
        ransac_options.Check();
    }
};

struct AbsolutePoseRefinementOptions {
    // Convergence criterion.
    double gradient_tolerance = 1.0;

    // Maximum number of solver iterations.
    int max_num_iterations = 100;

    // Scaling factor determines at which residual robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine 3D points scale.
    bool fix_x = false;
    bool fix_y = false;
    bool fix_z = false;

    // Whether to print final summary.
    bool print_summary = true;

    void Check() const {
        // CHECK_GE(gradient_tolerance, 0.0);
        // CHECK_GE(max_num_iterations, 0);
        // CHECK_GE(loss_function_scale, 0.0);
    }
};

// Estimate absolute pose (optionally focal length) from 2D-3D correspondences.
//
// Focal length estimation is performed using discrete sampling around the
// focal length of the given camera. The focal length that results in the
// maximal number of inliers is assigned to the given camera.
//
// @param options              Absolute pose estimation options.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
//
// @param robustor             ransac or lo-ransac.
// @param sampler              random, weighted, progressive sampling.
// @param priors               if weighted or progressive sampling, the priors
// should be provided
//
// @return                     Whether pose is estimated successfully.
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers,
                          std::vector<char>* inlier_mask,
                          RansacRobustor robustor, RansacSampler sampler,
                          const std::vector<double>& priors);

// Refine absolute pose (optionally focal length) from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
//
// @return                     Whether the solution is usable.
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const Eigen::Matrix4d& Tow,
                        const Eigen::Vector3d& center,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera,
                        Eigen::Vector3d* scale_factors);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_POSE_H_
