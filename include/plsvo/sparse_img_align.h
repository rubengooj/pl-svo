/*****************************************************************************
**       PL-SVO: Semi-direct Monocular Visual Odometry by Combining       	**
**                        Points and Line Segments                          **
******************************************************************************
**																			**
**	Copyright(c) 2016, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2016, MAPIR group, University of Malaga					**
**																			**
**  This library extends SVO to the case of also using line-segments, thus  **
**  it is highly based on the previous implementation of SVO:               **
**  https://github.com/uzh-rpg/rpg_svo                                      **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/


#ifndef SVO_SPARSE_IMG_ALIGN_H_
#define SVO_SPARSE_IMG_ALIGN_H_

#include <vikit/nlls_solver.h>
#include <vikit/performance_monitor.h>
#include <plsvo/global.h>

namespace vk {
class AbstractCamera;
}

namespace plsvo {

class Feature;
class PointFeat;
class LineFeat;

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlign : public vk::NLLSSolver<6, SE3>
{
  static const int patch_halfsize_ = 2;
  static const int patch_size_ = 2*patch_halfsize_;
  static const int patch_area_ = patch_size_*patch_size_;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  cv::Mat resimg_;

  SparseImgAlign(
      int n_levels,
      int min_level,
      int n_iter,
      Method method,
      bool display,
      bool verbose);

  size_t run(
      FramePtr ref_frame,
      FramePtr cur_frame);

  /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
  /// at the converged state.
  Matrix<double, 6, 6> getFisherInformation();

protected:
  FramePtr ref_frame_;            //!< reference frame, has depth for gradient pixels.
  FramePtr cur_frame_;            //!< only the image is known!
  int level_;                     //!< current pyramid level on which the optimization runs.
  bool display_;                  //!< display residual image.
  int max_level_;                 //!< coarsest pyramid level for the alignment.
  int min_level_;                 //!< finest pyramid level for the alignment.
  double scale_ls, scale_pt;

  // cache:
  /// Cached values for all pixels corresponding to a list of feature-patches
  struct Cache
  {
    Matrix<double, 6, Dynamic, ColMajor> jacobian;  // cached jacobian
    cv::Mat ref_patch;  // cached patch intensity values (with subpixel precision)
    std::vector<bool> visible_fts; // mask of visible features
    Cache() {} // default constructor
    Cache( size_t num_fts, int patch_area ) // constructor
    {
      // resize cache variables according to the maximum number of incoming features
      ref_patch = cv::Mat(num_fts, patch_area, CV_32F);
      jacobian.resize(Eigen::NoChange, num_fts*patch_area);
      visible_fts.resize(num_fts, false); // TODO: should it be reset at each level?
    }
  };
  Cache pt_cache_;
  Cache seg_cache_;
  std::vector<size_t> patch_offset;   // offset for the segment cache
  bool have_ref_patch_cache_;   // flag to avoid recomputing cache

  void precomputeReferencePatches();
  void precomputeGaussNewtonParamsPoints(Cache &cache, list<PointFeat*> &fts);
  void precomputeGaussNewtonParamsSegments(Cache &cache, list<LineFeat*> &fts);
  virtual double computeResiduals(const SE3& model, bool linearize_system, bool compute_weight_scale = true);
  void computeGaussNewtonParamsPoints(
      const SE3 &T_cur_from_ref, bool linearize_system, bool compute_weight_scale,
      Cache &cache, list<PointFeat*> &fts, Matrix<double, 6, 6> &H, Matrix<double, 6, 1> &Jres,
      std::vector<float> &errors, float &chi2);
  void computeGaussNewtonParamsSegments(const SE3 &T_cur_from_ref, bool linearize_system, bool compute_weight_scale,
      Cache &cache, list<plsvo::LineFeat*> &fts, Matrix<double, 6, 6> &H, Matrix<double, 6, 1> &Jres,
      std::vector<float> &errors, float &chi2);
  virtual int solve();
  virtual void update (const ModelType& old_model, ModelType& new_model);
  virtual void startIteration();
  virtual void finishIteration();
};

} // namespace plsvo

#endif // SVO_SPARSE_IMG_ALIGN_H_
