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

#ifndef SVO_DEPTH_FILTER_H_
#define SVO_DEPTH_FILTER_H_

#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <vikit/performance_monitor.h>
#include <plsvo/global.h>
#include <plsvo/feature_detection.h>
#include <plsvo/matcher.h>

namespace plsvo {

class Frame;
class Feature;
class PointFeat;
class LineFeat;
class Point;
class Line;

/// A seed is a probabilistic depth estimate.
struct Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int batch_counter;    //!< static counter for Keyframe Id (shared by all seed types)

  int batch_id;                //!< Batch id is the id of the keyframe for which the seed was created.
  int id;                      //!< Seed ID, only used for visualization.
  Seed(int batch_id, int id);
};

/// A seed for a single pixel.
struct PointSeed : public Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int seed_counter;     //!< static counter for the number of seeds of Point type

  PointFeat* ftr;                //!< Feature in the keyframe for which the depth should be computed.
  float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
  float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
  float mu;                    //!< Mean of normal distribution.
  float z_range;               //!< Max range of the possible depth.
  float sigma2;                //!< Variance of normal distribution.
  Matrix2d patch_cov;          //!< Patch covariance in reference image.
  PointSeed(PointFeat* ftr, float depth_mean, float depth_min);
};

/// A seed for a segment.
struct LineSeed : public Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int seed_counter;

  LineFeat* ftr;                //!< Feature in the keyframe for which the depth should be computed.
  float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
  float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
  float mu_s;                  //!< Mean of normal distribution (start point).
  float mu_e;                  //!< Mean of normal distribution (end point).
  float z_range_s;             //!< Max range of the possible depth (start point).
  float z_range_e;             //!< Max range of the possible depth(end point).
  float sigma2_s;              //!< Variance of normal distribution (start point).
  float sigma2_e;              //!< Variance of normal distribution(end point).
  Matrix2d patch_cov_s;        //!< Patch covariance in reference image (start point).
  Matrix2d patch_cov_e;        //!< Patch covariance in reference image(end point).

  LineSeed(LineFeat* ftr, float depth_mean, float depth_min);
};

/// Depth filter implements the Bayesian Update proposed in:
/// "Video-based, Real-Time Multi View Stereo" by G. Vogiatzis and C. Hernández.
/// In Image and Vision Computing, 29(7):434-441, 2011.
///
/// The class uses a callback mechanism such that it can be used also by other
/// algorithms than nslam and for simplified testing.
class DepthFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef boost::unique_lock<boost::mutex> lock_t;
  typedef boost::function<void ( Point*, double )> callback_t;
  typedef boost::function<void ( LineSeg*, double, double)> callback_t_ls;

  /// Depth-filter config parameters
  struct Options
  {
    bool check_ftr_angle;                       //!< gradient features are only updated if the epipolar line is orthogonal to the gradient.
    bool epi_search_1d;                         //!< restrict Gauss Newton in the epipolar search to the epipolar line.
    bool verbose;                               //!< display output.
    bool use_photometric_disparity_error;       //!< use photometric disparity error instead of 1px error in tau computation.
    int max_n_kfs;                              //!< maximum number of keyframes for which we maintain seeds.
    double sigma_i_sq;                          //!< image noise.
    double seed_convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    Options()
    : check_ftr_angle(false),
      epi_search_1d(false),
      verbose(false),
      use_photometric_disparity_error(false),
      max_n_kfs(3),
      sigma_i_sq(5e-4),
      seed_convergence_sigma2_thresh(200.0)
    {}
  } options_;

  DepthFilter(feature_detection::DetectorPtr<PointFeat> pt_feature_detector,
      feature_detection::DetectorPtr<LineFeat> seg_feature_detector,
      callback_t seed_converged_cb,
      callback_t_ls seed_converged_cb_ls);

  virtual ~DepthFilter();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  /// Add frame to the queue to be processed.
  void addFrame(FramePtr frame);

  /// Add new keyframe to the queue
  void addKeyframe(FramePtr frame, double depth_mean, double depth_min);
  void addKeyframe(FramePtr frame, double depth_mean, double depth_min, cv::Mat depth_image_, bool has_lines);

  /// Remove all seeds which are initialized from the specified keyframe. This
  /// function is used to make sure that no seeds points to a non-existent frame
  /// when a frame is removed from the map.
  void removeKeyframe(FramePtr frame);

  /// If the map is reset, call this function such that we don't have pointers
  /// to old frames.
  void reset();

  /// Returns a copy of the seeds belonging to frame. Thread-safe.
  /// Can be used to compute the Next-Best-View in parallel.
  /// IMPORTANT! Make sure you hold a valid reference counting pointer to frame
  /// so it is not being deleted while you use it.
  void getSeedsCopy(const FramePtr& frame, std::list<PointSeed>& seeds);

  /// Return a reference to the seeds. This is NOT THREAD SAFE!
  std::list<PointSeed, aligned_allocator<PointSeed> >& getSeeds() { return pt_seeds_; }

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updatePointSeed(
      const float x,
      const float tau2,
      PointSeed* seed);

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updateLineSeed(const float x_s,
      const float tau2_s, const float x_e, const float tau2_e,
      LineSeed *seed);

  /// Compute the uncertainty of the measurement.
  static double computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle);

  feature_detection::DetectorPtr<LineFeat> seg_feature_detector_;

protected:

  feature_detection::DetectorPtr<PointFeat> pt_feature_detector_;
  callback_t seed_converged_cb_;
  callback_t_ls seed_converged_cb_ls_;
  std::list<PointSeed, aligned_allocator<PointSeed> > pt_seeds_;
  std::list<LineSeed, aligned_allocator<LineSeed> > seg_seeds_;
  boost::mutex seeds_mut_;
  bool seeds_updating_halt_;            //!< Set this value to true when seeds updating should be interrupted.
  boost::thread* thread_;
  std::queue<FramePtr> frame_queue_;
  boost::mutex frame_queue_mut_;
  boost::condition_variable frame_queue_cond_;
  FramePtr new_keyframe_;               //!< Next keyframe to extract new seeds.
  bool new_keyframe_set_;               //!< Do we have a new keyframe to process?.
  double new_keyframe_min_depth_;       //!< Minimum depth in the new keyframe. Used for range in new seeds.
  double new_keyframe_mean_depth_;      //!< Maximum depth in the new keyframe. Used for range in new seeds.
  vk::PerformanceMonitor permon_;       //!< Separate performance monitor since the DepthFilter runs in a parallel thread.
  Matcher matcher_, matcherls_;

  /// Initialize new seeds from a frame.
  void initializeSeeds(FramePtr frame);
  void initializeSeeds(FramePtr frame, cv::Mat depth_image_, bool has_lines);

  /// Update all seeds with a new measurement frame (call feature-specific methods).
  virtual void updateSeeds(FramePtr frame);

  /// Update the point seeds with the new measurement frame
  void updatePointSeeds(FramePtr frame);

  /// Update the segment seeds with the new measurement frame
  void updateLineSeeds(FramePtr frame);

  /// When a new keyframe arrives, the frame queue should be cleared.
  void clearFrameQueue();

  /// A thread that is continuously updating the seeds.
  void updateSeedsLoop();
};

} // namespace plsvo

#endif // SVO_DEPTH_FILTER_H_
