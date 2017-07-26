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

#ifndef SVO_FRAME_HANDLER_H_
#define SVO_FRAME_HANDLER_H_

#include <set>
#include <vikit/abstract_camera.h>
#include <plsvo/frame_handler_base.h>
#include <plsvo/reprojector.h>
#include <plsvo/initialization.h>

namespace plsvo {

/// Monocular Visual Odometry Pipeline as described in the SVO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options
  {
    bool has_pt;  //!< use point features in the pipeline
    bool has_ls;  //!< use line segment features in the pipeline
    Options() :
      has_pt(true),
      has_ls(true)
    {}
    Options(bool has_pt, bool has_ls) :
      has_pt(has_pt),
      has_ls(has_ls)
    {}
  } options_;
  
  FrameHandlerMono(vk::AbstractCamera* cam);
  FrameHandlerMono(vk::AbstractCamera* cam, const Options& opts);
  virtual ~FrameHandlerMono();

  /// Auxiliar functions
  cv::Mat get_debug_image()
  {
    return debug_img;
  }

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp);
  void addImage(const cv::Mat& img, double timestamp, cv::Mat& rec);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

protected:
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.
  cv::Mat debug_img;

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  virtual void initialize(const Options opts);


  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);
};

} // namespace plsvo

#endif // SVO_FRAME_HANDLER_H_
