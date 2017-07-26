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


#include <plsvo/config.h>
#include <plsvo/frame_handler_mono.h>
#include <plsvo/map.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <plsvo/feature3D.h>
#include <plsvo/pose_optimizer.h>
#include <plsvo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <plsvo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <plsvo/bundle_adjustment.h>
#endif

namespace plsvo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
  initialize();
}

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera *cam, const FrameHandlerMono::Options& opts) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL),
  options_(opts)
{
  initialize(opts);
}

void FrameHandlerMono::initialize()
{
  // create a point feature detector instance
  feature_detection::DetectorPtr<PointFeat> pt_feature_detector;
  if(Config::hasPoints())
    pt_feature_detector = feature_detection::DetectorPtr<PointFeat>(
          new feature_detection::FastDetector(
            cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  else
    // create an abstract (void) detector that detects nothing to deactivate use of points
    pt_feature_detector = feature_detection::DetectorPtr<PointFeat>(
          new feature_detection::AbstractDetector<PointFeat>(
            cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

  // create a segment feature detector instance
  feature_detection::DetectorPtr<LineFeat> seg_feature_detector;
  if(Config::hasLines())
    seg_feature_detector = feature_detection::DetectorPtr<LineFeat>(
          new feature_detection::LsdDetector(
            cam_->width(), cam_->height(), Config::gridSizeSegs(), Config::nPyrLevelsSegs()));
  else
    // create an abstract (void) detector that detects nothing to deactivate use of line segs
    seg_feature_detector = feature_detection::DetectorPtr<LineFeat>(
          new feature_detection::AbstractDetector<LineFeat>(
            cam_->width(), cam_->height(), Config::gridSizeSegs(), Config::nPyrLevelsSegs()));

  // create the callback object for the Depth-Filter
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);

  DepthFilter::callback_t_ls depth_filter_cb_ls = boost::bind(
      &MapSegmentCandidates::newCandidateSegment, &map_.segment_candidates_, _1, _2, _3);

  // Setup the Depth-Filter object
  depth_filter_ = new DepthFilter(pt_feature_detector, seg_feature_detector, depth_filter_cb, depth_filter_cb_ls );
  depth_filter_->startThread();
}

void FrameHandlerMono::initialize(const Options opts)
{
  // create a point feature detector instance
  feature_detection::DetectorPtr<PointFeat> pt_feature_detector;
  if(opts.has_pt)
    pt_feature_detector = feature_detection::DetectorPtr<PointFeat>(
          new feature_detection::FastDetector(
            cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  else
    // create an abstract (void) detector that detects nothing to deactivate use of points
    pt_feature_detector = feature_detection::DetectorPtr<PointFeat>(
          new feature_detection::AbstractDetector<PointFeat>(
            cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

  // create a segment feature detector instance
  feature_detection::DetectorPtr<LineFeat> seg_feature_detector;
  if(opts.has_ls)
    seg_feature_detector = feature_detection::DetectorPtr<LineFeat>(
          new feature_detection::LsdDetector(
            cam_->width(), cam_->height(), Config::gridSizeSegs(), Config::nPyrLevelsSegs()));
  else
    // create an abstract (void) detector that detects nothing to deactivate use of line segs
    seg_feature_detector = feature_detection::DetectorPtr<LineFeat>(
          new feature_detection::AbstractDetector<LineFeat>(
            cam_->width(), cam_->height(), Config::gridSizeSegs(), Config::nPyrLevelsSegs()));

  // create the callback object for the Depth-Filter
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);

  DepthFilter::callback_t_ls depth_filter_cb_ls = boost::bind(
      &MapSegmentCandidates::newCandidateSegment, &map_.segment_candidates_, _1, _2, _3);

  // Setup the Depth-Filter object
  depth_filter_ = new DepthFilter(pt_feature_detector, seg_feature_detector, depth_filter_cb, depth_filter_cb_ls );
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{

  img.copyTo(debug_img);

  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  // The Frame constructor initializes new_frame_
  // and creates the image pyramid (also stored in Frame as img_pyr_)
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs(), last_frame_->nLsObs());
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp, cv::Mat& rec)
{

  img.copyTo(debug_img);

  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  // The Frame constructor initializes new_frame_
  // and creates the image pyramid (also stored in Frame as img_pyr_)
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  rec.copyTo(new_frame_->rec_img);
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs(), last_frame_->nLsObs());
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  // set first frame to identity transformation
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  // for now the initialization is done with points and endpoints only (consider use lines)
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  bool display = false;
  bool verbose = false;
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, display, verbose);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // show reference features
  cv::cvtColor(last_frame_->img(), FrameHandlerMono::debug_img, cv::COLOR_GRAY2BGR);
  {
    // draw point features
    {
      auto fts = last_frame_->pt_fts_;
      Patch patch( 4, debug_img );
      for(auto it=fts.begin(); it!=fts.end(); ++it)
      {
        patch.setPosition((*it)->px);
        patch.setRoi();
        cv::rectangle(debug_img,patch.rect,cv::Scalar(0,255,0));
      }
    }
    // draw segment features
    {
      auto fts = last_frame_->seg_fts_;
      std::for_each(fts.begin(), fts.end(), [&](plsvo::LineFeat* i){
          if( i->feat3D != NULL )
            cv::line(debug_img,cv::Point2f(i->spx[0],i->spx[1]),cv::Point2f(i->epx[0],i->epx[1]),cv::Scalar(0,255,0));
      });
    }
    //cv::imshow("cv: Ref image", debug_img);
    //cv::waitKey(30);
  }

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references_pt = reprojector_.n_matches_;
  const size_t repr_n_new_references    = repr_n_new_references_pt;
  const size_t repr_n_new_references_ls = reprojector_.n_ls_matches_;
  const size_t repr_n_mps               = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM( "Reprojection:\t nPoints & nLines = " << repr_n_mps << "\t \t nMatches = " << repr_n_new_references_pt + repr_n_new_references_ls );
  if( repr_n_new_references_pt + repr_n_new_references_ls < Config::qualityMinFts() )
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final, sfba_n_edges_final_pt, sfba_n_edges_final_ls;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final_pt, sfba_n_edges_final_ls);
  SVO_STOP_TIMER("pose_optimizer");
  sfba_n_edges_final = sfba_n_edges_final_pt + sfba_n_edges_final_ls;
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final,sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final_pt+sfba_n_edges_final_ls);
  if( sfba_n_edges_final_pt+sfba_n_edges_final_ls < 10) // check this (include other factors)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter(), Config::structureOptimMaxSegs(), Config::structureOptimNumIterSegs());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final_pt,sfba_n_edges_final_ls);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if( !needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD )
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(list<PointFeat*>::iterator it=new_frame_->pt_fts_.begin(); it!=new_frame_->pt_fts_.end(); ++it)
    if((*it)->feat3D != NULL)
      (*it)->feat3D->addFrameRef(*it);
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  for(list<LineFeat*>::iterator it=new_frame_->seg_fts_.begin(); it!=new_frame_->seg_fts_.end(); ++it)
    if((*it)->feat3D != NULL)
      (*it)->feat3D->addFrameRef(*it);
  map_.segment_candidates_.addCandidateSegmentToFrame(new_frame_);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean*2.0, 0.1*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;

}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    // Estimate rotation distance
    SE3 T_last, T_curr, delta_T;
    T_last = last_frame_->T_f_w_;
    T_curr = it->first->T_f_w_;
    delta_T = T_last.inverse() * T_curr;

    Vector6d relpos_ = delta_T.log();
    double delta_t = sqrt( pow(relpos_(0),2) + pow(relpos_(1),2) + pow(relpos_(2),2) );
    double delta_r = sqrt( pow(relpos_(3),2) + pow(relpos_(4),2) + pow(relpos_(5),2) ) * 180.0 / 3.1416;
    if( delta_t < Config::kfSelectMinDistT() && delta_r < Config::kfSelectMinDistR() )
        return false;

    /*// they estimated with the scene depth...
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;*/
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace plsvo
