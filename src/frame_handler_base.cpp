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

#include <vikit/abstract_camera.h>
#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <plsvo/frame_handler_base.h>
#include <plsvo/config.h>
#include <plsvo/feature.h>
#include <plsvo/matcher.h>
#include <plsvo/map.h>
#include <plsvo/feature3D.h>

namespace plsvo
{

// definition of global and static variables which were declared in the header
#ifdef SVO_TRACE
vk::PerformanceMonitor* g_permon = NULL;
#endif

FrameHandlerBase::FrameHandlerBase() :
  stage_(STAGE_PAUSED),
  set_reset_(false),
  set_start_(false),
  acc_frame_timings_(10),
  acc_num_obs_(10),
  num_obs_last_(0),
  num_obs_last_pt(0),
  num_obs_last_ls(0),
  tracking_quality_(TRACKING_INSUFFICIENT)
{
#ifdef SVO_TRACE
  // Initialize Performance Monitor
  g_permon = new vk::PerformanceMonitor();
  g_permon->addTimer("pyramid_creation");
  g_permon->addTimer("sparse_img_align");
  g_permon->addTimer("reproject");
  g_permon->addTimer("reproject_kfs");
  g_permon->addTimer("reproject_candidates");
  g_permon->addTimer("feature_align");
  g_permon->addTimer("pose_optimizer");
  g_permon->addTimer("point_optimizer");
  g_permon->addTimer("local_ba");
  g_permon->addTimer("tot_time");
  g_permon->addLog("timestamp");
  g_permon->addLog("img_align_n_tracked");
  g_permon->addLog("repr_n_mps");
  g_permon->addLog("repr_n_new_references");
  g_permon->addLog("sfba_thresh");
  g_permon->addLog("sfba_error_init");
  g_permon->addLog("sfba_error_final");
  g_permon->addLog("sfba_n_edges_final");
  g_permon->addLog("loba_n_erredges_init");
  g_permon->addLog("loba_n_erredges_fin");
  g_permon->addLog("loba_err_init");
  g_permon->addLog("loba_err_fin");
  g_permon->addLog("n_candidates");
  g_permon->addLog("dropout");
  g_permon->init(Config::traceName(), Config::traceDir());
#endif

  SVO_INFO_STREAM("SVO initialized");
}

FrameHandlerBase::~FrameHandlerBase()
{
  SVO_INFO_STREAM("SVO destructor invoked");
#ifdef SVO_TRACE
  delete g_permon;
#endif
}

bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
  if(set_start_)
  {
    resetAll();
    stage_ = STAGE_FIRST_FRAME;
  }

  if(stage_ == STAGE_PAUSED)
    return false;

  SVO_LOG(timestamp);
  SVO_DEBUG_STREAM("New Frame");
  SVO_START_TIMER("tot_time");
  timer_.start();

  // some cleanup from last iteration, can't do before because of visualization
  map_.emptyTrash();
  return true;
}

int FrameHandlerBase::finishFrameProcessingCommon(const size_t update_id,
    const UpdateResult dropout,
    const size_t num_pt_observations, const size_t num_ls_observations)
{
  SVO_DEBUG_STREAM("Frame: "<<update_id<<"\t fps-avg = "<< 1.0/acc_frame_timings_.getMean()<<"\t nObs = "<<acc_num_obs_.getMean());
  SVO_LOG(dropout);

  // save processing time to calculate fps
  acc_frame_timings_.push_back(timer_.stop());
  if(stage_ == STAGE_DEFAULT_FRAME)
    acc_num_obs_.push_back(num_pt_observations);
  num_obs_last_ = num_pt_observations + num_ls_observations;
  num_obs_last_pt = num_pt_observations;
  num_obs_last_ls = num_ls_observations;

  SVO_STOP_TIMER("tot_time");

#ifdef SVO_TRACE
  g_permon->writeToFile();
  {
    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
    size_t n_candidates = map_.point_candidates_.candidates_.size();
    SVO_LOG(n_candidates);
  }
#endif

  if(dropout == RESULT_FAILURE &&
      (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
  {
    stage_            = STAGE_RELOCALIZING;
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
  else if (dropout == RESULT_FAILURE)
    resetAll();
  if(set_reset_)
    resetAll();

  return 0;
}

void FrameHandlerBase::resetCommon()
{
  map_.reset();
  stage_ = STAGE_PAUSED;
  set_reset_ = false;
  set_start_ = false;
  tracking_quality_ = TRACKING_INSUFFICIENT;
  num_obs_last_ = 0;
  num_obs_last_pt = 0;
  num_obs_last_ls = 0;
  SVO_INFO_STREAM("RESET");
}

void FrameHandlerBase::setTrackingQuality(const size_t num_pt_observations, size_t num_ls_observations)
{
  tracking_quality_ = TRACKING_GOOD;
  if( num_pt_observations + num_ls_observations < Config::qualityMinFts() )
  {
    SVO_WARN_STREAM_THROTTLE(0.5, "Tracking less than "<< Config::qualityMinFts() << " point and line segment features!");
    tracking_quality_ = TRACKING_BAD;
  }
  // Drop features only for the case of points
  const int feature_drop_pt = static_cast<int>(std::min(num_obs_last_pt, Config::maxFts()))     - num_pt_observations;
  const int feature_drop_ls = static_cast<int>(std::min(num_obs_last_ls, Config::maxFtsSegs())) - num_ls_observations;
  //if( feature_drop_pt > int(Config::qualityMaxFtsDrop()) && feature_drop_ls > int(Config::qualityMaxFtsDropSegs())  )
  if( feature_drop_pt > int(Config::qualityMaxFtsDrop()) )
  {
    SVO_WARN_STREAM("Lost "<< feature_drop_pt << " point and " << feature_drop_ls << " line segment features!");
    tracking_quality_ = TRACKING_BAD;
  }
}

bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

bool lsLastOptimComparator(LineSeg* lhs, LineSeg* rhs)
{
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

void FrameHandlerBase::optimizeStructure(
    FramePtr frame,
    size_t max_n_pts,
    int max_iter,
    size_t max_n_segs,
    int max_iter_segs)
{
  // Point features
  deque<Point*> pts;
  for(list<PointFeat*>::iterator it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D != NULL)
      pts.push_back((*it)->feat3D);
  }
  max_n_pts = min(max_n_pts, pts.size());
  nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), ptLastOptimComparator);
  for(deque<Point*>::iterator it=pts.begin(); it!=pts.begin()+max_n_pts; ++it)
  {
    (*it)->optimize(max_iter);
    (*it)->last_structure_optim_ = frame->id_;
  }
  // Line features
  deque<LineSeg*> segs;
  for(list<LineFeat*>::iterator it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
  {
    if((*it)->feat3D != NULL)
      segs.push_back((*it)->feat3D);
  }
  max_n_segs = min(max_n_segs, segs.size());
  nth_element(segs.begin(), segs.begin() + max_n_segs, segs.end(), lsLastOptimComparator);
  for(deque<LineSeg*>::iterator it=segs.begin(); it!=segs.begin()+max_n_segs; ++it)
  {
    (*it)->optimize(max_iter_segs);
    (*it)->last_structure_optim_ = frame->id_;
  }
}


} // namespace plsvo
