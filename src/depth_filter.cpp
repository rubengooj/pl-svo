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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <plsvo/global.h>
#include <plsvo/depth_filter.h>
#include <plsvo/frame.h>
#include <plsvo/feature3D.h>
#include <plsvo/feature.h>
#include <plsvo/matcher.h>
#include <plsvo/config.h>
#include <plsvo/feature_detection.h>

namespace plsvo {

int Seed::batch_counter     = 0;
int PointSeed::seed_counter = 0;
int LineSeed::seed_counter  = 0;

Seed::Seed(int _batch_id, int _id) :
    batch_id(_batch_id),
    id(_id)
{}

PointSeed::PointSeed(PointFeat* ftr, float depth_mean, float depth_min) :
    Seed(batch_counter,seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{}

LineSeed::LineSeed(LineFeat *ftr, float depth_mean, float depth_min) :
    Seed(batch_counter,seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    mu_s(1.0/depth_mean),
    mu_e(1.0/depth_mean),
    z_range_s(1.0/depth_min),
    z_range_e(1.0/depth_min),
    sigma2_s(z_range_s*z_range_s/36),
    sigma2_e(z_range_e*z_range_e/36)
{}

DepthFilter::DepthFilter(
    feature_detection::DetectorPtr<PointFeat> pt_feature_detector,
    feature_detection::DetectorPtr<LineFeat>  seg_feature_detector,
    callback_t seed_converged_cb,
    callback_t_ls seed_converged_cb_ls) :
    pt_feature_detector_(pt_feature_detector),
    seg_feature_detector_(seg_feature_detector),
    seed_converged_cb_(seed_converged_cb),
    seed_converged_cb_ls_(seed_converged_cb_ls),
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0)
{}

DepthFilter::~DepthFilter()
{
  stopThread();
  SVO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread()
{
  thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
  SVO_INFO_STREAM("DepthFilter stop thread invoked.");
  if(thread_ != NULL)
  {
    SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
    seeds_updating_halt_ = true;
    thread_->interrupt();
    thread_->join();
    thread_ = NULL;
  }
}

void DepthFilter::addFrame(FramePtr frame)
{
  if(thread_ != NULL)
  {
    {
      lock_t lock(frame_queue_mut_);
      if(frame_queue_.size() > 2)
        frame_queue_.pop();
      frame_queue_.push(frame);
    }
    seeds_updating_halt_ = false;
    frame_queue_cond_.notify_one();
  }
  else
    updateSeeds(frame);
}

void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
{
  new_keyframe_min_depth_  = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  // if there exists an parallel thread for Depth-Filter just setup the control variables to jump into it
  if(thread_ != NULL)
  {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;
    seeds_updating_halt_ = true;
    frame_queue_cond_.notify_one();
  }
  // if there is no independent thread for Depth-Filter call the initialization of seeds explicitly (same effect)
  else
    initializeSeeds(frame);

}

void DepthFilter::initializeSeeds(FramePtr frame)
{

  list<PointFeat*> new_pt_features;
  list<LineFeat*> new_seg_features;

  if(Config::hasPoints())
  {
      /* detect new point features in point-unpopulated cells of the grid */
      // populate the occupancy grid of the detector with current features
      pt_feature_detector_->setExistingFeatures(frame->pt_fts_);
      // detect features to fill the free cells in the image
      pt_feature_detector_->detect(frame.get(), frame->img_pyr_,
                                Config::triangMinCornerScore(), new_pt_features);
  }

  if(Config::hasLines())
  {
      /* detect new segment features in line-unpopulated cells of the grid */
      // populate the occupancy grid of the detector with current features
      seg_feature_detector_->setExistingFeatures(frame->seg_fts_);
      // detect features
      seg_feature_detector_->detect(frame.get(), frame->img_pyr_,
                                Config::lsdMinLength(), new_seg_features);
  }

  // lock the parallel updating thread?
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
  // increase by one the keyframe counter (to account for this new one)
  ++PointSeed::batch_counter;

  // initialize a point seed for every new point feature
  std::for_each(new_pt_features.begin(), new_pt_features.end(), [&](PointFeat* ftr){
    pt_seeds_.push_back(PointSeed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  // initialize a segment seed for every new segment feature
  std::for_each(new_seg_features.begin(), new_seg_features.end(), [&](LineFeat* ftr){
    seg_seeds_.push_back(LineSeed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized "<<new_pt_features.size()<<" new point seeds and "
                    <<new_seg_features.size()<<" line segment seeds.");
  seeds_updating_halt_ = false;
}

void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  list<PointSeed>::iterator it=pt_seeds_.begin();
  size_t n_removed = 0;
  while(it!=pt_seeds_.end())
  {
    if(it->ftr->frame == frame.get())
    {
      it = pt_seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}

void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    pt_seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false;

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::updateSeedsLoop()
{
  while(!boost::this_thread::interruption_requested())
  {
    FramePtr frame;
    {
      lock_t lock(frame_queue_mut_);
      while(frame_queue_.empty() && new_keyframe_set_ == false)
        frame_queue_cond_.wait(lock);
      if(new_keyframe_set_)
      {
        new_keyframe_set_ = false;
        seeds_updating_halt_ = false;
        clearFrameQueue();
        frame = new_keyframe_;
      }
      else
      {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }
    updateSeeds(frame);
    if(frame->isKeyframe())
      initializeSeeds(frame);
  }
}

void DepthFilter::updateSeeds(FramePtr frame)
{
  // update the point-type seeds
  updatePointSeeds(frame);
  // update the line-type seeds
  updateLineSeeds(frame);
}

void DepthFilter::updatePointSeeds(FramePtr frame)
{
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  size_t n_updates=0, n_failed_matches=0, n_seeds = pt_seeds_.size();
  lock_t lock(seeds_mut_);
  list<PointSeed>::iterator it=pt_seeds_.begin();

  const double focal_length = frame->cam_->errorMultiplier2();
  double px_noise = 1.0;
  double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

  while( it!=pt_seeds_.end())
  {
    // set this value true when seeds updating should be interrupted
    if(seeds_updating_halt_)
      return;

    // check if seed is not already too old
    if((PointSeed::batch_counter - it->batch_id) > options_.max_n_kfs) {
      it = pt_seeds_.erase(it);
      continue;
    }

    // check if point is visible in the current image
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f) );
    if(xyz_f.z() < 0.0)  {
      ++it; // behind the camera
      continue;
    }
    if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
      ++it; // point does not project in image
      continue;
    }

    // we are using inverse depth coordinates
    float z_inv_min = it->mu + sqrt(it->sigma2);
    float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
    double z;
    if(!matcher_.findEpipolarMatchDirect(
        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
      it->b++; // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }

    // compute tau
    double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate
    updatePointSeed(1./z, tau_inverse*tau_inverse, &*it);
    ++n_updates;

    if(frame->isKeyframe())
    {
      // The feature detector should not initialize new seeds close to this location
      pt_feature_detector_->setGridOccpuancy(PointFeat(matcher_.px_cur_));
    }

    // if the seed has converged, we initialize a new candidate point and remove the seed
    if(sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
    {
      assert(it->ftr->feat3D == NULL); // TODO this should not happen anymore
      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0/it->mu)));
      Point* point = new Point(xyz_world, it->ftr);
      it->ftr->feat3D = point;
      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new PointFeat(frame.get(), matcher_.px_cur_, matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
      {
        seed_converged_cb_(point, it->sigma2); // put in candidate list
      }
      it = pt_seeds_.erase(it);
    }
    else if(isnan(z_inv_min))
    {
      SVO_WARN_STREAM("z_min is NaN");
      it = pt_seeds_.erase(it);
    }
    else
      ++it;
  }

}

void DepthFilter::updateLineSeeds(FramePtr frame)
{
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  size_t n_updates=0, n_failed_matches=0, n_seeds = seg_seeds_.size();
  lock_t lock(seeds_mut_);
  list<LineSeed>::iterator it=seg_seeds_.begin();

  const double focal_length = frame->cam_->errorMultiplier2();
  double px_noise = 1.0;
  double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

  while( it!=seg_seeds_.end())
  {
    // set this value true when seeds updating should be interrupted
    if(seeds_updating_halt_)
      return;

    // check if seed is not already too old
    if((LineSeed::batch_counter - it->batch_id) > options_.max_n_kfs) {
      it = seg_seeds_.erase(it);
      continue;
    }

    // check if segment is visible in the current image
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    const Vector3d xyz_f_s(T_ref_cur.inverse()*(1.0/it->mu_s * static_cast<LineFeat*>(it->ftr)->sf) );
    const Vector3d xyz_f_e(T_ref_cur.inverse()*(1.0/it->mu_e * static_cast<LineFeat*>(it->ftr)->ef) );
    if( xyz_f_s.z() < 0.0 || xyz_f_e.z() < 0.0 )  {
      ++it; // behind the camera
      continue;
    }
    if( !frame->cam_->isInFrame(frame->f2c(xyz_f_s).cast<int>()) ||
        !frame->cam_->isInFrame(frame->f2c(xyz_f_e).cast<int>()) ) {
      ++it; // segment does not project in image
      continue;
    }

    // we are using inverse depth coordinates
    float z_inv_min_s = it->mu_s + sqrt(it->sigma2_s);
    float z_inv_max_s = max(it->mu_s - sqrt(it->sigma2_s), 0.00000001f);
    float z_inv_min_e = it->mu_e + sqrt(it->sigma2_e);
    float z_inv_max_e = max(it->mu_e - sqrt(it->sigma2_e), 0.00000001f);
    double z_s, z_e;
    if(!matcherls_.findEpipolarMatchDirectSegmentEndpoint(
        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu_s, 1.0/z_inv_min_s, 1.0/z_inv_max_s, z_s) ||
       !matcherls_.findEpipolarMatchDirectSegmentEndpoint(
        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu_e, 1.0/z_inv_min_e, 1.0/z_inv_max_e, z_e)  )
    {
      it->b++; // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }

    // compute tau
    double tau_s = computeTau(T_ref_cur, static_cast<LineFeat*>(it->ftr)->sf, z_s, px_error_angle);
    double tau_inverse_s = 0.5 * (1.0/max(0.0000001, z_s-tau_s) - 1.0/(z_s+tau_s));
    double tau_e = computeTau(T_ref_cur, static_cast<LineFeat*>(it->ftr)->ef, z_e, px_error_angle);
    double tau_inverse_e = 0.5 * (1.0/max(0.0000001, z_e-tau_e) - 1.0/(z_e+tau_e));

    // update the estimate
    updateLineSeed(1./z_s, tau_inverse_s*tau_inverse_s, 1./z_e, tau_inverse_e*tau_inverse_e, &*it);
    ++n_updates;

    if(frame->isKeyframe())
    {
      // The feature detector should not initialize new seeds close to this location
      seg_feature_detector_->setGridOccpuancy(LineFeat(matcher_.px_cur_,matcherls_.px_cur_));
    }

    // if the seed has converged, we initialize a new candidate point and remove the seed
    if(sqrt(it->sigma2_s) < it->z_range_s/options_.seed_convergence_sigma2_thresh &&
       sqrt(it->sigma2_e) < it->z_range_e/options_.seed_convergence_sigma2_thresh  )
    {
      assert(static_cast<LineFeat*>(it->ftr)->feat3D == NULL); // TODO this should not happen anymore
      Vector3d xyz_world_s(it->ftr->frame->T_f_w_.inverse() * (static_cast<LineFeat*>(it->ftr)->sf * (1.0/it->mu_s)));
      Vector3d xyz_world_e(it->ftr->frame->T_f_w_.inverse() * (static_cast<LineFeat*>(it->ftr)->ef * (1.0/it->mu_e)));
      LineSeg* line = new LineSeg(xyz_world_s, xyz_world_e, it->ftr);
      static_cast<LineFeat*>(it->ftr)->feat3D = line;
      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new PointFeat(frame.get(), matcher_.px_cur_, matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
      {
        seed_converged_cb_ls_(line, it->sigma2_s, it->sigma2_e); // put in candidate list
      }
      it = seg_seeds_.erase(it);
    }
    else if( isnan(z_inv_min_s) || isnan(z_inv_min_e) )
    {
      SVO_WARN_STREAM("z_min_s or z_min_e is NaN");
      it = seg_seeds_.erase(it);
    }
    else
      ++it;
  }
}

void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<PointSeed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<PointSeed>::iterator it=pt_seeds_.begin(); it!=pt_seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

void DepthFilter::updatePointSeed(const float x, const float tau2, PointSeed* seed)
{
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if(std::isnan(norm_scale))
    return;
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  float s2 = 1./(1./seed->sigma2 + 1./tau2);
  float m = s2*(seed->mu/seed->sigma2 + x/tau2);
  float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
  float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
          + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

  // update parameters
  float mu_new = C1*m+C2*seed->mu;
  seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
  seed->mu = mu_new;
  seed->a = (e-f)/(f-e/f);
  seed->b = seed->a*(1.0f-f)/f;
}

void DepthFilter::updateLineSeed(const float x_s, const float tau2_s, const float x_e, const float tau2_e, LineSeed* seed)
{
  float norm_scale_s = sqrt(seed->sigma2_s + tau2_s);
  float norm_scale_e = sqrt(seed->sigma2_e + tau2_e);
  if(std::isnan(norm_scale_s)||std::isnan(norm_scale_e))
    return;
  boost::math::normal_distribution<float> nd_s(seed->mu_s, norm_scale_s);
  boost::math::normal_distribution<float> nd_e(seed->mu_e, norm_scale_e);

  /* update first endpoint of the line segment*/
  float s2 = 1./(1./seed->sigma2_s + 1./tau2_s);
  float m = s2*(seed->mu_s/seed->sigma2_s + x_s/tau2_s);
  float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd_s, x_s);
  float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range_s;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f_s = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e_s = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
            + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));
  // update parameters of first endpoint
  float mu_new_s = C1*m+C2*seed->mu_s;
  seed->sigma2_s = C1*(s2 + m*m) + C2*(seed->sigma2_s + seed->mu_s*seed->mu_s) - mu_new_s*mu_new_s;
  seed->mu_s = mu_new_s;

  /* update first endpoint of the line segment*/
  s2 = 1./(1./seed->sigma2_e + 1./tau2_e);
  m = s2*(seed->mu_e/seed->sigma2_e + x_e/tau2_e);
  C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd_e, x_e);
  C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range_e;
  normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f_e = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e_e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
            + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));
  // update parameters of first endpoint
  float mu_new_e = C1*m+C2*seed->mu_e;
  seed->sigma2_e = C1*(s2 + m*m) + C2*(seed->sigma2_e + seed->mu_e*seed->mu_e) - mu_new_e*mu_new_e;
  seed->mu_e = mu_new_e;

  /* update probability of inlier and outlier (a & b parameters of Beta distribution)*/
  float a_s = (e_s-f_s)/(f_s-e_s/f_s);
  float a_e = (e_e-f_e)/(f_e-e_e/f_e);
  //seed->a = sqrt(a_s*a_s+a_e*a_e);
  seed->a = std::max(a_s,a_e);

  float b_s = a_s*(1.f-f_s)/f_s;
  float b_e = a_e*(1.f-f_e)/f_e;
  //seed->b = sqrt(b_s*b_s+b_e*b_e);
  seed->b = std::min(b_s,b_e);

}

double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
  Vector3d t(T_ref_cur.translation());
  Vector3d a = f*z-t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f.dot(t)/t_norm); // dot product
  double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
  double beta_plus = beta + px_error_angle;
  double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
  double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}

} // namespace plsvo
