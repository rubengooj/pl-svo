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


#include <stdexcept>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <plsvo/feature3D.h>
#include <plsvo/config.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
#include <fast/fast.h>

namespace plsvo {

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
    id_(frame_counter_++),
    timestamp_(timestamp),
    cam_(cam),
    key_pts_(5),
    is_keyframe_(false),
    v_kf_(NULL)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(pt_fts_.begin(), pt_fts_.end(), [&](Feature* i){delete i;});
}

void Frame::initFrame(const cv::Mat& img)
{
  // check image
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to NULL
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

  // Build Image Pyramid
  frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;
  setKeyPoints();
}

void Frame::addFeature(PointFeat *ftr)
{
  pt_fts_.push_back(ftr);
}
void Frame::addFeature(LineFeat *ftr)
{
  seg_fts_.push_back(ftr);
}

void Frame::setKeyPoints()
{
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->feat3D == NULL)
        key_pts_[i] = NULL;

  std::for_each(pt_fts_.begin(), pt_fts_.end(), [&](PointFeat* ftr){ if(ftr->feat3D != NULL) checkKeyPoints(ftr); });
}

void Frame::checkKeyPoints(PointFeat* ftr)
{
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;

  // center pixel
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;

  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
      key_pts_[2] = ftr;
  }
  if(ftr->px[0] < cv && ftr->px[1] < cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }
  if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[4]->px[0]-cu) * (key_pts_[4]->px[1]-cv))
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(PointFeat* ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](PointFeat*& i){
    if(i == ftr) {
      i = NULL;
      found = true;
    }
  });
  if(found)
    setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w;
  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);
  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}


/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}

bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.pt_fts_.size());
  depth_min = std::numeric_limits<double>::max();
  // Points
  for(auto it=frame.pt_fts_.begin(), ite=frame.pt_fts_.end(); it!=ite; ++it)
  {
    if((*it)->feat3D != NULL)
    {
      const double z = frame.w2f((*it)->feat3D->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min);
    }
  }
  // Line segments
  for(auto it=frame.seg_fts_.begin(), ite=frame.seg_fts_.end(); it!=ite; ++it)
  {
    if((*it)->feat3D != NULL)
    {
      const double zs = frame.w2f((*it)->feat3D->spos_).z();
      const double ze = frame.w2f((*it)->feat3D->epos_).z();
      depth_vec.push_back(zs);
      depth_vec.push_back(ze);
      depth_min = fmin(zs, depth_min);
      depth_min = fmin(ze, depth_min);
    }
  }
  if(depth_vec.empty())
  {
    SVO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
    return false;
  }
  depth_mean = vk::getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace plsvo
