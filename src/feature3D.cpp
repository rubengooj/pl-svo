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
#include <vikit/math_utils.h>
#include <plsvo/feature3D.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
 
namespace plsvo {

Point::Point(const Vector3d& pos) :
  Feature3D(counter_++),
  pos_(pos),
  normal_set_(false),
  v_g2o_(NULL)
{}

Point::Point(const Vector3d& pos, PointFeat* ftr) :
  Feature3D(counter_++),
  pos_(pos),
  normal_set_(false),
  v_g2o_(NULL)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

LineSeg::LineSeg(const Vector3d& spos, const Vector3d& epos) :
  Feature3D(counter_++),
  spos_(spos),
  epos_(epos),
  v_g2o_(NULL)
{}

LineSeg::LineSeg(const Vector3d& spos, const Vector3d& epos, LineFeat* ftr) :
  Feature3D(counter_++),
  spos_(spos),
  epos_(epos),
  v_g2o_(NULL)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

void Point::initNormal()
{
  assert(!obs_.empty());
  const Feature* ftr = obs_.back();
  assert(ftr->frame != NULL);
  normal_ = ftr->frame->T_f_w_.rotation_matrix().transpose()*(-ftr->f);
  normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
  normal_set_ = true;
}

bool Point::getCloseViewObs(const Vector3d& framepos, Feature* &ftr) const
{
  // TODO: get frame with same point of view AND same pyramid level!
  Vector3d obs_dir(framepos - pos_); obs_dir.normalize();
  auto min_it=obs_.begin();
  double min_cos_angle = 0;
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    Vector3d dir((*it)->frame->pos() - pos_); dir.normalize();
    double cos_angle = obs_dir.dot(dir);
    if(cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }
  ftr = *min_it;
  if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
    return false;
  return true;
}

bool LineSeg::getCloseViewObs(const Vector3d& framepos, Feature *&ftr) const
{
  // TODO: get frame with same point of view AND same pyramid level!
  // For now, use angle central point to measure observation distance
  // Further rotation from that central point can be corrected by in-plane optimization?
  Vector3d cpos = 0.5*(spos_+epos_);
  Vector3d obs_dir(framepos - cpos); obs_dir.normalize();
  auto min_it=obs_.begin();
  double min_cos_angle = 0;
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    Vector3d dir((*it)->frame->pos() - cpos); dir.normalize();
    double cos_angle = obs_dir.dot(dir);
    if(cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }
  ftr = *min_it;
  if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
    return false;
  return true;
}

} // namespace plsvo


