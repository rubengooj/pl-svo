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


#include <set>
#include <plsvo/map.h>
#include <plsvo/feature3D.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <boost/bind.hpp>

namespace plsvo {

Map::Map() {}

Map::~Map()
{
  reset();
  SVO_INFO_STREAM("Map destructed");
}

void Map::reset()
{
  keyframes_.clear();
  point_candidates_.reset();
  segment_candidates_.reset();
  emptyTrash();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    if(*it == frame)
    {
      std::for_each((*it)->pt_fts_.begin(), (*it)->pt_fts_.end(), [&](PointFeat* ftr){
        removePtFrameRef(it->get(), ftr);
      });
      std::for_each((*it)->seg_fts_.begin(), (*it)->seg_fts_.end(), [&](LineFeat* ftr){
        removeLsFrameRef(it->get(), ftr);
      });

      keyframes_.erase(it);
      found = true;
      break;
    }
  }

  point_candidates_.removeFrameCandidates(frame);

  if(found)
    return true;

  SVO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

void Map::removePtFrameRef(Frame* frame, PointFeat* ftr)
{
  if(ftr->feat3D == NULL)
    return; // mappoint may have been deleted in a previous ref. removal
  Point* pt = ftr->feat3D;
  ftr->feat3D = NULL;
  if(pt->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeletePoint(pt);
    return;
  }
  pt->deleteFrameRef(frame);  // Remove reference from map_point
  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe
}

void Map::removeLsFrameRef(Frame* frame, LineFeat* ftr)
{
  if(ftr->feat3D == NULL)
    return; // map segment may have been deleted in a previous ref. removal
  LineSeg* ls = ftr->feat3D;
  ftr->feat3D = NULL;
  if(ls->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeleteSegment(ls);
    return;
  }
  ls->deleteFrameRef(frame);  // Remove reference from map_point
  // TODO: For now, there are no KeySegments in the frame (only points are used to check overlapping
  //  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe

}

void Map::safeDeletePoint(Point* pt)
{
  // Delete references to mappoints in all keyframes
  std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](PointFeat* ftr){
    ftr->feat3D=NULL;
    ftr->frame->removeKeyPoint(ftr);
  });
  pt->obs_.clear();

  // Delete mappoint
  deletePoint(pt);
}

void Map::safeDeleteSegment(LineSeg* ls)
{
  // Delete references to mappoints in all keyframes
  std::for_each(ls->obs_.begin(), ls->obs_.end(), [&](LineFeat* ftr){
    ftr->feat3D=NULL;
    //ftr_->frame->removeKeyPoint(ftr); // TODO: There are no "KeySegments" in frame
  });
  ls->obs_.clear();
  // Delete mappoint
  deleteSegment(ls);
}

void Map::deletePoint(Point* pt)
{
  pt->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(pt);
}

void Map::deleteSegment(LineSeg* ls)
{
  ls->type_ = LineSeg::TYPE_DELETED;
  trash_segments_.push_back(ls);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  keyframes_.push_back(new_keyframe);
}

void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr,double> >& close_kfs) const
{
  for(auto kf : keyframes_)
  {
    // check if kf has overlaping field of view with frame, use therefore KeyPoints
    for(auto keypoint : kf->key_pts_)
    {
      if(keypoint == nullptr)
        continue;
      if(frame->isVisible(keypoint->feat3D->pos_))
      {
        // store a pair formed by the keyframe pointer and the distance between keyframes
        close_kfs.push_back(
            std::make_pair(
                kf, (frame->T_f_w_.translation()-kf->T_f_w_.translation()).norm()));
        break; // this keyframe has an overlapping field of view -> add to close_kfs
      }
    }
  }
}

FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
  list< pair<FramePtr,double> > close_kfs;
  getCloseKeyframes(frame, close_kfs);
  if(close_kfs.empty())
  {
    return nullptr;
  }


  // Sort KFs with overlap according to their closeness
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  if(close_kfs.front().first != frame)
    return close_kfs.front().first;
  close_kfs.pop_front();
  return close_kfs.front().first;
}

FramePtr Map::getFurthestKeyframe(const Vector3d& pos) const
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
  }
  return furthest_kf;
}

bool Map::getKeyframeById(const int id, FramePtr& frame) const
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    if((*it)->id_ == id) {
      found = true;
      frame = *it;
      break;
    }
  return found;
}

void Map::transform(const Matrix3d& R, const Vector3d& t, const double& s)
{
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    Vector3d pos = s*R*(*it)->pos() + t;
    Matrix3d rot = R*(*it)->T_f_w_.rotation_matrix().inverse();
    (*it)->T_f_w_ = SE3(rot, pos).inverse();
    // Key Points
    for(auto ftr=(*it)->pt_fts_.begin(); ftr!=(*it)->pt_fts_.end(); ++ftr)
    {
      if((*ftr)->feat3D == NULL)
        continue;
      if((*ftr)->feat3D->last_published_ts_ == -1000)
        continue;
      (*ftr)->feat3D->last_published_ts_ = -1000;
      (*ftr)->feat3D->pos_ = s*R*(*ftr)->feat3D->pos_ + t;
    }
    // Line Segments
    for(auto ftr=(*it)->seg_fts_.begin(); ftr!=(*it)->seg_fts_.end(); ++ftr)
    {
      if((*ftr)->feat3D == NULL)
        continue;
      if((*ftr)->feat3D->last_published_ts_ == -1000)
        continue;
      (*ftr)->feat3D->last_published_ts_ = -1000;
      (*ftr)->feat3D->spos_ = s*R*(*ftr)->feat3D->spos_ + t;
      (*ftr)->feat3D->epos_ = s*R*(*ftr)->feat3D->epos_ + t;
    }
  }
}

void Map::emptyTrash()
{
  // Key Points
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& pt){
    delete pt;
    pt=NULL;
  });
  trash_points_.clear();
  point_candidates_.emptyTrash();
  // Line Segments
  std::for_each(trash_segments_.begin(), trash_segments_.end(), [&](LineSeg*& ls){
    delete ls;
    ls=NULL;
  });
  trash_segments_.clear();
  segment_candidates_.emptyTrash();
}

MapPointCandidates::MapPointCandidates()
{}

MapPointCandidates::~MapPointCandidates()
{
  reset();
}

void MapPointCandidates::newCandidatePoint(Point* point, double depth_sigma2)
{
  point->type_ = Point::TYPE_CANDIDATE;
  boost::unique_lock<boost::mutex> lock(mut_);
  candidates_.push_back(PointCandidate(point, point->obs_.front()));
}

void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  PointCandidateList::iterator it=candidates_.begin();
  while(it != candidates_.end())
  {
    if(it->first->obs_.front()->frame == frame.get())
    {
      // insert feature in the frame
      it->first->type_ = Point::TYPE_UNKNOWN;
      it->first->n_failed_reproj_ = 0;
      it->second->frame->addFeature(it->second);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

bool MapPointCandidates::deleteCandidatePoint(Point* point)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
  {
    if(it->first == point)
    {
      deleteCandidate(*it);
      candidates_.erase(it);
      return true;
    }
  }
  return false;
}

void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      deleteCandidate(*it);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

void MapPointCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
    delete c.first;
    delete c.second;
  });
  candidates_.clear();
}

void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
  // camera-rig: another frame might still be pointing to the candidate point
  // therefore, we can't delete it right now.
  delete c.second; c.second=NULL;
  c.first->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(c.first);
}

void MapPointCandidates::emptyTrash()
{
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& p){
    delete p; p=NULL;
  });
  trash_points_.clear();
}

MapSegmentCandidates::MapSegmentCandidates()
{}

MapSegmentCandidates::~MapSegmentCandidates()
{
  reset();
}

void MapSegmentCandidates::newCandidateSegment(LineSeg* ls, double depth_sigma2_s, double depth_sigma2_e)
{
  ls->type_ = LineSeg::TYPE_CANDIDATE;
  boost::unique_lock<boost::mutex> lock(mut_);
  candidates_.push_back(SegmentCandidate(ls, ls->obs_.front()));
}

void MapSegmentCandidates::addCandidateSegmentToFrame(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  SegmentCandidateList::iterator it=candidates_.begin();
  while(it != candidates_.end())
  {
    if(it->first->obs_.front()->frame == frame.get())
    {
      // insert feature in the frame
      it->first->type_ = LineSeg::TYPE_UNKNOWN;
      it->first->n_failed_reproj_ = 0;
      it->second->frame->addFeature(it->second);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

bool MapSegmentCandidates::deleteCandidateSegment(LineSeg* ls)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
  {
    if(it->first == ls)
    {
      deleteCandidate(*it);
      candidates_.erase(it);
      return true;
    }
  }
  return false;
}

void MapSegmentCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      deleteCandidate(*it);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

void MapSegmentCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](SegmentCandidate& c){
    delete c.first;
    delete c.second;
  });
  candidates_.clear();
}

void MapSegmentCandidates::deleteCandidate(SegmentCandidate& c)
{
  // camera-rig: another frame might still be pointing to the candidate point
  // therefore, we can't delete it right now.
  delete c.second; c.second=NULL;
  c.first->type_ = LineSeg::TYPE_DELETED;
  trash_segments_.push_back(c.first);
}

void MapSegmentCandidates::emptyTrash()
{
  std::for_each(trash_segments_.begin(), trash_segments_.end(), [&](LineSeg*& p){
    delete p; p=NULL;
  });
  trash_segments_.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
  for(auto it = frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D==NULL)
      continue;

    if((*it)->feat3D->type_ == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

    if(!(*it)->feat3D->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

    pointValidation((*it)->feat3D, id);
  }
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->feat3D == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(Point* point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->pt_fts_.begin(); it_ftr!=(*it)->frame->pt_fts_.end(); ++it_ftr)
     if((*it_ftr)->feat3D == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

void mapStatistics(Map* map)
{
  // compute average number of features which each frame observes
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  // compute average number of observations that each point has
  size_t n_frame_obs(0);
  size_t n_pts(0);
  std::set<Point*> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->pt_fts_.begin(); ftr!=(*it)->pt_fts_.end(); ++ftr)
    {
      if((*ftr)->feat3D == NULL)
        continue;
      if(points.insert((*ftr)->feat3D).second) {
        ++n_pts;
        n_frame_obs += (*ftr)->feat3D->nRefs();
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace plsvo
