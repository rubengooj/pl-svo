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
#include <stdexcept>
#include <plsvo/reprojector.h>
#include <plsvo/frame.h>
#include <plsvo/feature3D.h>
#include <plsvo/feature.h>
#include <plsvo/map.h>
#include <plsvo/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vikit/abstract_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>

namespace plsvo {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map) :
    map_(map)
{
  initializeGrid(cam);
}

Reprojector::~Reprojector()
{
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
  std::for_each(gridls_.cells.begin(), gridls_.cells.end(), [&](LineCandidates* c){ delete c; });
}

void Reprojector::initializeGrid(vk::AbstractCamera* cam)
{
  // Point features
  grid_.cell_size = Config::gridSize();
  grid_.grid_n_cols = ceil(static_cast<double>(cam->width())/grid_.cell_size);
  grid_.grid_n_rows = ceil(static_cast<double>(cam->height())/grid_.cell_size);
  grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
  grid_.cell_order.resize(grid_.cells.size());
  for(size_t i=0; i<grid_.cells.size(); ++i)
    grid_.cell_order[i] = i;
  random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
  // Line segment features
  if( Config::hasLines() )
  {
      gridls_.cell_size = Config::gridSizeSegs();
      gridls_.grid_n_cols = ceil(static_cast<double>(cam->width())/gridls_.cell_size);
      gridls_.grid_n_rows = ceil(static_cast<double>(cam->height())/gridls_.cell_size);
      gridls_.cells.resize(gridls_.grid_n_cols*gridls_.grid_n_rows);
      std::for_each(gridls_.cells.begin(), gridls_.cells.end(), [&](LineCandidates*& c){ c = new LineCandidates; });
      gridls_.cell_order.resize(gridls_.cells.size());
      for(size_t i=0; i<gridls_.cells.size(); ++i)
        gridls_.cell_order[i] = i;
      random_shuffle(gridls_.cell_order.begin(), gridls_.cell_order.end()); // maybe we should do it at every iteration!
  }
}

void Reprojector::resetReprojGrid()
{
  n_matches_    = 0;
  n_trials_     = 0;
  n_ls_matches_ = 0;
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
  if(Config::hasLines()) std::for_each(gridls_.cells.begin(), gridls_.cells.end(), [&](LineCandidates* c){ c->clear(); });
}

template<class FeatureT>
int Reprojector::setKfCandidates(FramePtr frame, list<FeatureT*> fts)
{
  int candidate_counter = 0;
  for(auto it=fts.begin(), ite_ftr=fts.end(); it!=ite_ftr; ++it)
  {
    // check if the feature has a 3D object assigned
    if((*it)->feat3D == NULL)
      continue;
    // make sure we project a point only once
    if((*it)->feat3D->last_projected_kf_id_ == frame->id_)
      continue;
    (*it)->feat3D->last_projected_kf_id_ = frame->id_;
    if(reproject(frame, (*it)->feat3D))
      // increment the number of candidates taken successfully
      candidate_counter++;
  }
  return candidate_counter;
}

template<class MapCandidatesT>
void Reprojector::setMapCandidates(FramePtr frame, MapCandidatesT &map_candidates)
{
  boost::unique_lock<boost::mutex> lock(map_candidates.mut_); // the mutex will be unlocked when out of scope
  auto it=map_candidates.candidates_.begin();
  while(it!=map_candidates.candidates_.end())
  {
    if(!reproject(frame, it->first))
    {
      // if the reprojection of the map candidate point failed,
      // increment the counter of failed reprojections (assess the point quality)
      it->first->n_failed_reproj_ += 3;
      if(it->first->n_failed_reproj_ > 30)
      {
        // if the reprojection failed too many times, remove the map candidate point
        map_candidates.deleteCandidate(*it);
        it = map_candidates.candidates_.erase(it);
        continue;
      }
    }
    ++it;
  } // end-while-loop
}

void Reprojector::reprojectMap(
    FramePtr frame,
    std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
{
  resetReprojGrid();

  // Reset candidate lines: TODO in the future this could be fused into resetGrid()
  n_ls_matches_ = 0;
  //lines_.clear();

  // Identify those Keyframes which share a common field of view.
  SVO_START_TIMER("reproject_kfs");
  list< pair<FramePtr,double> > close_kfs;
  map_.getCloseKeyframes(frame, close_kfs);

  // Sort KFs with overlap according to their closeness (2nd value of pairs in the list)
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  // Reproject all map features of the closest N kfs with overlap.
  size_t n_kfs = 0;
  overlap_kfs.reserve(options_.max_n_kfs);
  for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
      it_frame!=ite_frame && n_kfs<options_.max_n_kfs; ++it_frame, ++n_kfs)
  {
    FramePtr ref_frame = it_frame->first;
    // add the current frame to the (output) list of keyframes with overlap
    // initialize the counter of candidates from this frame (2nd element in pair) to zero
    overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));
    // Consider for candidate each mappoint in the ref KF that the current (input) KF observes
    // We only store in which grid cell the points fall.
    // Add each corresponding valid new Candidate to its cell in the grid.
    int num_pt_success = setKfCandidates( frame, ref_frame->pt_fts_ );
    overlap_kfs.back().second += num_pt_success;
    // Add each line segment in the ref KF that the current (input) KF observes
    int num_seg_success = setKfCandidates( frame, ref_frame->seg_fts_ );
    overlap_kfs.back().second += num_seg_success;
  }
  SVO_STOP_TIMER("reproject_kfs");

  // Now project all map candidates
  // (candidates in the map are created from converged seeds)
  SVO_START_TIMER("reproject_candidates");
  // Point candidates
  // (same logic as above to populate the cell grid but taking candidate points from the map object)
  setMapCandidates(frame, map_.point_candidates_);
  // Segment candidates
  setMapCandidates(frame, map_.segment_candidates_);
  SVO_STOP_TIMER("reproject_candidates");

  // Now we go through each grid cell and select one point to match.
  // At the end, we should have at maximum one reprojected point per cell.
  SVO_START_TIMER("feature_align");
  for(size_t i=0; i<grid_.cells.size(); ++i)
  {
    // we prefer good quality points over unkown quality (more likely to match)
    // and unknown quality over candidates (position not optimized)
    // we use the random cell order to visit cells uniformly on the grid
    if(refineBestCandidate(*grid_.cells.at(grid_.cell_order[i]), frame))
      ++n_matches_;
    if(n_matches_ > (size_t) Config::maxFts())
      break; // the random visit order of cells assures uniform distribution
             // of the features even if we break early (maxFts reached soon)
  }

  // Try to refine every segment candidate
  for(size_t i=0; i<gridls_.cells.size(); ++i)
  {
    if(refineBestCandidate(*gridls_.cells.at(gridls_.cell_order[i]), frame))
      ++n_ls_matches_;
    if(n_ls_matches_ > (size_t) Config::maxFtsSegs())
      break; // the random visit order of cells assures uniform distribution
             // of the features even if we break early (maxFts reached soon)
  }
  /*for(auto it = lines_.begin(), ite = lines_.end(); it!=ite; ++it)
  {
    if(refine(it->ls,it->spx,it->epx,frame))
      ++n_ls_matches_;
    if(n_ls_matches_ > (size_t) Config::maxFtsSegs())
      break;
  }*/
  SVO_STOP_TIMER("feature_align");
}

bool Reprojector::pointQualityComparator(PointCandidate& lhs, PointCandidate& rhs)
{
  // point quality is given by the Point::PointType enum
  // so that DELETED < CANDIDATE < UNKNOWN < GOOD
  if(lhs.pt->type_ > rhs.pt->type_)
    return true;
  return false;
}

bool Reprojector::lineQualityComparator(LineCandidate& lhs, LineCandidate& rhs)
{
  // DELETED < CANDIDATE < UNKNOWN < GOOD
  if(lhs.ls->type_ > rhs.ls->type_)
    return true;
  return false;
}

bool Reprojector::refineBestCandidate(Cell& cell, FramePtr frame)
{
  // sort the candidates inside the cell according to their quality
  cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
  Cell::iterator it=cell.begin();
  // in principle, iterate through the whole list of features in the cell
  // in reality, there is maximum one point per cell, so the loop returns if successful
  while(it!=cell.end())
  {
    ++n_trials_;
    // Try to refine the point feature in frame from current initial estimate
    bool success = refine( it->pt, it->px, frame );
    // Failed or not, this candidate was finally being erased in original code
    it = cell.erase(it); // it takes next position in the list as output of .erase
    if(success)
      // Maximum one point per cell.
      return true;
  }
  return false;
}

bool Reprojector::refineBestCandidate(LineCandidates& cell, FramePtr frame)
{
  // sort the candidates inside the cell according to their quality
  cell.sort(boost::bind(&Reprojector::lineQualityComparator, _1, _2));
  LineCandidates::iterator it=cell.begin();
  // in principle, iterate through the whole list of features in the cell
  // in reality, there is maximum one point per cell, so the loop returns if successful
  while(it!=cell.end())
  {
    ++n_trials_;
    // Try to refine the point feature in frame from current initial estimate
    bool success = refine( it->ls, it->spx, it->epx, frame );
    // Failed or not, this candidate was finally being erased in original code
    it = cell.erase(it); // it takes next position in the list as output of .erase
    if(success)
      // Maximum one point per cell.
      return true;
  }
  return false;
}

bool Reprojector::refine(Point* pt, Vector2d& px_est, FramePtr frame)
{
  if(pt->type_ == Point::TYPE_DELETED)
    return false;

  bool found_match = true;
  if(options_.find_match_direct)
    // refine px position in the candidate by directly applying subpix refinement
    // internally, it is optimizing photometric error
    // of the candidate px patch wrt the closest-view reference feature patch
    found_match = matcher_.findMatchDirect(*pt, *frame, px_est);
  // TODO: What happens if options_.find_match_direct is false??? Shouldn't findEpipolarMatchDirect be here?

  // apply quality logic
  {
    if(!found_match)
    {
      // if there is no match found for this point, decrease quality
      pt->n_failed_reproj_++;
      // consider removing the point from map depending on point type and quality
      if(pt->type_ == Point::TYPE_UNKNOWN && pt->n_failed_reproj_ > 15)
        map_.safeDeletePoint(pt);
      if(pt->type_ == Point::TYPE_CANDIDATE  && pt->n_failed_reproj_ > 30)
        map_.point_candidates_.deleteCandidatePoint(pt);
      return false;
    }
    // if there is successful match found for this point, increase quality
    pt->n_succeeded_reproj_++;
    if(pt->type_ == Point::TYPE_UNKNOWN && pt->n_succeeded_reproj_ > 10)
      pt->type_ = Point::TYPE_GOOD;
  }

  // create new point feature for this frame with the refined (aligned) candidate position in this image
  PointFeat* new_feature = new PointFeat(frame.get(), px_est, matcher_.search_level_);
  frame->addFeature(new_feature);

  // Here we add a reference in the feature to the 3D point, the other way
  // round is only done if this frame is selected as keyframe.
  // TODO: why not give it directly to the constructor PointFeat(frame.get(), pt, it->px, matcher_.serach_level_)
  new_feature->feat3D = pt;

  PointFeat* pt_ftr = static_cast<PointFeat*>( matcher_.ref_ftr_ );
  if(pt_ftr != NULL)
  {
    if(pt_ftr->type == PointFeat::EDGELET)
    {
      new_feature->type = PointFeat::EDGELET;
      new_feature->grad = matcher_.A_cur_ref_*pt_ftr->grad;
      new_feature->grad.normalize();
    }
  }

  // If the keyframe is selected and we reproject the rest, we don't have to
  // check this point anymore.
//  it = cell.erase(it);

  // Maximum one point per cell.
  return true;
}

bool Reprojector::refine(LineSeg* ls, Vector2d& spx_est, Vector2d& epx_est, FramePtr frame)
{

  if(ls->type_ == LineSeg::TYPE_DELETED)
    return false;

  bool found_match = true;
  if(options_.find_match_direct)
  {
    // refine start and end points in the segment independently
    found_match = matcher_.findMatchDirect(*ls, *frame, spx_est, epx_est);
  }
  // TODO: What happens if options_.find_match_direct is false??? Shouldn't findEpipolarMatchDirect be here?

  // apply quality logic - don't like the n_failed_reproj and n_succeeded_reproj_ numbers
  {
    if(!found_match)
    {
      // if there is no match found for this point, decrease quality
      ls->n_failed_reproj_++;
      // consider removing the point from map depending on point type and quality
      if(ls->type_ == LineSeg::TYPE_UNKNOWN && ls->n_failed_reproj_ > 15)
        map_.safeDeleteSegment(ls);
      if(ls->type_ == LineSeg::TYPE_CANDIDATE && ls->n_failed_reproj_ > 30)
        map_.segment_candidates_.deleteCandidateSegment(ls);
      return false;
    }
    // if there is successful match found for this point, increase quality
    ls->n_succeeded_reproj_++;
    if(ls->type_ == LineSeg::TYPE_UNKNOWN && ls->n_succeeded_reproj_ > 10){
      ls->type_ = LineSeg::TYPE_GOOD;
    }
  }

  // create new segment feature for this frame with the refined (aligned) candidate position in this image
  LineFeat* new_feature = new LineFeat(frame.get(), spx_est, epx_est, matcher_.search_level_);
  frame->addFeature(new_feature);

  // Here we add a reference in the feature to the 3D point, the other way
  // round is only done if this frame is selected as keyframe.
  // TODO: why not give it directly to the constructor PointFeat(frame.get(), pt, it->px, matcher_.serach_level_)
  new_feature->feat3D = ls;

  // If the keyframe is selected and we reproject the rest, we don't have to
  // check this point anymore.
  //  it = cell.erase(it);

  // Maximum one point per cell.
  return true;
}

bool Reprojector::reproject(FramePtr frame, Point* point)
{
  // get position in current frame image of the world 3D point
  Vector2d cur_px(frame->w2c(point->pos_));
  if(frame->cam_->isInFrame(cur_px.cast<int>(), 8)) // 8px is the patch size in the matcher
  {
    // get linear index (wrt row-wise vectorized grid matrix)
    // of the image grid cell in which the point px lies
    const int k = static_cast<int>(cur_px[1]/grid_.cell_size)*grid_.grid_n_cols
                + static_cast<int>(cur_px[0]/grid_.cell_size);
    grid_.cells.at(k)->push_back(PointCandidate(point, cur_px));
    return true;
  }
  return false;
}

bool Reprojector::reproject(FramePtr frame, LineSeg* segment)
{
  // get position in current frame image of the world 3D point
  Vector2d cur_spx(frame->w2c(segment->spos_));
  Vector2d cur_epx(frame->w2c(segment->epos_));
  if(frame->cam_->isInFrame(cur_spx.cast<int>(), 8) &&
     frame->cam_->isInFrame(cur_epx.cast<int>(), 8)) // 8px is the patch size in the matcher
  {
    const int sk = static_cast<int>(cur_spx[1]/gridls_.cell_size)*gridls_.grid_n_cols
                 + static_cast<int>(cur_spx[0]/gridls_.cell_size);
    const int ek = static_cast<int>(cur_epx[1]/gridls_.cell_size)*gridls_.grid_n_cols
                 + static_cast<int>(cur_epx[0]/gridls_.cell_size);
    // in the case of segments, add the candidate to a common list for the whole image
    gridls_.cells.at(sk)->push_back(LineCandidate(segment, cur_spx, cur_epx));
    gridls_.cells.at(ek)->push_back(LineCandidate(segment, cur_spx, cur_epx));
    return true;
  }
  return false;
}

} // namespace plsvo
