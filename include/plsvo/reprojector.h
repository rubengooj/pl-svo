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

#ifndef SVO_REPROJECTION_H_
#define SVO_REPROJECTION_H_

#include <plsvo/global.h>
#include <plsvo/matcher.h>

namespace vk {
class AbstractCamera;
}

namespace plsvo {

class Map;
class Point;
class LineSeg;

/// Project points from the map into the image and find the corresponding
/// feature (corner). We don't search a match for every point but only for one
/// point per cell. Thereby, we achieve a homogeneously distributed set of
/// matched features and at the same time we can save processing time by not
/// projecting all points.
class Reprojector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Reprojector config parameters
  struct Options {
    size_t max_n_kfs;   //!< max number of keyframes to reproject from
    bool find_match_direct;
    Options()
    : max_n_kfs(10),
      find_match_direct(true)
    {}
  } options_;

  size_t n_matches_;
  size_t n_trials_;
  size_t n_ls_matches_;

  Reprojector(vk::AbstractCamera* cam, Map& map);

  ~Reprojector();

  /// Project points from the map into the image.
  /// First find keyframes with overlapping field of view
  /// and project only those map-points.
  void reprojectMap(
      FramePtr frame,
      std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs); // pairs contain Frame ptr and number of candidates found from that frame

private:

  /// A candidate is a feature that projects into the image plane
  /// and for which we will search a maching feature in the image.
  //  template<class FeatureT>
  //  struct Candidate
  //  {

  //  };

  struct PointCandidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point* pt;       //!< 3D point.
    Vector2d px;     //!< projected 2D pixel location.
    PointCandidate(Point* pt, Vector2d& px) : pt(pt), px(px) {}
  };
  // A cell is just a list of Reprojector::Candidate
  typedef std::list<PointCandidate, aligned_allocator<PointCandidate> > Cell;
  typedef std::vector<Cell*> CandidateGrid;

  struct LineCandidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSeg* ls;       //!< 3D point.
    Vector2d spx;
    Vector2d epx;
    LineCandidate(LineSeg* ls, Vector2d& spx, Vector2d& epx) : ls(ls), spx(spx), epx(epx) {}
  };
  /// The candidate segments are collected in a single list for the whole image.
  /// There is no clear heuristic about how to discretize the image space for the segment case.
  typedef std::list<LineCandidate, aligned_allocator<LineCandidate> > LineCandidates;
  typedef std::vector<LineCandidates*> LineCandidateGrid;

  /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
  struct Grid
  {
    CandidateGrid cells;
    vector<int> cell_order;
    int cell_size;
    int grid_n_cols;
    int grid_n_rows;
  };

  struct GridLs
  {
    LineCandidateGrid cells;
    vector<int> cell_order;
    int cell_size;
    int grid_n_cols;
    int grid_n_rows;
  };

  Grid    grid_;
  GridLs  gridls_;
  Matcher matcher_;
  Map&    map_;

  // Comparison operator (candidate1 > candidate2)
  // that returns quality(cand1) > quality(cand2)
  static bool pointQualityComparator(PointCandidate& lhs, PointCandidate& rhs);
  static bool lineQualityComparator(LineCandidate& lhs, LineCandidate& rhs);
  void initializeGrid(vk::AbstractCamera* cam);
  void resetReprojGrid();
  /// Get candidates for refinement from the features in an overlapping keyframe
  template<class FeatureT>
  int setKfCandidates(FramePtr frame, list<FeatureT*> fts);
  /// Get candidates for refinement from the converged seeds in the map
  template<class MapCandidatesT>
  void setMapCandidates(FramePtr frame, MapCandidatesT& map_candidates);
  /// Add a new Reprojector::Candidate in the cell of the frame (if any) within which point projects
  bool reproject(FramePtr frame, Point* point);
  /// Add a new segment candidate to refine
  bool reproject(FramePtr frame, LineSeg* segment);
  /// Find highest quality candidate that matches to its closest reference feature observation
  /// and adds the refined feature to input (current) frame
  bool refineBestCandidate(Cell& cell, FramePtr frame);
  bool refineBestCandidate(LineCandidates& cell, FramePtr frame);
  /// Refine feature in frame (taking previous view as reference)
//  template<class FeatureT> // TODO
  bool refine(Point* pt, Vector2d& px_est, FramePtr frame);
  /// Refine line segment and in case of success, add it to frame as feature
  bool refine(LineSeg* pt, Vector2d& spx_est, Vector2d& epx_est, FramePtr frame);
};

} // namespace plsvo

#endif // SVO_REPROJECTION_H_
