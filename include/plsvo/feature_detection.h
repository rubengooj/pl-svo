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


#ifndef SVO_FEATURE_DETECTION_H_
#define SVO_FEATURE_DETECTION_H_

#include <plsvo/global.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>

namespace plsvo {

/// Implementation of various feature detectors.
namespace feature_detection {

/// Temporary container used for corner detection. Features are initialized from these.
struct Corner
{
  int x;        //!< x-coordinate of corner in the image.
  int y;        //!< y-coordinate of corner in the image.
  int level;    //!< pyramid level of the corner.
  float score;  //!< shi-tomasi score of the corner.
  float angle;  //!< for gradient-features: dominant gradient angle.
  Corner(int x, int y, float score, int level, float angle) :
    x(x), y(y), level(level), score(score), angle(angle)
  {}
};
typedef vector<Corner> Corners;

/// All detectors should derive from this abstract class.
template <class FeatureT>
class AbstractDetector
{
public:
  AbstractDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~AbstractDetector() {}

  virtual void detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
      list<FeatureT*>& fts) {}
  // Default method to instantiate AbstractDetector as a void detector that detects nothing

  /// Flag the grid cell as occupied
  virtual void setGridOccpuancy(const FeatureT& ft) {} // Default does nothing

  /// Set grid cells of existing features as occupied
  virtual void setExistingFeatures(const list<FeatureT*>& fts) {} // Default does nothing

  const int img_width_;
  const int img_height_;

protected:

  static const int border_ = 8; //!< no feature should be within 8px of border.
  const int cell_size_;
  const int n_pyr_levels_;
  const int grid_n_cols_;
  const int grid_n_rows_;
  vector<bool> grid_occupancy_;

  void resetGrid();

  inline int getCellIndex(int x, int y, int level)
  {
    const int scale = (1<<level);
    return (scale*y)/cell_size_*grid_n_cols_ + (scale*x)/cell_size_;
  }
};
template<class FeatureT>
AbstractDetector<FeatureT>::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        img_width_(img_width),
        img_height_(img_height),
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

template<typename FeatureT>
using DetectorPtr = boost::shared_ptr<AbstractDetector<FeatureT>>;

template<class FeatureT>
void AbstractDetector<FeatureT>::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

/// FAST detector by Edward Rosten.
class FastDetector : public AbstractDetector<PointFeat>
{
public:
  FastDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~FastDetector() {}

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const double detection_threshold,
      list<PointFeat*>& fts);

  virtual void setGridOccpuancy(const PointFeat& ft);
  virtual void setExistingFeatures(const list<PointFeat*>& fts);
};

/// LSD detector from OpenCV.
class LsdDetector : public AbstractDetector<LineFeat>
{
public:
  LsdDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~LsdDetector() {}

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const double detection_threshold,
      list<LineFeat*>& fts);

  virtual void detect(
      Frame* frame,
      const cv::Mat& rec_img,
      const double detection_threshold,
      list<LineFeat*>& fts);

  virtual void setGridOccpuancy(const LineFeat& ft);
  virtual void setExistingFeatures(const list<LineFeat*>& fts);

protected:

  void resetGridLs(){
    std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
  }

};

} // namespace feature_detection
} // namespace plsvo

#endif // SVO_FEATURE_DETECTION_H_
