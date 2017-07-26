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


#ifndef SVO_FEATURE_H_
#define SVO_FEATURE_H_

#include <plsvo/frame.h>

namespace plsvo {

/// A salient image region that is tracked across frames.
/// This class is abstract and should not be instantiated except for pointers
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Frame* frame;         //!< Pointer to frame in which the feature was detected.
  Vector2d px;          //!< Any feature must have a center or origin. Coordinates in pixels on pyramid level 0.
  Vector3d f;           //!< Unit-bearing vector of the feature center.
  int level;            //!< Image pyramid level where feature was extracted.

  Feature(const Vector2d& _px);
  Feature(Frame* _frame, const Vector2d& _px, int _level);
  Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level);
  virtual ~Feature() = 0; // the pure virtual destructor makes this class abstract
};
inline Feature::~Feature() {} // default destructor for inheriting classes

/// Point Feature in 2D
struct PointFeat : public Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {
    CORNER,
    EDGELET,
  };

  FeatureType type;     //!< Type can be corner or edgelet.
  Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.

  Point* feat3D;        //!< Pointer to 3D point which corresponds to the point feature.

  // Constructors
  PointFeat(const Vector2d& _px);
  PointFeat(Frame* _frame, const Vector2d& _px, int _level);
  PointFeat(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level);
  PointFeat(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level);
};

/// Segment Feature in 2D
struct LineFeat : public Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {
    LINE_SEGMENT
  };

  FeatureType type;     //!< Type can be line_segment only (for now)
  Vector2d spx;         //!< Start PiXel: Coordinates of start point in pixels on pyramid level 0.
  Vector2d epx;         //!< End PiXel: Coordinates of end point in pixels on pyramid level 0.
  Vector3d sf;          //!< Unit-bearing vector of the start point.
  Vector3d ef;          //!< Unit-bearing vector of the end point
  Vector3d line;        //!< Normalized vector of the line segment observation
  LineSeg* feat3D;      //!< Pointer to 3D line segment which corresponds to the feature.
  Vector2d grad;        //!< Dominant gradient direction for edgelets and segments, normalized.
  double length;         //!< Line segment length
  double angle;

  // Constructors
  LineFeat(const Vector2d& _spx, const Vector2d& _epx);
  LineFeat(Frame* _frame, const Vector2d& _spx, const Vector2d& _epx, int _level);
  LineFeat(Frame* _frame, const Vector2d& _spx, const Vector2d& _epx, int _level, double angle);
  LineFeat(Frame* _frame, const Vector2d& _spx, const Vector2d& _epx, const Vector3d& _sf, const Vector3d& _ef, int _level);
  LineFeat(Frame* _frame, LineSeg* _ls, const Vector2d& _spx, const Vector2d& _epx, const Vector3d& _sf, const Vector3d& _ef, int _level);

  // Sample points uniformly through a segment
  size_t setupSampling(size_t patch_size, Vector2d& dif);
};

/// Patch class for a image region associated to a certain feature
struct Patch
{
  // patch geometry
  int size;
  int halfsize;
  int area;
  int border;
  cv::Rect rect; // higher-level geometric object defining the patch
                 // this <int> rectangle is extended to contain the subpixel rectangle (float)

  // patch position
  float u_ref, v_ref;
  Vector2i pos_ref_i;
  int u_ref_i, v_ref_i;
  // bilateral interpolation weights
  float wTL, wTR, wBL, wBR;

  // image parameters
  cv::Mat full_img; // the complete input image
  cv::Mat roi; // the interest block in the image (Region Of Interest)
  int stride; // stride to jump between row indeces in vectorized image

  // level and scale remain as sth external
  Patch() {} // empty constructor
  Patch( int _size, const cv::Mat& _img );

  /// Set exact and floor position of the patch reference (center)
  void setPosition( const Vector2d& px );
  /// Compute bilateral interpolation weights for a certain location in the image with subpixel precision
  void computeInterpWeights();
  /// Set interface objects for the region of interest inside the image
  void setRoi();
  inline bool isInFrame( int boundary=0 )
  {
    // TODO: use cv::Rect to check if patch is contained? See answer http://stackoverflow.com/a/32324568
    // TODO: invert boolean operations to positive
    return !( u_ref_i < boundary || v_ref_i < boundary || u_ref_i >= full_img.cols-boundary || v_ref_i >= full_img.rows-boundary );
  }
  uchar* begin( int i );
  uchar* end( int i );
};

struct RotatedRectPatch
{
  // patch geometry
  float length;
  float width;
  float angle;
  int area;
  float bSize_x,bSize_y;
  cv::RotatedRect rect; // higher-level geometric object defining the patch
                        // this <int> rectangle is extended to contain the subpixel rectangle (float)
  cv::Rect2f brect; // bounding rectangle aligned with axes

  // patch position
  float u_ref, v_ref;
  Vector2i pos_ref_i;
  int u_ref_i, v_ref_i;
  // bilateral interpolation weights
  float wTL, wTR, wBL, wBR;

  // image parameters
  cv::Mat full_img; // the complete input image
  cv::Mat roi; // the interest block in the image (Region Of Interest)
  int stride; // stride to jump between row indeces in vectorized image

  // level and scale remain as sth external
  RotatedRectPatch() {} // empty constructor
  RotatedRectPatch( float length, float width, float angle, const cv::Mat& _img ); // angle in rad for now?

  /// Set exact and floor position of the patch reference (center)
  void setPosition( const Vector2d& px );
  /// Compute bilateral interpolation weights for a certain location in the image with subpixel precision
  void computeInterpWeights();
  /// Set interface objects for the region of interest inside the image
  void setRoi();
  inline bool isInFrame( int boundary=0 )
  {
    // TODO: use cv::Rect to check if patch is contained? See answer http://stackoverflow.com/a/32324568
    // TODO: invert boolean operations to positive
    return !( u_ref_i < boundary || v_ref_i < boundary || u_ref_i >= full_img.cols-boundary || v_ref_i >= full_img.rows-boundary );
  }

  float horBorderDist(float y);
  uchar* begin( int i );
  uchar* end( int i );
};

} // namespace plsvo

#endif // SVO_FEATURE_H_
