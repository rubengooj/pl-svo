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


#include <plsvo/feature.h>

namespace plsvo {

Feature::Feature(const Vector2d &_px) :
  frame(NULL),
  px(_px),
  f(Vector3d()),
  level(-1)
{}

Feature::Feature(Frame *_frame, const Eigen::Vector2d &_px, int _level) :
  frame(_frame),
  px(_px),
  f(frame->cam_->cam2world(px)),
  level(_level)
{}

Feature::Feature(Frame *_frame, const Vector2d &_px, const Vector3d &_f, int _level) :
  frame(_frame),
  px(_px),
  f(_f),
  level(_level)
{}

PointFeat::PointFeat(const Vector2d &_px) :
  Feature(_px),
  type(CORNER),
  grad(1.0,0.0),
  feat3D(NULL)
{ }

PointFeat::PointFeat(Frame *_frame, const Vector2d &_px, int _level) :
  Feature(_frame, _px, _level),
  type(CORNER),
  grad(1.0,0.0),
  feat3D(NULL)
{}

PointFeat::PointFeat(Frame *_frame, const Vector2d &_px, const Vector3d &_f, int _level) :
  Feature(_frame, _px, _f, _level),
  type(CORNER),
  grad(1.0,0.0),
  feat3D(NULL)
{}

PointFeat::PointFeat(Frame *_frame, Point *_point, const Vector2d &_px, const Vector3d &_f, int _level) :
  Feature(_frame, _px, _f, _level),
  type(CORNER),
  grad(1.0,0.0),
  feat3D(_point)
{}

LineFeat::LineFeat(const Vector2d &_spx, const Vector2d &_epx) :
  Feature(0.5f*(_spx+_epx)),
  type(LINE_SEGMENT),
  spx(_spx),
  epx(_epx),
  sf(Vector3d()),
  ef(Vector3d()),
  feat3D(NULL),
  grad(1.0,0.0),
  length((_epx-_spx).norm())
{ }

LineFeat::LineFeat(Frame *_frame, const Vector2d &_spx, const Vector2d &_epx, int _level) :
  Feature(_frame, 0.5f*(_spx+_epx), _level),
  type(LINE_SEGMENT),
  spx(_spx),
  epx(_epx),
  sf(frame->cam_->cam2world(spx)),
  ef(frame->cam_->cam2world(epx)),
  feat3D(NULL),
  length((_epx-_spx).norm())
{
  line = sf.cross(ef) ;
  line = line / sqrt(line(0)*line(0)+line(1)*line(1));
  // 2 first coordinates in line are the normalized normal in 2D (same direction as gradient)
  grad = Vector2d(line[0],line[1]);
}

LineFeat::LineFeat(Frame *_frame, const Vector2d &_spx, const Vector2d &_epx, int _level, double _angle) :
  Feature(_frame, 0.5f*(_spx+_epx), _level),
  type(LINE_SEGMENT),
  spx(_spx),
  epx(_epx),
  sf(frame->cam_->cam2world(spx)),
  ef(frame->cam_->cam2world(epx)),
  feat3D(NULL),
  length((_epx-_spx).norm()),
  angle(_angle)
{
  line = sf.cross(ef) ;
  line = line / sqrt(line(0)*line(0)+line(1)*line(1));
  // 2 first coordinates in line are the normalized normal in 2D (same direction as gradient)
  grad = Vector2d(line[0],line[1]);
}

LineFeat::LineFeat(Frame *_frame, const Vector2d &_spx, const Vector2d &_epx, const Vector3d &_sf, const Vector3d &_ef, int _level) :
  Feature(_frame, 0.5f*(_spx+_epx), _level),
  type(LINE_SEGMENT),
  spx(_spx),
  epx(_epx),
  sf(_sf),
  ef(_ef),
  feat3D(NULL),
  grad(1.0,0.0),
  length((_epx-_spx).norm())
{
  line = sf.cross(ef) ;
  line = line / sqrt(line(0)*line(0)+line(1)*line(1));
  // 2 first coordinates in line are the normalized normal in 2D (same direction as gradient)
  grad = Vector2d(line[0],line[1]);
}

LineFeat::LineFeat(Frame *_frame, LineSeg *_ls, const Vector2d &_spx, const Vector2d &_epx, const Vector3d &_sf, const Vector3d &_ef, int _level) :
  Feature(_frame, 0.5f*(_spx+_epx), _level),
  type(LINE_SEGMENT),
  spx(_spx),
  epx(_epx),
  sf(_sf),
  ef(_ef),
  feat3D(_ls),
  grad(1.0,0.0),
  length((_epx-_spx).norm())
{
  line = sf.cross(ef) ;
  line = line / sqrt(line(0)*line(0)+line(1)*line(1));
  // 2 first coordinates in line are the normalized normal in 2D (same direction as gradient)
  grad = Vector2d(line[0],line[1]);
}

size_t LineFeat::setupSampling(size_t patch_size, Vector2d &dif)
{
  // complete sampling of the segment surroundings,
  // with minimum overlap of the square patches
  // if segment is horizontal or vertical, N is seg_length/patch_size
  // if the segment has angle theta, we need to correct according to the distance from center to unit-square border: *corr
  // scale (pyramid level) is accounted for later
  dif = epx - spx; // difference vector from start to end point
  double tan_dir = std::min(fabs(dif[0]),fabs(dif[1])) / std::max(fabs(dif[0]),fabs(dif[1]));
  double sin_dir = tan_dir / sqrt( 1.0+tan_dir*tan_dir );
  double correction = 2.0 * sqrt( 1.0 + sin_dir*sin_dir );
  return std::max( 1.0, length / (2.0*patch_size*correction) );
  // If length is very low the segment approaches a point and the minimum 1 sample is taken (the central point)
}

Patch::Patch(int _size, const cv::Mat &_img)
{
  // assert the size is an even number for symmetry in the window
  assert( _size%2 == 0 );
  size = _size;
  halfsize = size/2;
  area = size*size;
  border = halfsize+1;

  // from image get stride
  full_img = _img;
  stride = _img.step[0]; // take as stride between rows the corresponding Mat property
}

void Patch::setPosition(const Vector2d &px)
{
  // set patch center position
  u_ref = px[0];
  v_ref = px[1];
  u_ref_i = floorf(u_ref);
  v_ref_i = floorf(v_ref);
  pos_ref_i = px.cast<int>(); // Casting to int truncates non-integer part (in >=0 equivalent to floor)
}

void Patch::computeInterpWeights()
{
  // compute bilateral interpolation weights for patch subpixel position
  const float subpix_u_ref = u_ref-u_ref_i;
  const float subpix_v_ref = v_ref-v_ref_i;
  wTL = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
  wTR = subpix_u_ref * (1.0-subpix_v_ref);
  wBL = (1.0-subpix_u_ref) * subpix_v_ref;
  wBR = subpix_u_ref * subpix_v_ref;
}

void Patch::setRoi()
{
  // set interest rectangle
  rect = cv::Rect( u_ref_i - halfsize, v_ref_i - halfsize, size, size );

  // copy submatrix of the full image corresponding to this patch
  // (OR NOT?) due to subpixel precision, we need to extend the size by 1 in each direction
  roi = cv::Mat(full_img, rect);
}

uchar* Patch::begin(int i)
{
  return roi.ptr(i);
}
uchar* Patch::end(int i)
{
  return roi.ptr(i) + roi.cols;
}

RotatedRectPatch::RotatedRectPatch(float length, float width, float angle, const cv::Mat &_img) :
  length(length),
  width(width),
  angle(angle)
{
  // set bounding dimensions
  bSize_y = cos(angle) * width + sin(angle) * length;
  bSize_x = sin(angle) * width + cos(angle) * length;
  // set bounding area
  area = (int) ceil(bSize_x) * ceil(bSize_y); // Upper bound in number of pixels that the rotated region can contain

  // from image get stride
  full_img = _img;
  stride = _img.step[0]; // take as stride between rows the corresponding Mat property
}

void RotatedRectPatch::setPosition(const Vector2d &px)
{
  // set patch center position
  u_ref = px[0];
  v_ref = px[1];
  u_ref_i = floorf(u_ref);
  v_ref_i = floorf(v_ref);
  pos_ref_i = px.cast<int>(); // Casting to int truncates non-integer part (in >=0 equivalent to floor)
}

void RotatedRectPatch::computeInterpWeights()
{
  // compute bilateral interpolation weights for patch subpixel position
  const float subpix_u_ref = u_ref-u_ref_i;
  const float subpix_v_ref = v_ref-v_ref_i;
  wTL = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
  wTR = subpix_u_ref * (1.0-subpix_v_ref);
  wBL = (1.0-subpix_u_ref) * subpix_v_ref;
  wBR = subpix_u_ref * subpix_v_ref;
}

void RotatedRectPatch::setRoi()
{
  // set interest rectangle
  rect = cv::RotatedRect(cv::Point2f(u_ref_i,v_ref_i), cv::Size2f(length,width), angle*180.0/M_PI);

  // copy submatrix of the full image corresponding to the bounding box of this patch
  roi = cv::Mat(full_img, rect.boundingRect());
}

float RotatedRectPatch::horBorderDist(float y)
{
  // The geometric index function resembles an odd function: f(-x)=-f(x)
  // All results can be obtained from the output for positive angle (0<ang<pi/2)
  if(-M_PI<=angle && angle<0)
  {
    angle = -angle;
    y = bSize_y-y;
  }

  float y_disc = cos(angle) * width;// Value of y (row) at which discontinuity due to rectangle vertex occurs
  float offset;
  if(y<=y_disc)
    offset = (y_disc-y)*tan(angle);
  else
    offset = (y-y_disc)/tan(angle);
  return (int)offset;
}

uchar* RotatedRectPatch::begin(int i)
{
  return roi.ptr(i) + (int)horBorderDist(i);
}
uchar* RotatedRectPatch::end(int i)
{
    return roi.ptr(i) + roi.cols - (int)horBorderDist(bSize_y-i);
}

} // namespace plsvo
