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


#ifndef SVO_POINT_H_
#define SVO_POINT_H_

#include <boost/noncopyable.hpp>
#include <plsvo/global.h>

namespace g2o {
  class VertexSBAPointXYZ;
}
typedef g2o::VertexSBAPointXYZ g2oPoint;

namespace plsvo {

class Feature;
class PointFeat;
class LineFeat;

typedef Matrix<double, 2, 3> Matrix23d;

/// A generic geometric feature in 3D from which particular elements inherit (points, segments, all 3D objects)
template <class FeatureT>
class Feature3D : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Current status of the 3D object (measure of quality)
  enum StatusType {
    TYPE_DELETED,
    TYPE_CANDIDATE,
    TYPE_UNKNOWN,
    TYPE_GOOD
  };

  static int                  counter_;                 //!< Counts the number of created points. Used to set the unique id.
  int                         id_;                      //!< Unique ID of the 3D feature.

  list<FeatureT*>             obs_;                     //!< References to keyframes which observe the 3D feature: 2D features derived from its observation
  size_t                      n_obs_;                   //!< Number of obervations: Keyframes AND successful reprojections in intermediate frames.

  int                         last_published_ts_;       //!< Timestamp of last publishing.
  int                         last_projected_kf_id_;    //!< Flag for the reprojection: don't reproject a 3D feature twice.
  StatusType                  type_;                    //!< Quality of the 3D feature.
  int                         n_failed_reproj_;         //!< Number of failed reprojections. Used to assess the quality of the 3D feature.
  int                         n_succeeded_reproj_;      //!< Number of succeeded reprojections. Used to assess the quality of the 3D feature.
  int                         last_structure_optim_;    //!< Timestamp of last 3D feature optimization

  Feature3D(int id_);
  virtual ~Feature3D() = 0; // Make this class abstract

  /// Add a reference to a frame.
  void addFrameRef(FeatureT *ftr);

  /// Remove reference to a frame.
  bool deleteFrameRef(Frame* frame);

  /// Check whether mappoint has reference to a frame.
  Feature* findFrameRef(Frame* frame);

  /// Get Frame with similar viewpoint.
  virtual bool getCloseViewObs(const Vector3d& framepos, Feature*& obs) const = 0;

  /// Optimize point position through minimizing the reprojection error.
  virtual void optimize(const size_t n_iter) = 0;
};

// Set default trivial implementation for inheriting classes (non-abstract)
template<class FeatureT> inline Feature3D<FeatureT>::~Feature3D() {}

/// A 3D point on the surface of the scene.
class Point : public Feature3D<PointFeat>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Vector3d                    pos_;                     //!< 3d pos of the point in the world coordinate frame.
  Vector3d                    normal_;                  //!< Surface normal at point.
  Matrix3d                    normal_information_;      //!< Inverse covariance matrix of normal estimation.
  bool                        normal_set_;              //!< Flag whether the surface normal was estimated or not.

  g2oPoint*                   v_g2o_;                   //!< Temporary pointer to the point-vertex in g2o during bundle adjustment.

  Point(const Vector3d& pos);
  Point(const Vector3d& pos, PointFeat *ftr);

  /// Initialize point normal. The inital estimate will point towards the frame.
  void initNormal();

  /// Get Frame with similar viewpoint.
  bool getCloseViewObs(const Vector3d& framepos, Feature*& obs) const;

  /// Optimize point position through minimizing the reprojection error.
  void optimize(const size_t n_iter);

  /// Get number of observations.
  inline size_t nRefs() const { return obs_.size(); }

  /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
  inline static void jacobian_xyz2uv(
      const Vector3d& p_in_f,
      const Matrix3d& R_f_w,
      Matrix23d& point_jac)
  {
    const double z_inv = 1.0/p_in_f[2];
    const double z_inv_sq = z_inv*z_inv;
    point_jac(0, 0) = z_inv;
    point_jac(0, 1) = 0.0;
    point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
    point_jac(1, 0) = 0.0;
    point_jac(1, 1) = z_inv;
    point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
    point_jac = - point_jac * R_f_w;
  }
};

/// A 3D line segment on the surface of the scene.
class LineSeg : public Feature3D<LineFeat>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector3d                    spos_;                    //!< 3d pos of the start point in the world coordinate frame.
  Vector3d                    epos_;                    //!< 3d pos of the end point in the world coordinate frame.

  g2oPoint*                   v_g2o_;                   //!< Temporary pointer to the point-vertex in g2o during bundle adjustment.

  LineSeg(const Vector3d& spos, const Vector3d& epos);
  LineSeg(const Vector3d& spos, const Vector3d& epos, LineFeat *ftr);

  /// Get Frame with similar viewpoint.
  bool getCloseViewObs(const Vector3d& framepos, Feature*& obs) const;

  /// Optimize point position through minimizing the reprojection error.
  void optimize(const size_t n_iter);

  /// Get number of observations.
  inline size_t nRefs() const { return obs_.size(); }

  /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
  inline static void jacobian_xyz2uv(
      const Vector3d& p_in_f,
      const Matrix3d& R_f_w,
      Matrix23d& point_jac)
  {
    const double z_inv = 1.0/p_in_f[2];
    const double z_inv_sq = z_inv*z_inv;
    point_jac(0, 0) = z_inv;
    point_jac(0, 1) = 0.0;
    point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
    point_jac(1, 0) = 0.0;
    point_jac(1, 1) = z_inv;
    point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
    point_jac = - point_jac * R_f_w;
  }
};

/* Define template methods in parent class */
/* --------------------------------------- */
template <class FeatureT>
Feature3D<FeatureT>::Feature3D(int _id) :
  id_(_id),
  n_obs_(0),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{}

template <class FeatureT>
int Feature3D<FeatureT>::counter_ = 0;

template <class FeatureT>
void Feature3D<FeatureT>::addFrameRef(FeatureT* ftr)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

template <class FeatureT>
Feature* Feature3D<FeatureT>::findFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    if((*it)->frame == frame)
      return *it;
  return NULL;    // no keyframe found
}

template <class FeatureT>
bool Feature3D<FeatureT>::deleteFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    if((*it)->frame == frame)
    {
      obs_.erase(it);
      return true;
    }
  }
  return false;
}

} // namespace plsvo

#endif // SVO_POINT_H_
