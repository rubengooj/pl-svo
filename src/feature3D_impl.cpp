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

void Point::optimize(const size_t n_iter)
{
  Vector3d old_point = pos_;
  double chi2 = 0.0;
  Matrix3d A;
  Vector3d b;

  for(size_t i=0; i<n_iter; i++)
  {
    A.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
      Matrix23d J;
      const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
      Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J);
      const Vector2d e(vk::project2d((*it)->f) - vk::project2d(p_in_f));
      new_chi2 += e.squaredNorm();
      A.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * e;
    }

    // solve linear system
    const Vector3d dp(A.ldlt().solve(b));

    // check if error increased
    if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
    {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i
           << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      pos_ = old_point; // roll-back
      break;
    }

    // update the model
    Vector3d new_point = pos_ + dp;
    old_point = pos_;
    pos_ = new_point;
    chi2 = new_chi2;
#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i
         << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << vk::norm_max(b)
         << endl;
#endif

    // stop when converged
    if(vk::norm_max(dp) <= EPS)
      break;

  }
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif
}

void LineSeg::optimize(const size_t n_iter)
{
  Vector3d old_spoint = spos_;
  Vector3d old_epoint = epos_;
  double chi2s = 0.0;
  double chi2e = 0.0;
  Matrix3d As,Ae;
  Vector3d bs,be;

  for(size_t i=0; i<n_iter; i++)
  {
    As.setZero();
    Ae.setZero();
    bs.setZero();
    be.setZero();
    double new_chi2_s = 0.0;
    double new_chi2_e = 0.0;

    // compute residuals TODO: optimizing from endpoint to endpoint
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
      Matrix23d J, Js, Je;
      const Vector3d sp_in_f( (*it)->frame->T_f_w_ * spos_);
      Point::jacobian_xyz2uv( sp_in_f, (*it)->frame->T_f_w_.rotation_matrix(), Js);
      const Vector3d ep_in_f( (*it)->frame->T_f_w_ * epos_);
      Point::jacobian_xyz2uv( ep_in_f, (*it)->frame->T_f_w_.rotation_matrix(), Je);
      const Vector2d es(vk::project2d(static_cast<LineFeat*>(*it)->sf) - vk::project2d(sp_in_f));
      const Vector2d ee(vk::project2d(static_cast<LineFeat*>(*it)->ef) - vk::project2d(ep_in_f));
      new_chi2_s   += es.squaredNorm();
      As.noalias() += Js.transpose() * Js;
      bs.noalias() -= Js.transpose() * es;
      new_chi2_e   += ee.squaredNorm();
      Ae.noalias() += Je.transpose() * Je;
      be.noalias() -= Je.transpose() * ee;
    }

    // solve linear system
    const Vector3d dps(As.ldlt().solve(bs));
    const Vector3d dpe(Ae.ldlt().solve(be));

    // check if error increased
    if( (i > 0 && new_chi2_s > chi2s) || (bool) std::isnan((double)dps[0]) || (i > 0 && new_chi2_e > chi2e) || (bool) std::isnan((double)dpe[0]) )
    {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i
           << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      spos_ = old_spoint; // roll-back
      epos_ = old_epoint; // roll-back
      break;
    }

    // update the model
    Vector3d new_spoint = spos_ + dps;
    old_spoint = spos_;
    spos_ = new_spoint;
    chi2s = new_chi2_s;
    Vector3d new_epoint = epos_ + dpe;
    old_epoint = epos_;
    epos_ = new_epoint;
    chi2e = new_chi2_e;

#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i
         << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << vk::norm_max(b)
         << endl;
#endif

    // stop when converged
    if( vk::norm_max(dps) <= EPS || vk::norm_max(dpe) <= EPS )
      break;
  }
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif

}
} // namespace plsvo
