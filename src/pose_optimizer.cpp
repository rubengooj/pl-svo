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
#include <plsvo/pose_optimizer.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <plsvo/feature3D.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>

namespace plsvo {
namespace pose_optimizer {

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs_pt,
    size_t& num_obs_ls)
{

  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::TukeyWeightFunction weight_function;
  SE3 T_old(frame->T_f_w_);
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  vk::robust_cost::MADScaleEstimator scale_estimator;
  std::vector<float> errors; errors.reserve(frame->pt_fts_.size()+frame->seg_fts_.size());
  for(auto it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D == NULL)
      continue;
    Vector2d e = vk::project2d((*it)->f)
               - vk::project2d(frame->T_f_w_ * (*it)->feat3D->pos_);
    e *= 1.0 / (1<<(*it)->level);
    errors.push_back(e.norm());
  }
  double estimated_scale_pt = scale_estimator.compute(errors);
  num_obs_pt = errors.size();

  std::vector<float> errors_ls; errors_ls.reserve(frame->seg_fts_.size());
  for(auto it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
  {
    LineFeat* it_ = static_cast<LineFeat*>( *it );
    if(it_->feat3D == NULL)
      continue;
    // Check distance function
    Vector3d line   = it_->line;   
    Vector3d s_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->spos_) );
    Vector3d e_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->epos_) );
    float es = line.dot(s_proj);
    float ee = line.dot(e_proj);
    errors.push_back( sqrt(es*es+ee*ee) );
    errors_ls.push_back( sqrt(es*es+ee*ee) );
  }
  if(errors.empty())
    return;
  num_obs_ls = errors_ls.size();

  // Estimate scale of line segments' errors
  double estimated_scale_ls = 1.f;
  if(!errors_ls.empty())
    estimated_scale_ls = scale_estimator.compute(errors_ls);
  estimated_scale = estimated_scale_pt;

  size_t num_obs = num_obs_pt + num_obs_ls;
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale_pt = estimated_scale_pt;
  double scale_ls = estimated_scale_ls;
  for(size_t iter=0; iter<n_iter; iter++)
  {

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
    {
      if((*it)->feat3D == NULL)
        continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->feat3D->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale_pt);
      //double weight = 1.0 / ( 1.0 + e.norm()/scale_pt );
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }

    // compute residual for line segments
    for(auto it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
    {
      LineFeat* it_ = static_cast<LineFeat*>( *it );
      if((it_)->feat3D == NULL)
        continue;
      Vector2d  e, l_aux;
      Matrix26d J_s, J_e, J;
      Vector3d xyz_f_s(frame->T_f_w_ * (it_)->feat3D->spos_);
      Vector3d xyz_f_e(frame->T_f_w_ * (it_)->feat3D->epos_);
      Frame::jacobian_xyz2uv(xyz_f_s, J_s);
      Frame::jacobian_xyz2uv(xyz_f_e, J_e);

      Vector3d s_proj = vk::unproject2d( vk::project2d(xyz_f_s) );
      Vector3d e_proj = vk::unproject2d( vk::project2d(xyz_f_e) );
      Vector3d line   = it_->line;
      float      ds   = line.dot(s_proj);
      float      de   = line.dot(e_proj);
      e << ds, de;

      double sqrt_inv_cov = 1.0 / (1<<(it_)->level);    // CHECK
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm());       // just for debug

      l_aux << line(0), line(1);
      J_s *= sqrt_inv_cov * ds / e.norm();
      J_e *= sqrt_inv_cov * ds / e.norm();
      J.block(0,0,1,6) = l_aux.transpose() * J_s;
      J.block(1,0,1,6) = l_aux.transpose() * J_e;

      double weight = weight_function.value(e.norm()/scale_ls);
      //double weight = 1.0 / ( 1.0 + e.norm()/scale_ls );
      A.noalias()  += J.transpose()*J*weight;
      b.noalias()  -= J.transpose()*e*weight;
      new_chi2     += e.squaredNorm()*weight;
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  const double pixel_variance = 1.0;
  frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled_pt = reproj_thresh / frame->cam_->errorMultiplier2();
  double reproj_thresh_scaled_ls = reproj_thresh_scaled_pt * estimated_scale_ls / estimated_scale_pt;
  size_t n_deleted_refs_pt = 0;
  size_t n_deleted_refs_ls = 0;
  for(list<PointFeat*>::iterator it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D == NULL)
      continue;

    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->feat3D->pos_ );
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());    
    if(e.norm() > reproj_thresh_scaled_pt)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (*it)->feat3D = NULL;
      ++n_deleted_refs_pt;
    }
  }

  for(list<LineFeat*>::iterator it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
  {
    LineFeat* it_ = static_cast<LineFeat*>( *it );
    if((it_)->feat3D == NULL)
      continue;
    Vector2d e;
    Vector3d line   = it_->line;
    Vector3d s_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->spos_) );
    Vector3d e_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->epos_) );
    e << line.dot(s_proj), line.dot(e_proj);
    double sqrt_inv_cov = 1.0 / (1<<(it_)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled_ls)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (it_)->feat3D = NULL;
      ++n_deleted_refs_ls;
    }
  }

  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs_pt << " points \t " << n_deleted_refs_ls << " lines"
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs    -= (n_deleted_refs_pt+n_deleted_refs_pt);
  num_obs_pt -= n_deleted_refs_pt;
  num_obs_ls -= n_deleted_refs_ls;
}

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const size_t n_iter_ref,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs_pt,
    size_t& num_obs_ls)
{

  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::TukeyWeightFunction weight_function;
  SE3 T_old(frame->T_f_w_);
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  vk::robust_cost::MADScaleEstimator scale_estimator;
  std::vector<float> errors; errors.reserve(frame->pt_fts_.size()+frame->seg_fts_.size());
  for(auto it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D == NULL)
      continue;
    Vector2d e = vk::project2d((*it)->f)
               - vk::project2d(frame->T_f_w_ * (*it)->feat3D->pos_);
    e *= 1.0 / (1<<(*it)->level);
    errors.push_back(e.norm());
  }
  double estimated_scale_pt = scale_estimator.compute(errors);
  num_obs_pt = errors.size();

  std::vector<float> errors_ls; errors_ls.reserve(frame->seg_fts_.size());
  for(auto it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
  {
    LineFeat* it_ = static_cast<LineFeat*>( *it );
    if(it_->feat3D == NULL)
      continue;
    // Check distance function
    Vector3d line   = it_->line;
    Vector3d s_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->spos_) );
    Vector3d e_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->epos_) );
    float es = line.dot(s_proj);
    float ee = line.dot(e_proj);
    errors.push_back( sqrt(es*es+ee*ee) );
    errors_ls.push_back( sqrt(es*es+ee*ee) );
  }
  if(errors.empty())
    return;
  num_obs_ls = errors_ls.size();

  // Estimate scale of line segments' errors
  double estimated_scale_ls = 1.f;
  if(!errors_ls.empty())
    estimated_scale_ls = scale_estimator.compute(errors_ls);
  estimated_scale = estimated_scale_pt;

  size_t num_obs = num_obs_pt + num_obs_ls;
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale_pt = estimated_scale_pt;
  double scale_ls = estimated_scale_ls;
  for(size_t iter=0; iter<n_iter; iter++)
  {

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
    {
      if((*it)->feat3D == NULL)
        continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->feat3D->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale_pt);
      //double weight = 1.0 / ( 1.0 + e.norm()/scale_pt );
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }

    // compute residual for line segments
    for(auto it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
    {
      LineFeat* it_ = static_cast<LineFeat*>( *it );
      if((it_)->feat3D == NULL)
        continue;
      Vector2d  e, l_aux;
      Matrix26d J_s, J_e, J;
      Vector3d xyz_f_s(frame->T_f_w_ * (it_)->feat3D->spos_);
      Vector3d xyz_f_e(frame->T_f_w_ * (it_)->feat3D->epos_);
      Frame::jacobian_xyz2uv(xyz_f_s, J_s);
      Frame::jacobian_xyz2uv(xyz_f_e, J_e);

      Vector3d s_proj = vk::unproject2d( vk::project2d(xyz_f_s) );
      Vector3d e_proj = vk::unproject2d( vk::project2d(xyz_f_e) );
      Vector3d line   = it_->line;
      float      ds   = line.dot(s_proj);
      float      de   = line.dot(e_proj);
      e << ds, de;

      double sqrt_inv_cov = 1.0 / (1<<(it_)->level);    // CHECK
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm());       // just for debug

      l_aux << line(0), line(1);
      J_s *= sqrt_inv_cov * ds / e.norm();
      J_e *= sqrt_inv_cov * ds / e.norm();
      J.block(0,0,1,6) = l_aux.transpose() * J_s;
      J.block(1,0,1,6) = l_aux.transpose() * J_e;

      double weight = weight_function.value(e.norm()/scale_ls);
      //double weight = 1.0 / ( 1.0 + e.norm()/scale_ls );
      A.noalias()  += J.transpose()*J*weight;
      b.noalias()  -= J.transpose()*e*weight;
      new_chi2     += e.squaredNorm()*weight;
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  const double pixel_variance = 1.0;
  frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled_pt = reproj_thresh / frame->cam_->errorMultiplier2();
  double reproj_thresh_scaled_ls = reproj_thresh_scaled_pt * estimated_scale_ls / estimated_scale_pt;
  size_t n_deleted_refs_pt = 0;
  size_t n_deleted_refs_ls = 0;
  for(list<PointFeat*>::iterator it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
  {
    if((*it)->feat3D == NULL)
      continue;

    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->feat3D->pos_ );
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled_pt)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (*it)->feat3D = NULL;
      ++n_deleted_refs_pt;
    }
  }

  for(list<LineFeat*>::iterator it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
  {
    LineFeat* it_ = static_cast<LineFeat*>( *it );
    if((it_)->feat3D == NULL)
      continue;
    Vector2d e;
    Vector3d line   = it_->line;
    Vector3d s_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->spos_) );
    Vector3d e_proj = vk::unproject2d( vk::project2d(frame->T_f_w_ * it_->feat3D->epos_) );
    e << line.dot(s_proj), line.dot(e_proj);
    double sqrt_inv_cov = 1.0 / (1<<(it_)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled_ls)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (it_)->feat3D = NULL;
      ++n_deleted_refs_ls;
    }
  }

  // refinement with inliers
  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  for(size_t iter=0; iter<n_iter_ref; iter++)
  {

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->pt_fts_.begin(); it!=frame->pt_fts_.end(); ++it)
    {
      if((*it)->feat3D == NULL)
        continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->feat3D->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale_pt);
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2    += e.squaredNorm()*weight;
    }

    // compute residual for line segments
    for(auto it=frame->seg_fts_.begin(); it!=frame->seg_fts_.end(); ++it)
    {
      LineFeat* it_ = static_cast<LineFeat*>( *it );
      if((it_)->feat3D == NULL)
        continue;
      Vector2d  e, l_aux;
      Matrix26d J_s, J_e, J;
      Vector3d xyz_f_s(frame->T_f_w_ * (it_)->feat3D->spos_);
      Vector3d xyz_f_e(frame->T_f_w_ * (it_)->feat3D->epos_);
      Frame::jacobian_xyz2uv(xyz_f_s, J_s);
      Frame::jacobian_xyz2uv(xyz_f_e, J_e);

      Vector3d s_proj = vk::unproject2d( vk::project2d(xyz_f_s) );
      Vector3d e_proj = vk::unproject2d( vk::project2d(xyz_f_e) );
      Vector3d line   = it_->line;
      float      ds   = line.dot(s_proj);
      float      de   = line.dot(e_proj);
      e << ds, de;

      double sqrt_inv_cov = 1.0 / (1<<(it_)->level);    // CHECK
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug

      l_aux << line(0), line(1);
      J_s *= sqrt_inv_cov * ds / e.norm();
      J_e *= sqrt_inv_cov * ds / e.norm();
      J.block(0,0,1,6) = l_aux.transpose() * J_s;
      J.block(1,0,1,6) = l_aux.transpose() * J_e;

      double weight = weight_function.value(e.norm()/scale_ls);
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2    += e.squaredNorm()*weight;
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }


  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs_pt << " points \t " << n_deleted_refs_ls << " lines"
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs    -= (n_deleted_refs_pt+n_deleted_refs_pt);
  num_obs_pt -= n_deleted_refs_pt;
  num_obs_ls -= n_deleted_refs_ls;
}

} // namespace pose_optimizer
} // namespace plsvo

