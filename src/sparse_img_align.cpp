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
#include <plsvo/sparse_img_align.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <plsvo/config.h>
#include <plsvo/feature3D.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>

namespace plsvo {

SparseImgAlign::SparseImgAlign(
    int max_level, int min_level, int n_iter,
    Method method, bool display, bool verbose) :
        display_(display),
        max_level_(max_level),
        min_level_(min_level)
{
  n_iter_      = n_iter;
  n_iter_init_ = n_iter_;
  method_      = method;
  verbose_     = verbose;
  eps_         = 0.000001;
}

size_t SparseImgAlign::run(FramePtr ref_frame, FramePtr cur_frame)
{
  reset();

  if(ref_frame->pt_fts_.empty() && ref_frame->seg_fts_.empty())
  {
    SVO_WARN_STREAM("SparseImgAlign: no features (points or segments) to track!");
    return 0;
  }

  ref_frame_ = ref_frame;
  cur_frame_ = cur_frame;

  // The maximum number of segment samples (without overlapping) is proportional to
  // the accumulated length of the segment features
  float total_length = 0;
  std::for_each(ref_frame->seg_fts_.begin(), ref_frame->seg_fts_.end(), [&](plsvo::Feature* i){
    LineFeat* seg = static_cast<LineFeat*>(i);
    total_length += seg->length;
  });
  int max_num_seg_samples = std::ceil( total_length / patch_size_ );

  // setup cache structures to allocate the optimization data (residues, jacobians, etc.)
  pt_cache_  = Cache( ref_frame_->pt_fts_.size(), patch_area_ );
  seg_cache_ = Cache( max_num_seg_samples, patch_area_ );

  SE3 T_cur_from_ref(cur_frame_->T_f_w_ * ref_frame_->T_f_w_.inverse());

  for(level_=max_level_; level_>=min_level_; --level_)
  {
    mu_ = 0.1;
    pt_cache_.jacobian.setZero();
    seg_cache_.jacobian.setZero();
    have_ref_patch_cache_ = false;
    if(verbose_)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
    optimize(T_cur_from_ref);
  }
  cur_frame_->T_f_w_ = T_cur_from_ref * ref_frame_->T_f_w_;

  return n_meas_/patch_area_;
}

Matrix<double, 6, 6> SparseImgAlign::getFisherInformation()
{
  double sigma_i_sq = 5e-4*255*255; // image noise
  Matrix<double,6,6> I = H_/sigma_i_sq;
  return I;
}

void SparseImgAlign::precomputeReferencePatches()
{
  precomputeGaussNewtonParamsPoints(pt_cache_,ref_frame_->pt_fts_);
  precomputeGaussNewtonParamsSegments(seg_cache_,ref_frame_->seg_fts_);
  // set flag to true to avoid repeating unnecessary computations in the following iterations
  have_ref_patch_cache_ = true;
}

double SparseImgAlign::computeResiduals(
    const SE3& T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale)
{
  // Warp the (cur)rent image such that it aligns with the (ref)erence image

  // setup residues image if display is active
  const cv::Mat& cur_img = cur_frame_->img_pyr_.at(level_);
  if(linearize_system && display_)
    resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

  // do the precomputation of reference parameters (patch intensities and jacobian)
  // for the Inverse Compositional approach if not done yet
  if(have_ref_patch_cache_ == false)
    precomputeReferencePatches();

  // they didn't use weights
  linearize_system     = true;
  compute_weight_scale = false;
  use_weights_         = true;

  /* computations for the point patches */
  /* ---------------------------------- */
  // declare the GN parameters H=sum(J_i*J_i') and Jres=sum(J_i*res_i)
  Matrix<double, 6, 6> pt_H;
  Matrix<double, 6, 1> pt_Jres;
  // define other interest variables
  std::vector<float> pt_errors;
  float pt_chi2 = 0.0;

  // compute the parameters for Gauss-Newton update related to point patches
  computeGaussNewtonParamsPoints(
        T_cur_from_ref, linearize_system, compute_weight_scale,
        pt_cache_, ref_frame_->pt_fts_, pt_H, pt_Jres, pt_errors, pt_chi2 );

  /* computations for the segment patches */
  /* ------------------------------------ */
  // declare the GN parameters H=sum(J_i*J_i') and Jres=sum(J_i*res_i)
  Matrix<double, 6, 6> seg_H;
  Matrix<double, 6, 1> seg_Jres;
  // define other interest variables
  std::vector<float> seg_errors;
  float seg_chi2 = 0.0;

  // compute the parameters for Gauss-Newton update
  computeGaussNewtonParamsSegments(
        T_cur_from_ref, linearize_system, compute_weight_scale,
        seg_cache_, ref_frame_->seg_fts_, seg_H, seg_Jres, seg_errors, seg_chi2 );

  /* fuse optimization variables coming from both points and segments */
  /* ---------------------------------------------------------------- */
  if(linearize_system)
  {
    // sum the contribution from both points and segments
    H_    = pt_H    + seg_H;
    Jres_ = pt_Jres + seg_Jres;
  }

  float chi2 = pt_chi2 + seg_chi2;

  // compute total scale and compare with the rest
  std::vector<float> errors = pt_errors;
  for(vector<float>::iterator it = seg_errors.begin(); it != seg_errors.end(); ++it){
    errors.push_back(*it);
  }
  // compute the weights on the first iteration
  vk::robust_cost::MADScaleEstimator scale_estimator;
  if( compute_weight_scale && iter_ == 0 ){
    if(pt_errors.size()!=0)  scale_pt = scale_estimator.compute(pt_errors);
    if(seg_errors.size()!=0) scale_ls = scale_estimator.compute(seg_errors);
    if(errors.size()!=0)     scale_   = scale_estimator.compute(errors);
    if(scale_ls < 0.0001) scale_ls = 1.0; // after initializing there's no LS
  }
  else if(iter_== 0){
    scale_   = 1.0;
    scale_ls = 1.0;
    scale_pt = 1.0;
  }

  return chi2/n_meas_;
}

void SparseImgAlign::precomputeGaussNewtonParamsPoints(Cache &cache, list<PointFeat*> &fts)
{

  // initialize patch parameters (mainly define its geometry)
  Patch patch( patch_size_, ref_frame_->img_pyr_.at(level_) );
  const float scale = 1.0f/(1<<level_);
  const Vector3d ref_pos = ref_frame_->pos();
  const double focal_length = ref_frame_->cam_->errorMultiplier2();

  /* precompute intensity and jacobian for the point patches in the reference image */
  {
    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
    for(auto it=fts.begin(), ite=fts.end();
        it!=ite; ++it, ++feature_counter, ++visiblity_it)
    {
      // if point is not valid (empty or null) skip this feature
      if((*it)->feat3D == NULL)
        continue;

      // set patch position for current feature
      patch.setPosition((*it)->px*scale);
      // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
      if(!patch.isInFrame(patch.halfsize+1))
        continue;
      // compute the bilinear interpolation weights constant along the patch scan
      patch.computeInterpWeights();
      // set the patch at the corresponding ROI in the image
      patch.setRoi();

      // flag the feature as valid/visible
      *visiblity_it = true;

      // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
      const double depth(((*it)->feat3D->pos_ - ref_pos).norm());
      const Vector3d xyz_ref((*it)->f*depth);

      // evaluate projection jacobian
      Matrix<double,2,6> frame_jac;
      Frame::jacobian_xyz2uv(xyz_ref, frame_jac);

      // iterate through all points in the Region Of Interest defined by the patch
      // the pointer points to the data in the original image matrix
      // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
      size_t pixel_counter = 0;
      float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
      uint8_t* img_ptr;                 // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
      const int stride = patch.stride;  // the stride stored in the patch is that necessary to jump between the full matrix rows
      for(int y=0; y<patch.size; ++y)   // sweep the path row-wise (most efficient for RowMajor storage)
      {
        // get the pointer to first element in row y of the patch ROI
        // Mat.ptr() acts on the dimension #0 (rows)
        img_ptr = patch.roi.ptr(y);
        for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
        {
          // precompute interpolated reference patch color
          *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

          // we use the inverse compositional: thereby we can take the gradient always at the same position
          // get gradient of warped image (~gradient at warped position)
          float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
              -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
          float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
              -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

          // cache the jacobian
          pt_cache_.jacobian.col(feature_counter*patch.area + pixel_counter) =
              (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
        }
      }
    }
  }

}

void SparseImgAlign::precomputeGaussNewtonParamsSegments(Cache &cache, list<LineFeat*> &fts)
{
    // initialize patch parameters (mainly define its geometry)
    Patch patch( patch_size_, ref_frame_->img_pyr_.at(level_) );

    const float scale         = 1.0f/(1<<level_);
    const Vector3d ref_pos    = ref_frame_->pos();
    const double focal_length = ref_frame_->cam_->errorMultiplier2();

    // TODO: feature_counter is no longer valid because each segment
    // has a variable number of patches (and total pixels)
    std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
    patch_offset = std::vector<size_t>(fts.size(),0); // vector of offsets in cache for each patch
    std::vector<size_t>::iterator offset_it = patch_offset.begin();
    size_t cache_idx = 0; // index of the current pixel as stored in cache
    for(auto ft_it=fts.begin(), ft_ite=fts.end();
        ft_it!=ft_ite; ++ft_it, ++visiblity_it, ++offset_it)
    {
      // cast generic Feature* iterator to a LineFeat* pointer here to fully control the object
      LineFeat* it = static_cast<LineFeat*>( *ft_it );

      // set cache index to current feature offset
      *offset_it = cache_idx;

      // if line segment is not valid (empty or null) skip this feature
      if( it->feat3D == NULL )
        continue;

      // skip this feature if the patches for start or end points do not fully lie within the image
      if(!ref_frame_->cam_->isInFrame((it->spx*scale).cast<int>(),patch.halfsize+1,level_) ||
         !ref_frame_->cam_->isInFrame((it->epx*scale).cast<int>(),patch.halfsize+1,level_))
        continue;

      // flag the feature as valid/visible
      *visiblity_it = true;

      // 1. Estimate number of samples (TODO: first, we have implemented for a fixed N_samples)
      // 2. Compute discrete increment between 2D and 3D points from start to end
      //      i.   Estimate depth at extreme points
      //      ii.  Estimate xyz_ref at extreme points
      //      iii. Get 3D increment
      // 3. Iterate over the segment
      //      i.   Estimate jacobian
      //      ii.  Iterate around the points of the patch

      // Compute the number of samples and total increment
      Vector2d inc2d; // will store the difference vector from start to end points in the segment first
      // later will parameterize the 2D step to sample the segment
      size_t N_samples = it->setupSampling(patch.size, inc2d);
      // Adjust the number of samples in terms of the current pyramid level
      N_samples = 1 + (N_samples-1) / (1<<level_); // for lvl 0 take all, for lvl n downsample by 2^n

      // Parameterize 2D segment
      inc2d = inc2d * scale / (N_samples-1); // -1 to get nr of intervals
      Vector2d px_ref = it->spx * scale; // 2D point in the image segment (to update in the loop), initialize at start 2D point

      // Parameterize 3D segment with start point and discrete 3D increment
      double p_depth = (it->feat3D->spos_-ref_pos).norm(); // depth (norm of the vector) of segment 3D start point
      Vector3d p_ref = it->sf * p_depth; // 3D start point in the frame coordinates
      double q_depth = (it->feat3D->epos_-ref_pos).norm(); // depth (norm of the vector) of segment 3D end point
      Vector3d q_ref = it->ef * q_depth; // 3D start point in the frame coordinates
      Vector3d inc3d = (q_ref-p_ref) / (N_samples-1); // 3D increment to go from start to end of segment in N steps
      Vector3d xyz_ref = p_ref; // 3D point in the segment (to update it in the loop), initialized at start 3D point

      // Evaluate over the patch for each point sampled in the segment (including extremes)
      for(unsigned int sample = 0; sample<N_samples; ++sample, px_ref+=inc2d, xyz_ref+=inc3d )
      {
        // set patch position for current point in the segment
        patch.setPosition( px_ref );
        // compute the bilinear interpolation weights constant along the patch scan
        patch.computeInterpWeights();
        // set the patch at the corresponding ROI in the image
        patch.setRoi();

        // evaluate projection jacobian
        Matrix<double,2,6> frame_jac;
        Frame::jacobian_xyz2uv(xyz_ref, frame_jac);

        // iterate through all points in the Region Of Interest defined by the patch
        // the pointer points to the data in the original image matrix
        // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
        float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
        uint8_t* img_ptr;                 // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
        const int stride = patch.stride;  // the stride stored in the patch is that necessary to jump between the full matrix rows
        for(int y=0; y<patch.size; ++y)   // sweep the path row-wise (most efficient for RowMajor storage)
        {
          // get the pointer to first element in row y of the patch ROI
          // Mat.ptr() acts on the dimension #0 (rows)
          img_ptr = patch.roi.ptr(y);
          for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
          {
            // precompute interpolated reference patch color
            *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

            // we use the inverse compositional: thereby we can take the gradient always at the same position
            // get gradient of warped image (~gradient at warped position)
            float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
            float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

            // cache the jacobian
            seg_cache_.jacobian.col(cache_idx) =
                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
          }//end col-sweep in current row
        }//end row-sweep
      }//end segment-sweep (through sampled patches)
    }//end feature-sweep
}

void SparseImgAlign::computeGaussNewtonParamsPoints(
    const SE3& T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale,
    Cache& cache,
    list<PointFeat*> &fts,
    Matrix<double, 6, 6> &H,
    Matrix<double, 6, 1> &Jres,
    std::vector<float>& errors,
    float& chi2)
{
  // initialize patch parameters (mainly define its geometry)
  Patch patch( patch_size_, cur_frame_->img_pyr_.at(level_) );
  Patch resPatch;
  if(linearize_system && display_)
    resPatch = Patch( patch_size_, resimg_ );

  // compute the weights on the first iteration

  if(compute_weight_scale)
    errors.reserve(cache.visible_fts.size());

  const float scale = 1.0f/(1<<level_);
  const Vector3d ref_pos(ref_frame_->pos());

  // reset chi2 variable to zero
  chi2 = 0.0;
  vk::robust_cost::TukeyWeightFunction weight_estimator;

  // set GN parameters to zero prior to accumulate results
  H.setZero();
  Jres.setZero();
  size_t feature_counter = 0; // is used to compute the index of the cached jacobian
  std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
  for(auto it=fts.begin(); it!=fts.end();
      ++it, ++feature_counter, ++visiblity_it)
  {
    // check if feature is within image
    if(!*visiblity_it)
      continue;

    // compute pixel location in cur img
    const double depth = ((*it)->feat3D->pos_ - ref_pos).norm();
    const Vector3d xyz_ref((*it)->f*depth);
    const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    const Vector2d uv_cur_pyr(cur_frame_->cam_->world2cam(xyz_cur) * scale);

    // set patch position for current feature
    patch.setPosition(uv_cur_pyr);
    // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
    if(!patch.isInFrame(patch.halfsize))
      continue;
    // compute the bilinear interpolation weights constant along the patch scan
    patch.computeInterpWeights();
    // set the patch at the corresponding ROI in the image
    patch.setRoi();
    // iterate through all points in the Region Of Interest defined by the patch
    // the pointer points to the data in the original image matrix
    // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
    size_t pixel_counter = 0;
    float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
    uint8_t* img_ptr; // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
    const int stride = patch.stride; // the stride stored in the patch is that necessary to jump between the full matrix rows
    cv::MatIterator_<float> itDisp;
    if(linearize_system && display_)
    {
      resPatch.setPosition(uv_cur_pyr);
      resPatch.setRoi();
      itDisp = resPatch.roi.begin<float>();
    }
    for(int y=0; y<patch.size; ++y) // sweep the path row-wise (most efficient for RowMajor storage)
    {
      // get the pointer to first element in row y of the patch ROI
      // Mat.ptr() acts on the dimension #0 (rows)
      img_ptr = patch.roi.ptr(y);
      for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
      {
        // compute residual
        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
        const float res = intensity_cur - (*cache_ptr);
        const float res2 = res*res;

        // used to compute scale for robust cost
        if(compute_weight_scale)
          errors.push_back(fabsf(res));

        // robustification
        float weight = 1.0;
        if(use_weights_)
        {
          if(compute_weight_scale && iter_ != 0)
          {
              //weight = 2.0*fabsf(res) / (1.0+res2/scale_pt);
              weight = 1.0 / (1.0+fabsf(res)/scale_pt);
              //weight = weight_estimator.value(fabsf(res)/scale_pt);
          }
          else
          {
              //weight = 2.0*fabsf(res) / (1.0+res2);
              weight = 1.0 / (1.0+fabsf(res));
              //weight = weight_estimator.value(fabsf(res));
          }
        }

        chi2 += res*res*weight;
        n_meas_++;

        if(linearize_system)
        {
          // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
          const Vector6d J(cache.jacobian.col(feature_counter*patch.area + pixel_counter));
          H.noalias() += J*J.transpose()*weight;
          Jres.noalias() -= J*res*weight;
          if(display_)
          {
            *itDisp = res/255.0;
            ++itDisp;
          }
        }
      }
    }
  }
}

void SparseImgAlign::computeGaussNewtonParamsSegments(
    const SE3 &T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale,
    SparseImgAlign::Cache &cache,
    list<LineFeat*> &fts,
    Matrix<double, 6, 6> &H,
    Matrix<double, 6, 1> &Jres,
    std::vector<float> &errors,
    float &chi2)
{

  // initialize patch parameters (mainly define its geometry)
  Patch patch( patch_size_, cur_frame_->img_pyr_.at(level_) );
  Patch resPatch;
  if(linearize_system && display_)
    resPatch = Patch( patch_size_, resimg_ );

  // compute the weights on the first iteration
  if(compute_weight_scale)
    errors.reserve(cache.visible_fts.size());
  const float scale = 1.0f/(1<<level_);
  const Vector3d ref_pos(ref_frame_->pos());

  // reset chi2 variable to zero
  chi2 = 0.0;
  vk::robust_cost::TukeyWeightFunction weight_estimator;

  // set GN parameters to zero prior to accumulate results
  H.setZero();
  Jres.setZero();
  std::vector<size_t>::iterator offset_it = patch_offset.begin();
  size_t cache_idx = 0; // index of the current pixel as stored in cache
  std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

  Matrix<double,6,6> H_ls    = Matrix<double, 6, 6>::Zero();
  Matrix<double,6,1> Jres_ls = Matrix<double, 6, 1>::Zero();

  for(auto ft_it=fts.begin(), ft_ite=fts.end();
      ft_it!=ft_ite; ++ft_it, ++offset_it, ++visiblity_it)
  {
    // cast generic Feature* iterator to a LineFeat* pointer here to fully control the object
    LineFeat* it = static_cast<LineFeat*>( *ft_it );

    // check if we have already removed this line feature
    if( it->feat3D == NULL)
      continue;

    // check if feature is within image
    if(!*visiblity_it)
      continue;

    // setup current index in cache according to stored offset values
    cache_idx = *offset_it;

    // Compute the number of samples and total increment
    Vector2d inc2d; // will store the difference vector from start to end points in the segment first
                    // later will parameterize the 2D step to sample the segment
    size_t N_samples = it->setupSampling(patch.size, inc2d);

    // Adjust the number of samples in terms of the current pyramid level
    N_samples = 1 + (N_samples-1) / (1<<level_); // for lvl 0 take all, for lvl n downsample by 2^n

    // Parameterize 3D segment with start point and discrete 3D increment
    double p_depth = (it->feat3D->spos_-ref_pos).norm(); // depth (norm of the vector) of segment 3D start point
    Vector3d p_ref = it->sf * p_depth; // 3D start point in the frame coordinates
    double q_depth = (it->feat3D->epos_-ref_pos).norm(); // depth (norm of the vector) of segment 3D end point
    Vector3d q_ref = it->ef * q_depth; // 3D start point in the frame coordinates
    Vector3d inc3d = (q_ref-p_ref) / (N_samples-1); // 3D increment to go from start to end of segment in N steps
    Vector3d xyz_ref = p_ref; // 3D point in the segment (to update it in the loop), initialized at start 3D point

    // Evaluate over the patch for each point sampled in the segment (including extremes)
    Matrix<double,6,6> H_    = Matrix<double, 6, 6>::Zero();
    Matrix<double,6,1> Jres_ = Matrix<double, 6, 1>::Zero();
    vector<float> ls_res;
    bool good_line = true;
    for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3d )
    {
      // compute pixel location in cur img
      const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
      const Vector2d uv_cur_pyr(cur_frame_->cam_->world2cam(xyz_cur) * scale);
      // set patch position for current feature
      patch.setPosition(uv_cur_pyr);
      // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
      if(!patch.isInFrame(patch.halfsize))
      {
        cache_idx += patch.size; // Do not lose position of the next patch in cache!
        good_line = false;
        sample    = N_samples;
        continue;
      }
      // compute the bilinear interpolation weights constant along the patch scan
      patch.computeInterpWeights();
      // set the patch at the corresponding ROI in the image
      patch.setRoi();
      // iterate through all points in the Region Of Interest defined by the patch
      // the pointer points to the data in the original image matrix
      // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
      float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
      uint8_t* img_ptr; // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
      const int stride = patch.stride; // the stride stored in the patch is that necessary to jump between the full matrix rows
      cv::MatIterator_<float> itDisp;
      if(linearize_system && display_)
      {
        resPatch.setPosition(uv_cur_pyr);
        resPatch.setRoi();
        itDisp = resPatch.roi.begin<float>();
      }
      for(int y=0; y<patch.size; ++y) // sweep the path row-wise (most efficient for RowMajor storage)
      {
        // get the pointer to first element in row y of the patch ROI
        // Mat.ptr() acts on the dimension #0 (rows)
        img_ptr = patch.roi.ptr(y);
        for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
        {
          // compute residual
          const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
          const float res = intensity_cur - (*cache_ptr);
          ls_res.push_back(res);
          // if not?
          if(linearize_system)
          {
            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
            const Vector6d J(cache.jacobian.col(cache_idx));
            H_.noalias()     += J*J.transpose() ;
            Jres_.noalias()  -= J*res;
            if(display_)
            {
              *itDisp = res/255.0;
              ++itDisp;
            }
          }
        }//end col-sweep of current row
      }//end row-sweep
    }//end segment-sweep

    float res_ = 0.0, res2_;
    for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
      //res_ += pow(*it,2);
      res_ += fabsf(*it);
      //res_ += pow(*it,2);
    }
    //res_ = sqrt(res_)/double(N_samples) ;
    res_ = res_ / double(N_samples);
    if( good_line && res_ < 200.0)  // debug
    {
        /*float res_ = 0.0, res2_;
        for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
          //res_ += pow(*it,2);
          res_ += fabsf(*it);
          //res_ += pow(*it,2);
        }
        //res_ = sqrt(res_)/double(N_samples) ;
        res_ = res_ / double(N_samples);*/

        // used to compute scale for robust cost
        if(compute_weight_scale)
          errors.push_back(res_);
        // robustification
        float weight = 1.0;
        if(use_weights_)
        {
          if(compute_weight_scale && iter_ != 0)
          {
              //weight = 2.0*fabsf(res) / (1.0+res2/scale_pt);
              weight = 1.0 / (1.0+res_/scale_ls);
              //weight = weight_estimator.value(res_/scale_ls);
          }
          else
          {
              //weight = 2.0*fabsf(res) / (1.0+res2);
              weight = 1.0 / (1.0+res_);
              //weight = weight_estimator.value(res_);
          }
        }

        // update total H and J
        H.noalias()    += H_    * weight / res_;  // only divide hessian once (H/res2 g/res)
        Jres.noalias() += Jres_ * weight ;        // it is already negative
        chi2           += res_*res_*weight;
        n_meas_++;

    }
    else
        it->feat3D = NULL;

    good_line = true;
    ls_res.clear();
    H_.setZero();
    Jres_.setZero();
  }//end feature-sweep
}

int SparseImgAlign::solve()
{
  x_ = H_.ldlt().solve(Jres_);
  if((bool) std::isnan((double) x_[0]))
    return 0;
  return 1;
}

void SparseImgAlign::update(
    const ModelType& T_curold_from_ref,
    ModelType& T_curnew_from_ref)
{
  T_curnew_from_ref =  T_curold_from_ref * SE3::exp(-x_);
}

void SparseImgAlign::startIteration()
{}

void SparseImgAlign::finishIteration()
{
  if(display_)
  {
    //cv::namedWindow("residuals", cv::WINDOW_AUTOSIZE);
    //cv::imshow("residuals", resimg_*10);
    //cv::waitKey(0);
  }
}

} // namespace plsvo
