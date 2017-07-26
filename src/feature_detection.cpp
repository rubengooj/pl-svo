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

#include <plsvo/feature_detection.h>
#include <plsvo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

using namespace cv;
using namespace cv::line_descriptor;

struct compare_line_by_unscaled_length
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return (a.lineLength > b.lineLength);
    }
};

namespace plsvo {
namespace feature_detection {

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    list<PointFeat*>& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
#if __SSE2__
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif
	// nm stands for non-maximal
    vector<int> scores, nm_corners;
	// compute scores for all fast corners
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
	// get list of nonmax corners in a 3x3 window
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])
        continue;
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }

  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
      fts.push_back(new PointFeat(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}

void FastDetector::setExistingFeatures(const list<PointFeat*> &fts)
{
  std::for_each(fts.begin(), fts.end(), [&](PointFeat* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void FastDetector::setGridOccpuancy(const PointFeat &ft)
{
  const Vector2d& px = ft.px;
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

LsdDetector::LsdDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
  AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void LsdDetector::detect(Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    list<LineFeat*> &fts)
{
    // Parameters (TODO: include into config file or whatever and commit it to opencv)
    vector<KeyLine> keyline, keylines, keylines_sort;

    // Define the LSD detector object
    Ptr<LSDDetectorC> lsd = LSDDetectorC::createLSDDetectorC();

    // TODO: grab from config file
    LSDDetectorC::LSDOptions opts;
    opts.refine       = LSD_REFINE_ADV;
    opts.scale        = 1.2;
    opts.sigma_scale  = 0.6;
    opts.quant        = 2.0;
    opts.ang_th       = 22.5;
    opts.log_eps      = 1.0;
    opts.density_th   = 0.6;
    opts.n_bins       = 1024;

    // Detect features on each pyramid level
    double detection_threshold_ = detection_threshold * double(img_height_*img_width_) / double(img_height_+img_width_);    // min length relative to size of image (change)
    for(int L=0; L<n_pyr_levels_; ++L)
    {
        // detect lines in pyramid level L
        const double scale = double(1<<L);
        lsd->detect( img_pyr[L], keyline, 1, 1, opts);
        // sort lines according to their unscaled length
        sort( keyline.begin(), keyline.end(), compare_line_by_unscaled_length() );
        //detection_threshold_ = keyline[int(0.25*double(keyline.size()))].lineLength ;
        std::for_each( keyline.begin(), keyline.end(), [&](KeyLine kl){
          if( scale*kl.lineLength > detection_threshold_ ){
              const int sk = static_cast<int>((kl.startPointY*scale)/cell_size_)*grid_n_cols_
                           + static_cast<int>((kl.startPointX*scale)/cell_size_);
              const int mk = static_cast<int>(((kl.startPointY+kl.endPointY)*scale/2)/cell_size_)*grid_n_cols_
                           + static_cast<int>(((kl.startPointY+kl.endPointY)*scale/2)/cell_size_);
              const int ek = static_cast<int>((kl.endPointY*scale)/cell_size_)*grid_n_cols_
                           + static_cast<int>((kl.endPointX*scale)/cell_size_);
              if( !grid_occupancy_[sk] && !grid_occupancy_[ek] && !grid_occupancy_[mk] )
              //if( !grid_occupancy_[sk] && !grid_occupancy_[ek] )
              {
                  fts.push_back( new LineFeat(frame,Vector2d(kl.startPointX*scale,kl.startPointY*scale),Vector2d(kl.endPointX*scale,kl.endPointY*scale),L,kl.angle) );
              }
          }
        });
        keyline.clear();
    }
    resetGridLs();

}

void LsdDetector::detect(Frame* frame,
    const cv::Mat& rec_img,
    const double detection_threshold,
    list<LineFeat*> &fts)
{
    // Parameters (TODO: include into config file or whatever and commit it to opencv)
    vector<KeyLine> keyline, keylines, keylines_sort;

    // Define the LSD detector object
    Ptr<LSDDetectorC> lsd = LSDDetectorC::createLSDDetectorC();

    // TODO: grab from config file
    LSDDetectorC::LSDOptions opts;
    opts.refine       = LSD_REFINE_STD;
    opts.scale        = 1.0;
    opts.sigma_scale  = 0.6;
    opts.quant        = 2.0;
    opts.ang_th       = 22.5;
    opts.log_eps      = 0.0;
    opts.density_th   = 0.7;
    opts.n_bins       = 1024;

    // Detect features on each pyramid level
    double detection_threshold_ = detection_threshold * double(img_height_*img_width_) / double(img_height_+img_width_);    // min length relative to size of image (change)

    lsd->detect( rec_img, keyline, 1, 1, opts);
    // sort lines according to their unscaled length
    sort( keyline.begin(), keyline.end(), compare_line_by_unscaled_length() );

    detection_threshold_ = keyline[int(0.5*double(keyline.size()))].lineLength * 2.0;

    std::for_each( keyline.begin(), keyline.end(), [&](KeyLine kl){
      if( kl.lineLength > detection_threshold_ ){
          const int sk = static_cast<int>((kl.startPointY)/cell_size_)*grid_n_cols_
                       + static_cast<int>((kl.startPointX)/cell_size_);
          /*const int mk = static_cast<int>(((kl.startPointY+kl.endPointY)*scale/2)/cell_size_)*grid_n_cols_
                       + static_cast<int>(((kl.startPointY+kl.endPointY)*scale/2)/cell_size_);*/
          const int ek = static_cast<int>((kl.endPointY)/cell_size_)*grid_n_cols_
                       + static_cast<int>((kl.endPointX)/cell_size_);
          //if( !grid_occupancy_[sk] && !grid_occupancy_[ek] && !grid_occupancy_[mk] )
          if( !grid_occupancy_[sk] && !grid_occupancy_[ek] )
          {
              fts.push_back( new LineFeat(frame,Vector2d(kl.startPointX,kl.startPointY),Vector2d(kl.endPointX,kl.endPointY),kl.octave) );
          }
      }
    });
    keyline.clear();

    resetGridLs();

}



void LsdDetector::setGridOccpuancy(const LineFeat& ft)
{
  const Vector2d& spx = ft.spx;
  grid_occupancy_.at(
      static_cast<int>(spx[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(spx[0]/cell_size_)) = true;
  const Vector2d& epx = ft.epx;
  grid_occupancy_.at(
      static_cast<int>(epx[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(epx[0]/cell_size_)) = true;
  // mid point
  /*grid_occupancy_.at(
      static_cast<int>((epx[1]+spx[1])/(2*cell_size_))*grid_n_cols_
    + static_cast<int>((epx[0]+spx[0])/(2*cell_size_))) = true;*/
}

void LsdDetector::setExistingFeatures(const list<LineFeat*>& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](LineFeat* i){
    grid_occupancy_.at(
        static_cast<int>(i->spx[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->spx[0]/cell_size_)) = true;
    grid_occupancy_.at(
        static_cast<int>(i->epx[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->epx[0]/cell_size_)) = true;
    // mid point
    /*grid_occupancy_.at(
        static_cast<int>((i->epx[1]+i->spx[1])/(2*cell_size_))*grid_n_cols_
      + static_cast<int>((i->epx[0]+i->spx[0])/(2*cell_size_))) = true;*/
  });
}

} // namespace feature_detection
} // namespace plsvo

