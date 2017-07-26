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

#include <iomanip>
#include <string>

#ifdef HAS_MRPT
#include <mrpt/opengl.h>
#include <mrpt/gui.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileBase.h>
using namespace mrpt::math;
using namespace mrpt::poses;
using namespace mrpt::opengl;
using namespace mrpt::gui;
using namespace mrpt::utils;
#endif
#include <eigen3/Eigen/Eigen>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

// Auxiliar functions
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

class sceneRepresentation{

public:

    sceneRepresentation();
    sceneRepresentation(string configFile);
    ~sceneRepresentation();
    void initialize3DScene(Matrix<double,4,4> x_0);
    void initialize3DSceneLines(Matrix<double,4,4> x_0);
    void initialize3DSceneImg(Matrix<double,4,4> x_0);
    void initialize3DSceneGT(Matrix<double,4,4> x_0);

    void initializeScene(Matrix<double,4,4> x_0);
    void initializeScene(Matrix<double,4,4> x_0, Matrix<double,4,4> x_0gt);

    bool updateScene();

    void setText(int frame_, float time_, int nPoints_, int nPointsH_=0, int nLines_=0, int nLinesH_=0);
    void setPose(Matrix<double,4,4> x_);
    void setGT(Matrix<double,4,4> xgt_);
    void setComparison(Matrix<double,4,4> xcomp_);
    void setImage(Mat image_);
    void setImage(string image_);
    void setLegend();
    void setHelp();
    void setPoints(CMatrixFloat pData_);
    void setLines(CMatrixFloat lData_);

    void setPointsSVO(vector< Matrix<double,3,1> > pointsSVO_);
    void setLinesSVO( vector< Matrix<double,6,1> > linesSVO_);

    void setStereoCalibration(Matrix3f K_, float b_);


    bool waitUntilClose();
    bool isOpen();
    void getYPR(float &yaw, float &pitch, float &roll);
    void getPose(Matrix<double,4,4> &T);
    
    CImage          img_mrpt_legend, img_mrpt_image, img_mrpt_help;
    bool hasGT;

private:

    CMatrixDouble getPoseFormat(Matrix<double,4,4> T);
    CMatrixDouble33 getCovFormat(MatrixXf cov_);
    CPose3D getPoseXYZ(VectorXf x);

    CDisplayWindow3D*           win;
    COpenGLScenePtr             theScene;
    COpenGLViewportPtr          image, legend, help;
    CSetOfObjectsPtr    srefObj, srefObj1, gtObj, srefObjGT, elliObjL, elliObjP;
    CEllipsoidPtr       elliObj;
    CSetOfLinesPtr      lineObj;
    CPointCloudPtr      pointObj;
    CFrustumPtr         frustObj, frustObj1, bbObj, bbObj1;
    CAxisPtr            axesObj;

    vector< Matrix<double,3,1> > pointsSVO;
    vector< Matrix<double,6,1> > linesSVO;

    float           sbb, saxis, srad, sref, sline, sfreq, szoom, selli, selev, sazim, sfrust, slinef;
    CVectorDouble   v_aux, v_aux_, v_aux1, v_aux1_, v_auxgt, gt_aux_, v_auxgt_;
    CPose3D         pose, pose_0, pose_gt, pose_ini, ellPose, pose1,  change, frustumL_, frustumR_;
    Matrix<double,4,4>        x_ini;
    mrptKeyModifier kmods;
    int             key;
    CMatrixDouble33 cov3D;
    bool            hasText, hasCov, hasChange, hasImg, hasLines, hasPoints, hasFrustum, hasComparison, hasLegend, hasHelp, hasAxes, hasTraj, isKitti;

    Matrix<double,4,4>        x, xgt, xcomp;
    MatrixXf        cov, W;
    unsigned int    frame, nPoints, nPointsH, nLines, nLinesH;
    float           time;
    string          img, img_legend, img_help;
    CMatrixFloat    lData, pData;

    float           b, sigmaP, sigmaL, f, cx, cy, bsigmaL, bsigmaP;

};

