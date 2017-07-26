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


#include <plsvo/sceneRepresentation.h>

// Constructors and destructor

sceneRepresentation::sceneRepresentation(){

    sbb     = 1.0f;
    sref    = 0.2f;
    srad    = 0.005f;
    sline   = 2.0f;
    saxis   = 0.5f;
    sfreq   = 3.0f;
    szoom   = 3.0f;
    selli   = 2.0f;
    selev   = 30.f;
    sazim   = -135.f;
    sfrust  = 0.2f;
    slinef  = 0.1f;
    win     = new CDisplayWindow3D("3D Scene",1920,1080);

    hasText         = true;
    hasAxes         = false;
    hasLegend       = false;
    hasHelp         = false;
    hasCov          = false;
    hasTraj         = true;
    hasGT           = false;
    hasChange       = false;
    hasComparison   = false;
    hasImg          = false;
    hasLines        = false;
    hasPoints       = false;
    hasFrustum      = false;

}

sceneRepresentation::sceneRepresentation(string configFile){

    CConfigFile config(configFile);
    sbb             = config.read_double("Scene","sbb",1.f);
    sref            = config.read_double("Scene","sref",0.2f);
    srad            = config.read_double("Scene","srad",0.005f);
    sline           = config.read_double("Scene","sline",2.f);
    saxis           = config.read_double("Scene","saxis",0.5f);
    sfreq           = config.read_double("Scene","sfreq",3.f);
    szoom           = config.read_double("Scene","szoom",3.f);
    selli           = config.read_double("Scene","selli",1.f);
    selev           = config.read_double("Scene","selev",30.f);
    sazim           = config.read_double("Scene","sazim",-135.f);
    sfrust          = config.read_double("Scene","sfrust",0.2f);
    slinef          = config.read_double("Scene","slinef",0.1f);
    win             = new CDisplayWindow3D("3D Scene",1920,1080);

    hasText         = config.read_bool("Scene","hasText",true);
    hasAxes         = config.read_bool("Scene","hasAxes",true);
    hasLegend       = config.read_bool("Scene","hasLegend",false);
    hasHelp         = config.read_bool("Scene","hasHelp",false);
    hasCov          = config.read_bool("Scene","hasCov",false);
    hasGT           = config.read_bool("Scene","hasGT",false);
    hasTraj         = config.read_bool("Scene","hasTraj",true);
    hasChange       = config.read_bool("Scene","hasChange",false);
    hasComparison   = config.read_bool("Scene","hasComparison",false);
    hasImg          = config.read_bool("Scene","hasImg",true);
    hasLines        = config.read_bool("Scene","hasLines",false);
    hasPoints       = config.read_bool("Scene","hasPoints",false);
    hasFrustum      = config.read_bool("Scene","hasFrustum",false);
    isKitti         = config.read_bool("Scene","isKitti",false);

    Matrix<double,4,4> x_cw;
    x_cw << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    CPose3D x_aux(getPoseFormat(x_cw));
    pose_ini = x_aux;

}

sceneRepresentation::~sceneRepresentation(){

}

// Initialize the scene

void sceneRepresentation::initializeScene(Matrix<double,4,4> x_0){

    // Initialize the scene and window
    win->setCameraElevationDeg(selev);
    win->setCameraAzimuthDeg(sazim);
    win->setCameraZoom(szoom);
    theScene = win->get3DSceneAndLock();

    // Initialize poses of different objects
    if(hasChange)
        change.setFromValues(0,0,0,0,0,-90*CV_PI/180.f);
    else
        change.setFromValues(0,0,0,0,0,0);
    CPose3D x_aux(getPoseFormat(x_0));
    pose    = x_aux;
    pose1   = x_aux;
    pose_0  = x_aux;
    pose_gt = pose_ini - change;
    x_ini   = x_0;
    pose.getAsVector(v_aux);
    pose1.getAsVector(v_aux1);
    pose_gt.getAsVector(v_auxgt);

    // Initialize the camera object
    //bbObj  = stock_objects::BumblebeeCamera();
    bbObj  = CFrustum::Create();
    {
        bbObj->setPose(pose_0+frustumL_);
        bbObj->setScale(sbb/15.0);
        bbObj->setColor(0.0,0.0,0.0);
        theScene->insert(bbObj);
    }
    srefObj = stock_objects::CornerXYZSimple();
    srefObj->setPose(pose_0);
    srefObj->setScale(sref);
    theScene->insert(srefObj);
    frustumL_.setFromValues(0,0,0,  0, -90.f*CV_PI/180.f, -90.f*CV_PI/180.f);
    frustumR_.setFromValues(0.12f,0,0,  0, -90.f*CV_PI/180.f, -90.f*CV_PI/180.f);
    if(hasFrustum){
        frustObj = CFrustum::Create();
        {
            frustObj->setPose(pose_0+frustumL_);
            frustObj->setLineWidth (slinef);
            frustObj->setScale(sfrust);
            theScene->insert(frustObj);
        }
        frustObj1 = CFrustum::Create();
        {
            frustObj1->setPose(pose_0+frustumR_);
            frustObj1->setLineWidth (slinef);
            frustObj1->setScale(sfrust);
            theScene->insert(frustObj1);
        }
    }

    // Initialize the axes
    if(hasAxes){
        axesObj = CAxis::Create();
        axesObj->setFrequency(sfreq);
        axesObj->enableTickMarks(false);
        axesObj->setAxisLimits(-saxis,-saxis,-saxis, saxis,saxis,saxis);
        theScene->insert( axesObj );
    }

    // Initialize the ground truth camera object
    if(hasGT){
        gtObj = stock_objects::BumblebeeCamera();
        {
            gtObj->setPose(pose_ini);
            gtObj->setScale(sbb);
            theScene->insert(gtObj);
        }
        srefObjGT = stock_objects::CornerXYZSimple();
        {
            srefObjGT->setPose(pose_ini);
            srefObjGT->setScale(sref);
            theScene->insert(srefObjGT);
        }
    }

    // Initialize a second camera to compare when changing parameters
    if(hasComparison){
        //bbObj1 = stock_objects::BumblebeeCamera();
        bbObj1  = CFrustum::Create();
        {
            bbObj1->setPose(pose_0+frustumL_);
            bbObj1->setScale(sbb/15.0);
            bbObj1->setColor(0.0,0.0,0.0);
            theScene->insert(bbObj1);
        }
        srefObj1 = stock_objects::CornerXYZSimple();
        {
            srefObj1->setPose(pose_0);
            srefObj1->setScale(sref);
            theScene->insert(srefObj1);
        }
    }

    // Initialize the text
    if(hasText){
        string text = "Frame: \t \t0 \nFrequency: \t0 Hz \nLines:  \t0 (0)\nPoints: \t0 (0)";
        win->addTextMessage(0.85,0.95, text, TColorf(1.0,1.0,1.0), 0, MRPT_GLUT_BITMAP_HELVETICA_10 );
    }

    // Initialize the lines
    lineObj = CSetOfLines::Create();
    lineObj->setLineWidth(1.0);
    lineObj->setColor(0,0,0);
    theScene->insert( lineObj );
    
    // Initialize the point cloud
    pointObj = CPointCloud::Create();
    pointObj->setPointSize(3.0);
    pointObj->setColor(0,0,0);
    theScene->insert( pointObj );

    // Initialize the viewports
    setHelp();
    setLegend();

    image = theScene->createViewport("image");
    if(isKitti){
        if(hasImg)
            image->setViewportPosition(20, 20, 621, 188);
        else
            image->setViewportPosition(2000, 2000, 621, 188);
    }
    else{
        if(hasImg)
            image->setViewportPosition(20, 20, 640, 480);
        else
            image->setViewportPosition(2000, 2000, 640, 480);
    }
    
    // BUG: i cannot check if CImage is empty
    if(hasImg){
        img_mrpt_image.loadFromFile("./empty_img.png",1);
    }

    // Re-paint the scene
    win->unlockAccess3DScene();
    win->repaint();

}

// Update the scene

bool sceneRepresentation::updateScene(){

    theScene = win->get3DSceneAndLock();
    bool restart = false;

    // Update camera pose
    CPose3D x_aux(getPoseFormat(x));
    pose = pose + x_aux;
    v_aux_ = v_aux;
    pose.getAsVector(v_aux);
    if(hasTraj){
        CSimpleLinePtr obj = CSimpleLine::Create();
        obj->setLineCoords(v_aux_(0),v_aux_(1),v_aux_(2), v_aux(0),v_aux(1),v_aux(2));
        obj->setLineWidth(sline);
        obj->setColor(0,0,0.7);
        theScene->insert( obj );
    }
    bbObj->setPose(pose+frustumL_);
    srefObj->setPose(pose);
    if(hasFrustum){
        frustObj->setPose(pose+frustumL_);
        frustObj1->setPose(pose+frustumR_);
    }

    // Update the GT camera pose
    if(hasGT){
        CPose3D x_auxgt(getPoseFormat(xgt));
        //pose_gt = pose_gt + x_auxgt;
        pose_gt = x_auxgt;
        v_auxgt_ = v_auxgt;

        pose_gt.getAsVector(v_auxgt);
        float y_ = v_auxgt(1);
        float z_ = v_auxgt(2);
        float b_ = v_auxgt(4);
        float c_ = v_auxgt(5);
        v_auxgt(1) =  z_;
        v_auxgt(2) = -y_;        
        v_auxgt(4) = -c_;
        v_auxgt(5) =  b_;
        pose_gt = TPose3D(v_auxgt(0),v_auxgt(1),v_auxgt(2),v_auxgt(3),v_auxgt(4),v_auxgt(5));

        if(hasTraj){
            CSimpleLinePtr obj = CSimpleLine::Create();
            obj->setLineCoords(v_auxgt_(0),v_auxgt_(1),v_auxgt_(2), v_auxgt(0),v_auxgt(1),v_auxgt(2));
            obj->setLineWidth(sline);
            obj->setColor(0,0,0);
            theScene->insert( obj );
        }
        gtObj->setPose(pose_gt);
        srefObjGT->setPose(pose_gt);
    }

    // Update the comparison camera pose
    if(hasComparison){
        CPose3D x_aux1(getPoseFormat(xcomp));
        pose1 = pose1 + x_aux1;
        v_aux1_ = v_aux1;
        pose1.getAsVector(v_aux1);
        if(hasTraj){
            CSimpleLinePtr obj = CSimpleLine::Create();
            obj->setLineCoords(v_aux1_(0),v_aux1_(1),v_aux1_(2), v_aux1(0),v_aux1(1),v_aux1(2));
            obj->setLineWidth(sline);
            obj->setColor(0,0.7,0);
            theScene->insert( obj );
        }
        bbObj1->setPose(pose1+frustumL_);
        srefObj1->setPose(pose1);
    }

    // Update the text
    if(hasText){
        string text = "Frame: \t \t" + to_string(frame) + " \n" + "Frequency: \t" + to_string_with_precision(1000.f/time,4) + " Hz \n" + "Lines:  \t" + to_string(nLines) + " (" + to_string(nLinesH) + ") \nPoints: \t" + to_string(nPoints) + " (" + to_string(nPointsH) + ")";
        win->addTextMessage(0.85,0.95, text, TColorf(1.0,1.0,1.0), 0, MRPT_GLUT_BITMAP_HELVETICA_10 );
    }

    // Update the image
    if(hasImg){
        //image->setImageView_fast( img_mrpt_image );
        image->setImageView( img_mrpt_image );
    }

    // Update the lines
    lineObj->clear();
    if(hasLines){
        for( vector<Matrix<double,6,1>>::iterator it = linesSVO.begin(); it!= linesSVO.end();++it)
            lineObj->appendLine((*it)(0),(*it)(1),(*it)(2),(*it)(3),(*it)(4),(*it)(5));
        lineObj->setPose(pose_0);
    }

    // Update the points
    pointObj->clear();
    if(hasPoints){        
        for( vector<Matrix<double,3,1>>::iterator it = pointsSVO.begin(); it!= pointsSVO.end();++it)
            pointObj->insertPoint( (*it)(0),(*it)(1),(*it)(2) );
        pointObj->setPose(pose_0);
    }

    // Re-paint the scene
    win->unlockAccess3DScene();
    win->repaint();

    // Key events   -       TODO: change the trick to employ viewports
    if(win->keyHit()){       
        key = win->getPushedKey(&kmods);
        if(key == MRPTK_SPACE){                     // Space    Reset VO
            theScene->clear();
            initializeScene(x_ini);
        }
        else if (key == MRPTK_ESCAPE){              // Esc      Restart VO
            theScene->clear();
            initializeScene(x_ini);
            restart = true;
        }
        else if ( (key == 104) || (key == 72) ){    // H        help
            hasHelp   = !hasHelp;
            if(!hasHelp)
                help->setViewportPosition(2000, 2000, 300, 376);
            else
                help->setViewportPosition(1600, 20, 300, 376);
        }
        else if ( (key == 103) || (key == 71) ){    // G        legend
            hasLegend   = !hasLegend;
            if(!hasLegend)
                legend->setViewportPosition(2000, 2000, 250, 97);
            else
                legend->setViewportPosition(20, 900, 250, 97);
        }
        else if ( (key ==  97) || (key == 65) ){    // A        axes
            hasAxes   = !hasAxes;
            if(!hasAxes){
                axesObj.clear();
            }
            else{
                axesObj = CAxis::Create();
                axesObj->setFrequency(sfreq);
                axesObj->enableTickMarks(false);
                axesObj->setAxisLimits(-saxis,-saxis,-saxis, saxis,saxis,saxis);
                theScene->insert( axesObj );
            }
        }
        else if ( (key == 102) || (key == 70) ){    // F        frustum
            hasFrustum   = !hasFrustum;
            if(!hasFrustum){
                frustObj.clear();
                frustObj1.clear();
            }
            else{
                frustObj = CFrustum::Create();
                frustObj->setLineWidth (slinef);
                frustObj->setScale(sfrust);
                frustObj->setPose(pose+frustumL_);
                theScene->insert(frustObj);
                frustObj1 = CFrustum::Create();
                frustObj1->setLineWidth (slinef);
                frustObj1->setScale(sfrust);
                frustObj1->setPose(pose+frustumR_);
                theScene->insert(frustObj1);
            }
        }
        else if ( (key == 112) || (key == 80) ){    // P        points
            hasPoints = !hasPoints;
            if(!hasPoints){
                elliObjP.clear();
            }
            else{
                elliObjP = CSetOfObjects::Create();
                elliObjP->setScale(selli);
                elliObjP->setPose(pose);
                theScene->insert(elliObjP);
            }
        }
        else if ( (key == 108) || (key == 76) ){    // L        lines
            hasLines  = !hasLines;
            if(!hasLines){
                elliObjL.clear();
            }
            else{
                elliObjL = CSetOfObjects::Create();
                elliObjL->setScale(selli);
                elliObjL->setPose(pose);
                theScene->insert(elliObjL);
            }
        }
        else if ( (key == 116) || (key == 84) ){    // T        text
            hasText  = !hasText;
            if(!hasText){
                string text = "";
                win->addTextMessage(0.85,0.95, text, TColorf(1.0,1.0,1.0), 0, MRPT_GLUT_BITMAP_HELVETICA_10 );
            }
        }
        else if ( (key ==  99) || (key == 67) ){    // C        covariance
            hasCov   = !hasCov;
            if(!hasCov){
                elliObj.clear();
            }
            else{
                elliObj = CEllipsoid::Create();
                elliObj->setScale(selli);
                elliObj->setQuantiles(2.0);
                elliObj->setColor(0,0.3,0);
                elliObj->enableDrawSolid3D(true);
                elliObj->setPose(pose);
                theScene->insert(elliObj);
            }
        }
        else if (  key == 43){                      // +        Increases the scale of the covariance ellipse
            selli += 0.5f;
            if(hasCov)
                elliObj->setScale(selli);
        }
        else if (  key == 45){                      // -        Decreases the scale of the covariance ellipse
            selli -= 0.5f;
            if(hasCov)
                elliObj->setScale(selli);
        }
        else if ( (key == 105) || (key == 73) ){    // I        image
            hasImg   = !hasImg;          
            if(isKitti){
                if(hasImg)
                    image->setViewportPosition(20, 20, 621, 188);
                else
                    image->setViewportPosition(2000, 2000, 621, 188);
            }
            else{
                if(hasImg)
                    image->setViewportPosition(20, 20, 400, 300);
                else
                    image->setViewportPosition(2000, 2000, 400, 300);
            }

        }
    }

    return restart;

}

// Setters

void sceneRepresentation::setText(int frame_, float time_, int nPoints_, int nPointsH_, int nLines_, int nLinesH_){
    frame    = frame_;
    time     = time_;
    nPoints  = nPoints_;
    nPointsH = nPointsH_;
    nLines   = nLines_;
    nLinesH  = nLinesH_;

}

void sceneRepresentation::setPose(Matrix<double,4,4> x_){
    x = x_;
}

void sceneRepresentation::setGT(Matrix<double,4,4> xgt_){
    xgt = xgt_;
}

void sceneRepresentation::setComparison(Matrix<double,4,4> xcomp_){
    xcomp = xcomp_;
}

void sceneRepresentation::setImage(Mat image_){
    imwrite("./img_aux.jpg",image_);
    img_mrpt_image.loadFromFile("./img_aux.jpg",1);
}

void sceneRepresentation::setImage(string image_){
    img_mrpt_image.loadFromFile(image_,1);
}

void sceneRepresentation::setLegend(){
    // Initialize the legend
    legend = theScene->createViewport("legend");
    if(hasLegend)
        legend->setViewportPosition(20, 900, 250, 97);
    else
        legend->setViewportPosition(2000, 2000, 250, 97);

    if(!hasGT) {
        if(hasComparison){
            img_legend = "aux/legend_comp.png";
            img_mrpt_legend.loadFromFile(img_legend,1);
            legend->setImageView_fast( img_mrpt_legend );
        }
        else
            img_mrpt_legend.loadFromFile("",1);
    }
    else if(hasComparison){
        img_legend = "aux/legend_full.png";
        img_mrpt_legend.loadFromFile(img_legend,1);
        legend->setImageView_fast( img_mrpt_legend );
    }
    else{
        img_legend = "aux/legend.png";
        img_mrpt_legend.loadFromFile(img_legend,1);
        legend->setImageView_fast( img_mrpt_legend );
    }
}

void sceneRepresentation::setHelp(){
    // Initialize the legend
    help = theScene->createViewport("help");
    img_help = "aux/help.png";
    img_mrpt_help.loadFromFile(img_help,1);
    help->setImageView_fast( img_mrpt_help );
    if(hasHelp)
        help->setViewportPosition(1600, 20, 300, 376);
    else
        help->setViewportPosition(2000, 2000, 300, 376);
}

void sceneRepresentation::setPoints(CMatrixFloat pData_){
    pData = pData_;
}

void sceneRepresentation::setLines(CMatrixFloat lData_){
    lData = lData_;
}

void sceneRepresentation::setPointsSVO(vector< Matrix<double,3,1> > pointsSVO_){
    pointsSVO = pointsSVO_;
}

void sceneRepresentation::setLinesSVO(vector< Matrix<double,6,1> > linesSVO_){
    linesSVO  = linesSVO_;
}

void sceneRepresentation::setStereoCalibration(Matrix3f K_, float b_){
    f  = K_(0,0);
    cx = K_(0,2);
    cy = K_(1,2);
    b  = b_;
    sigmaP = 1.f;   // TODO: READ FROM CONFIG FILE
    sigmaL = 1.f;

    bsigmaL = b*b*sigmaL*sigmaL;
    bsigmaP = b*b*sigmaP*sigmaP;
}

// Public methods

bool sceneRepresentation::waitUntilClose(){
    while(win->isOpen());
    return true;
}

bool sceneRepresentation::isOpen(){
    return win->isOpen();
}

// Auxiliar methods

CPose3D sceneRepresentation::getPoseXYZ(VectorXf x){
    CPose3D pose(x(0),x(1),x(2),x(3),x(4),x(5));
    return pose;
}

CMatrixDouble sceneRepresentation::getPoseFormat(Matrix<double,4,4> T){
    CMatrixDouble T_(4,4);
    for(unsigned int i = 0; i < 4; i++){
        for(unsigned int j = 0; j < 4; j++){
            T_(i,j) = T(i,j);
        }
    }
    return T_;
}

CMatrixDouble33 sceneRepresentation::getCovFormat(MatrixXf cov_){
    CMatrixDouble33 cov3;

    Matrix3f        cov3_eigen = cov_.block(0,0,3,3);

    for(unsigned int i = 0; i < 3; i++){
        for(unsigned int j = 0; j < 3; j++){
            cov3(i,j) = cov3_eigen(i,j);
        }
    }
    return cov3;
}

void sceneRepresentation::getYPR(float &yaw, float &pitch, float &roll){
    double y, p, r;
    pose.getYawPitchRoll(y,p,r);
    yaw   = y;
    pitch = p;
    roll  = r;
}

void sceneRepresentation::getPose(Matrix<double,4,4> &T){
    CMatrixDouble44 T_;
    pose.getHomogeneousMatrix(T_);
    for(unsigned int i = 0; i < 4; i++){
        for(unsigned int j = 0; j < 4; j++){
            T(i,j) = T_(i,j);
        }
    }
}

