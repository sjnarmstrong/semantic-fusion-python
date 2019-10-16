/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef ELASTIC_FUSION_INTERFACE_H_
#define ELASTIC_FUSION_INTERFACE_H_ 
#include <memory>
#include <iostream>

#include <ElasticFusion.h>

#include <utilities/Types.h>
#include <Eigen/Core>

class ElasticFusionInterface {
public:
  // NOTE this must be performed in the header to globally initialise these
  // variables...
  ElasticFusionInterface() 
  : initialised_(false) 
  , height_(480)
  , width_(640)
  , tracking_only_(false)
  {
      //Intrinsics are initialised later
    //Resolution::getInstance(width_, height_);
    // Primesense intrinsics
    //Intrinsics::getInstance(528, 528, width_ / 2, height_ / 2);
  }
  virtual ~ElasticFusionInterface();

  virtual bool Init(std::vector<ClassColour> class_colour_lookup,
                    int timeDelta = 200,
                    int countThresh = 35000,
                    float errThresh = 5e-05,
                    float covThresh = 1e-05,
                    bool closeLoops = true,
                    bool iclnuim = false,
                    bool reloc = false,
                    float photoThresh = 115,
                    float confidence = 10,
                    float depthCut = 8,
                    float icpThresh = 10,
                    bool fastOdom = false,
                    float fernThresh = 0.3095,
                    bool so3 = true,
                    bool frameToFrameRGB = false,
                    std::string fileName = ""
  );
  virtual bool ProcessFrame(const ImagePtr rgb,
                            const DepthPtr depth,
                            const int64_t timestamp,
                            const Eigen::Matrix4f * inPose = 0,
                            const float weightMultiplier = 1.f,
                            const bool bootstrap = false);
  virtual bool ProcessFrameNumpy(ImagePtr rgb_arr, int n_rgb,
                                 DepthPtr depth_arr, int n_depth,
                                 const long timestamp,
                                 float * pose = 0,
                                 int n_pose_x = 0,
                                 int n_pose_y = 0,
                                 const float weightMultiplier = 1.f,
                                 const bool bootstrap = false)
  {
      Eigen::Matrix4f *inPose = nullptr;
      if (n_pose_x == 4 && n_pose_y == 4) {
          inPose = new Eigen::Matrix4f(Eigen::Map<Eigen::Matrix<float,4,4, Eigen::RowMajor>>(pose));
          // std::cout << "Detected inpose "<<*inPose<<std::endl;

      } else if (n_pose_x != 0 || n_pose_y != 0) {
          std::cout << "Pose should be a 4x4 array skipping the provided pose!!"<<std::endl;
      }
      auto ret = ProcessFrame(rgb_arr, depth_arr, timestamp, inPose, weightMultiplier, bootstrap);
      delete inPose;
      return ret;
  }

  void getCurrentPose(float ** out_pose, int * d_0, int * d_1){
      auto curr_pose = elastic_fusion_->getCurrPose();
//      std::cout << "CurrPose "<<curr_pose<<std::endl;
      *out_pose = (float *)malloc(4*4*sizeof(float));
      assert (*out_pose != NULL);
      *d_0 = 4;
      *d_1 = 4;
//      std::cout << "Starting cpy "<< curr_pose.size()<<std::endl;
      memcpy(*out_pose,curr_pose.data(),4*4 * sizeof(float));
  }

  int height() const { return height_; }
  int width() const { return width_; }

  const std::vector<int>& GetSurfelIdsCpu();
  cudaTextureObject_t GetSurfelIdsGpu();
  void UpdateSurfelClass(const int surfel_id, const int class_id);
  void UpdateSurfelClassGpu(const int n, const float* surfelclasses, const float* surfelprobs, const float threshold);

  int* GetDeletedSurfelIdsGpu();

  float* GetMapSurfelsGpu() {
    if (elastic_fusion_) {
      return elastic_fusion_->getGlobalModel().getMapSurfelsGpu();
    }
    return nullptr;
  }

  int GetMapSurfelCount() {
    if (elastic_fusion_) {
      return elastic_fusion_->getGlobalModel().lastCount();
    }
    return 0;
  }

  int GetMapSurfelDeletedCount() {
    if (elastic_fusion_ && !tracking_only_) {
      return elastic_fusion_->getGlobalModel().deletedCount();
    }
    return 0;
  }

  void RenderMapToBoundGlBuffer(const pangolin::OpenGlRenderState& camera, const bool classes);
  GPUTexture* getRawImageTexture();
  GPUTexture* getRawDepthTexture();
  GPUTexture* getIdsTexture();

  void setTrackingOnly(const bool tracking) { 
    if (elastic_fusion_) {
      elastic_fusion_->setTrackingOnly(tracking);
      tracking_only_ = tracking;
    }
  }

  ElasticFusion& getElasticFusionInstance(){
      return *elastic_fusion_;
  }

private:
  bool initialised_;
  int height_;
  int width_;
  std::unique_ptr<ElasticFusion> elastic_fusion_;
  std::vector<int> surfel_ids_;
  std::vector<float> class_color_lookup_;
  float* class_color_lookup_gpu_;
  bool tracking_only_;
};

#endif /* ELASTIC_FUSION_INTERFACE_H_ */
