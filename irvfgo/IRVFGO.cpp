#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <sys/time.h>
#include <unistd.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <boost/make_shared.hpp>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <gtsam/navigation/NavState.h>

using namespace gtsam;
using namespace std;
using namespace gtsam::noiseModel;
using symbol_shorthand::X;
using symbol_shorthand::V;
using symbol_shorthand::B;
using symbol_shorthand::M;
using symbol_shorthand::Y;
using symbol_shorthand::L;
using symbol_shorthand::K;

class VisualDockFactor2: public NoiseModelFactor2<Pose3,Pose3> {
  typedef NoiseModelFactor2<Pose3, Pose3> Base;
  Cal3_S2::shared_ptr K_; 
  Point3 P_;
  Point2 measured_;
  Pose3 camera_to_body;    // c frame to b frame and d frame to b frame

  public:
    VisualDockFactor2(const Key& pose1, const Key& pose2,const Point2& pixel,const Point3& Pt_world,const Cal3_S2::shared_ptr& calib, const Pose3& cameratobody,const SharedGaussian& noiseModel) :
      Base(noiseModel,pose1, pose2),K_(calib),P_(Pt_world),measured_(pixel),camera_to_body(cameratobody){
    }

    Vector evaluateError(const Pose3& pose1, const Pose3& pose2,
      boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 =
        boost::none) const {
          Matrix66 D_local_usv;
          Matrix66 D_local_berth;
          Matrix66 D_camera_body;
          Matrix66 D_berth_body;
          
          Pose3 camera_pose = pose1.transformPoseFrom(camera_to_body,D_camera_body,boost::none);
          Pose3 UUV_Dock = pose2.between(camera_pose,D_local_berth,D_local_usv);
          PinholeCamera<Cal3_S2> camera(UUV_Dock,*K_);
          
          Matrix D_local_pose;
          Matrix D_local_point;
          auto it = camera.project(P_, D_local_pose, D_local_point, boost::none);
          
          Vector2 error =  it - measured_;
          Matrix63 A;
          A<<Z_3x3,I_3x3;
          if(H1){
            H1->resize(2,6);
            *H1 = D_local_pose * D_local_usv * D_camera_body;
          } 
          if(H2){
                H2->resize(2,6);
                *H2 = D_local_pose * D_local_berth;
          } 
          return error;
        }
};
  
class MyAHRSFactor: public NoiseModelFactor1<Pose3> {
  protected:
    Rot3 ahrs_measurement;
    typedef NoiseModelFactor1<Pose3> Base;

  public:
    MyAHRSFactor(const Key& usv,const Rot3 & r, const SharedGaussian& noiseModel) : 
        Base(noiseModel, usv) ,ahrs_measurement(r){ 
    }

    Vector evaluateError(const Pose3& usv,
      boost::optional<Matrix&> H1 = boost::none) const {
        Matrix36 D_local_Pose;
        auto attitude = usv.rotation(&D_local_Pose);
        Matrix33 D_dR_R,D_error_R;
        const Rot3 dR = attitude.between(ahrs_measurement,&D_dR_R);
        Vector3 error;
        error << Rot3::Logmap(dR,&D_error_R);
        if(H1){
          H1->resize(3,6);
          *H1 = D_error_R * D_dR_R * D_local_Pose;
        }
        return error;
      }
};

class DVLFactor: public NoiseModelFactor1<Vector3> {
  protected:
    Vector3 dvl_measurement;
    typedef NoiseModelFactor1<Vector3> Base;

  public:
    DVLFactor(const Key& usv,const Vector3& vel, const SharedGaussian& noiseModel) : 
        Base(noiseModel, usv) ,dvl_measurement(vel){ 
    }

    Vector evaluateError(const Vector3& usv,boost::optional<Matrix&> H1 = boost::none) const {
      Vector3 error;
      error = dvl_measurement - usv;
      if(H1){
        H1->resize(3,3);
        *H1 = -I_3x3;
      }
      return error;
    }
};

class MyGPSFactor: public NoiseModelFactor1<Pose3> {
  protected:
    Vector3 xyz_measurement;
    typedef NoiseModelFactor1<Pose3> Base;

  public:
    MyGPSFactor(const Key& usv,const Vector3& xyz, const SharedGaussian& noiseModel) : 
        Base(noiseModel, usv) ,xyz_measurement(xyz){ 
    }

    Vector evaluateError(const Pose3& usv,boost::optional<Matrix&> H1 = boost::none) const {
      Matrix36 D_local_pose;
      Point3 predict = usv.translation(&D_local_pose);
      Matrix33 D_local_vector3;
      D_local_vector3<<1,0,0,
            0,1,0,
            0,0,1;
      Vector3 error;
      error = predict - xyz_measurement;
      if(H1){
        H1->resize(3,6);
        *H1 = D_local_vector3 * D_local_pose;
      }
      return error;
    }
};

int main(int argc, char* argv[]) {
  Cal3_S2::shared_ptr calib(new Cal3_S2(613, 613, 0, 661, 386));

  string data_filename;
  string output_filename;
  data_filename = argv[1];
  output_filename = argv[2]; 
  ifstream file(data_filename.c_str());
  FILE* fp_out = fopen(output_filename.c_str(), "w+");
  
  Rot3 prior_usv_rotation = Rot3::Ypr(-0.025133,0,0);
  Point3 prior_usv_point(0.003619,0.447945,0);
  Pose3 prior_usv_pose(prior_usv_rotation, prior_usv_point);
  Vector3 prior_usv_velocity(0,0,0);
  NavState prior_usv_state(prior_usv_pose,prior_usv_velocity);
  Rot3 prior_berth_rotation = Rot3::Ypr(0,0,0);
  Point3 prior_berth_point(8,27,0);
  Pose3 prior_berth_pose(prior_berth_rotation, prior_berth_point);
  Vector3 prior_vel_body(0,0,0);
  imuBias::ConstantBias prior_imu_bias(Vector3(0.00017453,0.00017453,0.00017453),Vector3(9.8e-6,9.8e-6,9.8e-6));
  
  int timestamp = 0;
  int remain = 0;
  bool dvl_flag = false;
  bool pixel_flag = false;
  
  Point3 last_berth_position;
  last_berth_position = prior_berth_point;
  Point3 prop_berth_position;
  NavState berth_measurement;
  double opt_timeused;
 
  noiseModel::Diagonal::shared_ptr usv_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 1.5, 1.5,1.5).finished());
  noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01,0.01,0.01,0.01, 0.01, 0.01).finished());
  noiseModel::Diagonal::shared_ptr berth_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.1,0.1,0.1,1.5,1.5,1.5).finished());
  noiseModel::Diagonal::shared_ptr vel_noise_model = noiseModel::Diagonal::Sigmas((Vector(3) << 0.1,0.1,0.1).finished());
  SharedDiagonal measurementNoise = Diagonal::Sigmas(Vector2(2, 2));
  double gyro_noise_sigmas = 9.19869283870009e-09;
  Matrix33 measured_acc_cov = (Matrix33()<<3.09281088583449e-09,0,0,
			                   0, 1.54640544291725e-09,0,
			                   0,0,1.03093696194483e-09).finished();
  Matrix33 measured_omega_cov = Matrix33::Identity(3,3) * gyro_noise_sigmas;

  NonlinearFactorGraph *graph = new NonlinearFactorGraph();
  graph->add(PriorFactor<Pose3>(X(timestamp), prior_usv_pose, usv_noise_model));
  graph->add(PriorFactor<Vector3>(V(timestamp), prior_usv_velocity, vel_noise_model));
  graph->add(PriorFactor<imuBias::ConstantBias>(B(timestamp), prior_imu_bias,bias_noise_model));
  graph->add(PriorFactor<Pose3>(Y(timestamp), prior_berth_pose, berth_noise_model));

  Values initial_values;
  initial_values.insert(X(timestamp), prior_usv_pose); 
  initial_values.insert(V(timestamp), prior_usv_velocity); 
  initial_values.insert(B(timestamp), prior_imu_bias);
  initial_values.insert(Y(timestamp),prior_berth_pose);
 
  auto p = PreintegrationParams::MakeSharedU(0);
  p->setAccelerometerCovariance(measured_acc_cov);
  p->setGyroscopeCovariance(measured_omega_cov);
  p->setIntegrationCovariance(I_3x3 * 0.1);
  p->setUse2ndOrderCoriolis(false); 
  PreintegratedImuMeasurements preint_imu(p);

  NavState usv_prop_state(prior_usv_pose,prior_usv_velocity);
  NavState usv_prev_state = usv_prop_state;
 
  imuBias::ConstantBias prev_bias = prior_imu_bias;
  auto prop_bias = prev_bias;
  auto prev_berth_pose = prior_berth_pose;

  double dt = 0.005; 

  Eigen::Matrix<double,3,1> dvl_measure = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> myahrs_measure = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> mygps_measure = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,8,1> pixel_orign = Eigen::Matrix<double,8,1>::Zero();
  Eigen::Matrix<double,12,1> pnp_orign = Eigen::Matrix<double,12,1>::Zero();

  double depth_measure;
  double vel_body_measure;
  vector<Point2> pixel_measurement(4);
  
  Rot3 pre_berth_rot = prior_berth_rotation;
  Point3 pre_berth_pos = prior_berth_point;

  string value;

  Rot3 cameratobodyrotation(1,0,0,0,0,1,0,-1,0);
  Pose3 cameratobody(cameratobodyrotation,Point3(0,0,0));
  
  const int Bag = 30;
  int cur_remove_idx = 4;
   
  while (file.good()) {
    getline(file, value, ',');
    string type = value;
    if (type == "imu") {  
      Eigen::Matrix<double,6,1> imu = Eigen::Matrix<double,6,1>::Zero();
      for (int i=0; i < 5; ++i) {
          getline(file,value,',');
          imu(i) = atof(value.c_str());
      }
      getline(file, value, '\n');
      imu(5) = atof(value.c_str());
      Vector3 gyo_imu(imu(0),imu(1),imu(2));
      Vector3 acc_imu_body(imu(3),imu(4),imu(5));
      preint_imu.integrateMeasurement(acc_imu_body, gyo_imu, dt);
    } 
    else if(type == "vel"){
      for(int i = 0;i < 2;++i){
          getline(file,value,',');
          dvl_measure(i) = atof(value.c_str());
      }
      getline(file,value,'\n');
      dvl_measure(2) = atof(value.c_str());
    }
    else if(type == "ahrs"){
      for(int i = 0;i < 2;++i){
          getline(file,value,',');
          myahrs_measure(i) = atof(value.c_str());
      }
      getline(file,value,'\n');
      myahrs_measure(2) = atof(value.c_str());
    }
    else if(type == "depth"){
      getline(file,value,'\n');
      depth_measure = -atof(value.c_str());
    }
    else if(type == "vel_body"){
      getline(file,value,'\n');
      vel_body_measure = -atof(value.c_str());
    }
    else if(type == "pixel"){
      for(int i = 0;i < 7; ++i){
          getline(file,value,',');
          pixel_orign(i) = atof(value.c_str());
      }
      getline(file,value,'\n');
      pixel_orign(7) = atof(value.c_str());

      int idx = 0;
      for(int i = 0;i < 4;++i){
          pixel_measurement[i].x() = pixel_orign(idx++);
          pixel_measurement[i].y() = pixel_orign(idx++);
      }
    }
    else if(type == "pnp"){
      for(int i = 0;i < 11; ++i){
          getline(file,value,',');
          pnp_orign(i) = atof(value.c_str());
      }
      getline(file,value,'\n');
      pnp_orign(11) = atof(value.c_str());
    }
    else if(type == "gps"){
      for(int i = 0;i < 2;++i){
          getline(file,value,',');
          mygps_measure(i) = atof(value.c_str());
      }
      getline(file,value,'\n');
      mygps_measure(2) = atof(value.c_str());
  
      timestamp++;
      ImuFactor imu_factor(X(timestamp-1),V(timestamp-1),X(timestamp),V(timestamp),B(timestamp-1),preint_imu);
      graph->add(imu_factor);
      imuBias::ConstantBias zeros_bias(Vector3(0,0,0),Vector3(0,0,0));
      graph->add(BetweenFactor<imuBias::ConstantBias>(B(timestamp - 1),B(timestamp),zeros_bias,bias_noise_model));
      
      graph->emplace_shared<VisualDockFactor2>(X(timestamp),Y(0), pixel_measurement[0], Point3(0.3, 0, 0.225),calib,cameratobody,measurementNoise);
      graph->emplace_shared<VisualDockFactor2>(X(timestamp),Y(0), pixel_measurement[1],  Point3(-0.3, 0, 0.225),calib,cameratobody,measurementNoise);
      graph->emplace_shared<VisualDockFactor2>(X(timestamp),Y(0), pixel_measurement[2],  Point3(0.3, 0, -0.225),calib,cameratobody,measurementNoise);
      graph->emplace_shared<VisualDockFactor2>(X(timestamp),Y(0), pixel_measurement[3],  Point3(-0.3, 0, -0.225),calib,cameratobody,measurementNoise);

      noiseModel::Diagonal::shared_ptr ahrs_noise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.02,0.02,0.02).finished());
      MyAHRSFactor ahrsfactor(X(timestamp),Rot3::Ypr(myahrs_measure(2),myahrs_measure(1),myahrs_measure(0)),ahrs_noise);
      graph->add(ahrsfactor);
      noiseModel::Diagonal::shared_ptr dvl_noise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.03,0.03,0.03).finished());
      DVLFactor dvlfactor(V(timestamp),dvl_measure,dvl_noise);
      graph->add(dvlfactor);
      noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.05,0.05,0.05).finished());
      MyGPSFactor mygpsfactor(X(timestamp),mygps_measure,gps_noise);
      graph->add(mygpsfactor);
      
      usv_prop_state = preint_imu.predict(usv_prev_state, prev_bias);
      Point3 new_berth_point = prev_berth_pose.translation();
      Pose3 predict_berth_value = Pose3(Rot3::Ypr(prev_berth_pose.rotation().yaw(),0,0),new_berth_point);
      
      initial_values.insert(X(timestamp), usv_prop_state.pose());
      initial_values.insert(V(timestamp),usv_prop_state.velocity());
      initial_values.insert(B(timestamp),prev_bias);

      chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
      const LevenbergMarquardtParams& params = LevenbergMarquardtParams();
      auto p = const_cast<LevenbergMarquardtParams&>(params);
      const string verbose = "summary";
      p.setVerbosityLM(verbose);
      auto pa = const_cast<const LevenbergMarquardtParams&>(p);
      LevenbergMarquardtOptimizer OPT(*graph, initial_values);
      try{
        Values result = OPT.optimize();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        opt_timeused = time_used.count();
    
        usv_prev_state = NavState(result.at<Pose3>(X(timestamp)),result.at<Vector3>(V(timestamp)));
        prev_bias = result.at<imuBias::ConstantBias>(B(timestamp));
        prev_berth_pose = result.at<Pose3>(Y(0));
        prop_berth_position = prev_berth_pose.translation();
        preint_imu.resetIntegrationAndSetBias(prev_bias);
        auto pose_usv_berthing_afterOpti = result.at<Pose3>(X(timestamp)).between(result.at<Pose3>(Y(0)));
        
        cout<<timestamp<<"------------usv attitue------------"<<endl;
        cout<<usv_prev_state.attitude().yaw()*180/M_PI<<endl;
        cout<<usv_prev_state.attitude().pitch()*180/M_PI<<endl;
        cout<<usv_prev_state.attitude().roll()*180/M_PI<<endl;
        cout<<usv_prev_state.position().x()<<endl;
        cout<<usv_prev_state.position().y()<<endl;
        cout<<usv_prev_state.position().z()<<endl;
    
        
        cout<<"------------berth attitue------------"<<endl;
        cout<<prev_berth_pose.rotation().yaw()*180/M_PI<<endl;
        cout<<prev_berth_pose.rotation().pitch()*180/M_PI<<endl;
        cout<<prev_berth_pose.rotation().roll()*180/M_PI<<endl;
        cout<<prev_berth_pose.translation().x()<<endl;
        cout<<prev_berth_pose.translation().y()<<endl;
        cout<<prev_berth_pose.translation().z()<<endl;

        fprintf(fp_out, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",timestamp, usv_prev_state.attitude().yaw(),usv_prev_state.attitude().pitch(),usv_prev_state.attitude().roll(),
        usv_prev_state.position().x(),usv_prev_state.position().y(),usv_prev_state.position().z(),
        usv_prev_state.velocity()(0),usv_prev_state.velocity()(1),usv_prev_state.velocity()(2),
        prev_berth_pose.rotation().yaw(),prev_berth_pose.rotation().pitch(),prev_berth_pose.rotation().roll(),
        prev_berth_pose.translation().x(),prev_berth_pose.translation().y(),prev_berth_pose.translation().z(),opt_timeused);
      }
      catch (exception& e){
        cout<<e.what()<<endl;
        break;
      } 

      if(timestamp >= Bag){
        initial_values.erase(X(timestamp-Bag));
        initial_values.erase(V(timestamp-Bag));
        initial_values.erase(B(timestamp-Bag));
        
        if(timestamp == Bag){
            graph->remove(0);
            graph->remove(1);
            graph->remove(2);
        }
        
        for(int i = 0;i < 9;++i){
        graph->remove(cur_remove_idx);
        ++cur_remove_idx;
        }
      }   
    }
    else{
      cout<<"some error or finished: \n";
    }
  }
}
