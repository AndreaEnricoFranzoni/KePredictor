#ifndef KE_TRAITS_HPP
#define KE_TRAITS_HPP

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <variant>
#include <type_traits>
#include <cmath>


// Types for doing algebric operations
struct KE_Traits
{
public:
  
  // Data are stored in dynamic matrices (sizes are known only at compile time, and easily very larges)
  // of doubles: the only conversion from R types is Numeric -> double. some precision is lost. But double are
  // used since time series are of real numbers.
  using StoringMatrix = Eigen::MatrixXd;
  
  using StoringVector = Eigen::VectorXd;    //col-wise
  
  using StoringArray  = Eigen::ArrayXd;
  
};


//domain dimension
enum DOM_DIM
{
  uni_dim  = 0,      //1D domain
  bi_dim   = 1,      //2D domain
};


//if k is imposed 
enum K_IMP
{
  NO  = 0,     //k is not passed as parameter, has to be found using cumulative explanatory power
  YES = 1,     //k is already known 
};



//CV train/validation set strategy
enum CV_STRAT
{
  AUGMENTING_WINDOW = 0,  //using augmenting window during CV
};


//CV error evaluation
enum CV_ERR_EVAL
{
  MSE = 0,  //using mse to evaluate the prediction of the trained model in the validation
};





#endif /*KE_TRAITS_HPP*/