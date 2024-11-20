#ifndef KE_UTILS_HPP
#define KE_UTILS_HPP

#include "traits_ke.hpp"
#include <limits>




//function to map a matrix into a vector: for 2d case
KE_Traits::StoringVector
from_matrix_to_col(const KE_Traits::StoringMatrix &mat)
{
  return KE_Traits::StoringVector(mat.reshaped());
}

//function to map a vector into a matrix: for 2d case
KE_Traits::StoringMatrix
from_col_to_matrix(const KE_Traits::StoringVector &col, int rows, int cols)
{
  return Eigen::Map<const KE_Traits::StoringMatrix>(col.data(),rows,cols);
}


//function to add nans in the result
//pred is the vector with the prediction, row ret the vector containing which points actually have evals, complete size is the size of all the points
KE_Traits::StoringVector
add_nans_vec(const KE_Traits::StoringVector &pred, const std::vector<int> &row_ret, int complete_size)
{
  if(row_ret.size()==0){return pred;}
  KE_Traits::StoringVector pred_comp(complete_size);
  
  pred_comp.setConstant(std::numeric_limits<double>::quiet_NaN());
  
  int counter=0;
  std::for_each(row_ret.cbegin(),row_ret.cend(),[&pred_comp,&pred,&counter](int el){pred_comp[el]=pred[counter]; counter++;});
  
  return pred_comp;
}

#endif  //KE_UTILS_HPP