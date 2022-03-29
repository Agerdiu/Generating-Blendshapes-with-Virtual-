#ifndef TEST_CASES_H
#define TEST_CASES_H

#include "Common.h"
#include "TriangleGradient.h"
#include "MeshTransfer.h"
#include "BlendshapeGenerator.h"


struct CostFunctor {
	template <typename T>
	bool operator()(const T* const x, T* residual) const 
	{
		residual[0] = T(10.0) - x[0];
		return true;
	}
};


class TestCases
{
public:
	static void testCeres();
	static void testEigenMatrix();
	static void testTriangleGradient();
    static void testMeshTransfer();
    static void testBlendshapeGeneration();
};
void blending_shapes(trimesh::TriMesh &blended_shape,
                     const trimesh::TriMesh& B0,
                     const std::vector<trimesh::TriMesh>& dB,
                     const std::vector<double>& weights);
vector<double> estimateWeights_3d(                       const trimesh::TriMesh& S,
                                                         const std::vector<int>& landmark_index,
                                                         const trimesh::TriMesh& B0,
                                                         const std::vector<trimesh::TriMesh>& dB,
                                                         const std::vector<double>& w0_vec,   //46 init weight
                                                         const std::vector<double>& wp_vec,   //46 prior weight
                                                         bool  prior_weight_flag);

#endif


