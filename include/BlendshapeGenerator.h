#ifndef BLENDSHAPE_GENERATOR_H
#define BLENDSHAPE_GENERATOR_H

#include "Common.h"
#include "MeshTransfer.h"

class BlendshapeGenerator{
public:
    void blendshapeGeneration();
    
    
    std::vector<double> estimateWeights(const trimesh::TriMesh& S, 
                                        const trimesh::TriMesh& B0,
                                        const std::vector<trimesh::TriMesh>& dB,
                                        const std::vector<double>& w0,
                                        const std::vector<double>& wp,
                                        bool  prior_weight_flag = true);
    std::vector<double> estimateWeightsMask(const trimesh::TriMesh& S,
                                                                 const trimesh::TriMesh& B0,
                                                                 const std::vector<trimesh::TriMesh>& dB,
                                                                 const std::vector<double>& w0_vec,   //46 init weight
                                                                 const std::vector<double>& wp_vec,   //46 prior weight
                                                                 const trimesh::TriMesh& m_mask,
                                                                 bool  prior_weight_flag);

    trimesh::TriMesh reconstructMesh(const std::vector<double>& weight, 
                                     const trimesh::TriMesh& B0,
                                     const std::vector<trimesh::TriMesh>& dB);
    
    std::vector<trimesh::TriMesh> optimizeBlendshapes(const std::vector<trimesh::TriMesh>& S,
                                                      const std::vector<std::vector<Eigen::Matrix3d> >& Sgrad,
                                                      const trimesh::TriMesh& B0,
                                                      const std::vector<std::vector<double> >& alpha,
                                                      double beta, double gamma,
                                                      const std::vector<std::vector<Eigen::Matrix3d> >& prior,
                                                      const std::vector<std::vector<double> >& w_prior);

private:
    std::vector<trimesh::TriMesh> m_S;
    
};

class LaplacianDeformer{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LaplacianDeformer(){}
    ~LaplacianDeformer(){}

    // deformable object
    void setSource(const trimesh::TriMesh& src);

    // anchor point
    void setAnchors();

    void setAnchors(const std::vector<int>& _indexs,
                    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& _coords,
                    const std::vector<double>& _weights);

    void setBarycentric(const std::vector<Eigen::Triplet<double> >& _barycentric_B,
                        const std::vector<Eigen::Vector3d>& _barycentric_C,
                        const std::vector<double>& _barycentric_M);

    // compute the laplacian matrix
    void preCompute();

    // solve Ax = b
    void solve();

    // save deformed results
    void saveResult(const string& filename);
    trimesh::TriMesh getResult();

private:
    double  findNeighborIdx(int a, int b);
    void    computeWeights();
    void    computeLaplacianBeltrami();
    void    computeRotations();
    double  computeEnergy();

private:
    trimesh::TriMesh             m_src;
    //std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > m_vertices_init;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > m_vertices_updated;
    int                          m_numVerts;   // number of vertices
    int                          m_numFaces;   // number of faces

    //used for guide
    int                          m_numAnchors;
    std::vector<int>             m_anchor_indexs;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > m_anchor_coords;
    std::vector<double>          m_anchor_weights;

    std::vector<std::vector<double> >   m_weights;
    Eigen::SparseMatrix<double>         m_laplacianBeltrami;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> >        m_rotations;

    Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix<double> > m_system_solver;
    //Eigen::SparseLU<Eigen::SparseMatrix<double> > m_system_solver;
    //Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > m_system_solver;

    // Wan-Chun Ma et.al. Semantically-aware blendshape rigs from facial performance measurements.
    int                                     m_numBarycentric;
    std::vector<Eigen::Triplet<double> >    m_barycentric_B;  //alpha, beta, gamma
    std::vector<Eigen::Vector3d>            m_barycentric_C;  //cj = uj - hi * ni
    std::vector<double>                     m_barycentric_M;  //weight
};


#endif
