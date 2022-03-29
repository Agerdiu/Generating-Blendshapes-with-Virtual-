#include "BlendshapeGenerator.h"
#include "Utils.h"
const int NUM_POSES  = 20;   // number of poses
const int NUM_SHAPES = 158;   // number of blendshapes

void LaplacianDeformer::setSource(const trimesh::TriMesh& src){
    m_src = src;

    m_src.need_neighbors();
    m_src.need_normals();
    m_src.need_adjacentfaces();

    // properity
    m_numVerts = m_src.vertices.size();
    m_numFaces = m_src.faces.size();

    //m_vertices.resize(m_numVerts);
    m_vertices_init.resize(m_numVerts);
    m_vertices_updated.resize(m_numVerts);

    for(int i=0; i<m_numVerts; i++){
        trimesh::vec3 pt = m_src.vertices[i];
        //m_vertices[i] = Eigen::Vector3d(pt[0], pt[1], pt[2]);
        m_vertices_init[i] = Eigen::Vector3d(pt[0], pt[1], pt[2]);
        m_vertices_updated[i] = Eigen::Vector3d(pt[0], pt[1], pt[2]);
    }


    m_numAnchors = 0;
    m_numBarycentric = 0;

}

void LaplacianDeformer::setAnchors()
{
    for(int i=0; i<m_src.vertices.size(); i++)
    {
        if(m_src.is_bdy(i)) {
            trimesh::vec3 pt = m_src.vertices[i];
            m_anchor_indexs.push_back(i);
            m_anchor_coords.push_back(Eigen::Vector3d(pt[0], pt[1], pt[2]));
            m_anchor_weights.push_back(1.0);
        }

    }
    m_numAnchors = m_anchor_indexs.size();
}

void LaplacianDeformer::setAnchors(const std::vector<int>& _indexs,
                                   const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& _coords,
                                   const std::vector<double>& _weights)
{
    m_anchor_indexs = _indexs;
    m_anchor_coords = _coords;
    m_anchor_weights = _weights;

    for(int i=0; i<m_src.vertices.size(); i++)
    {
        if(m_src.is_bdy(i)) {
            trimesh::vec3 pt = m_src.vertices[i];
            m_anchor_indexs.push_back(i);
            m_anchor_coords.push_back(Eigen::Vector3d(pt[0], pt[1], pt[2]));
            m_anchor_weights.push_back(1.0);
        }

    }

    m_numAnchors = m_anchor_indexs.size();

    //preCompute();
}


void LaplacianDeformer::setBarycentric(const std::vector<Eigen::Triplet<double> >& _barycentric_B,
                                       const std::vector<Eigen::Vector3d>& _barycentric_C,
                                       const std::vector<double>& _barycentric_M)
{
    //Bx = C
    m_barycentric_B = _barycentric_B;  // left
    m_barycentric_C = _barycentric_C;  // right
    m_barycentric_M = _barycentric_M;  // weigth

    m_numBarycentric = m_barycentric_C.size();
    //cout<<m_numBarycentric<<endl;
    return;
}


void LaplacianDeformer::preCompute()
{
    computeWeights();
    computeLaplacianBeltrami();
}


// given vertex a,
// find the index of its neighboring point b
double LaplacianDeformer::findNeighborIdx(int a, int b){
    for(int i = 0;i < m_src.neighbors[a].size();i++){
        if(b == m_src.neighbors[a][i])  return i;
    }
    return -1;
}


// calculate the weight of each egde for Laplacian Matrix
void LaplacianDeformer::computeWeights(){
    m_weights.clear();   // m_numVerts * m_neighVerts
    m_weights.resize(m_numVerts);

    for(int i=0; i<m_numVerts; i++){
        int num = m_src.neighbors[i].size();
        m_weights[i].resize(num);
    }

    std::vector< Eigen::Triplet<double> > Trip;
    Eigen::Vector3d a, b, c;
    Eigen::Vector3d v1, v2;
    double theta, cot;
    for(int i=0; i<m_numFaces; i++){
        int a_idx, b_idx;
        trimesh::TriMesh::Face face = m_src.faces[i];

        // coordinate of 3 vertices of a face
        a = m_vertices_init[face[0]];
        b = m_vertices_init[face[1]];
        c = m_vertices_init[face[2]];

        v1 = a - c;
        v2 = b - c;
        theta = acos( v1.dot(v2)/( v1.norm()*v2.norm() ) );
        cot = 1.0 / tan(theta); cot = fabs(cot);

        a_idx = findNeighborIdx(face[1],face[0]);
        b_idx = findNeighborIdx(face[0],face[1]);
        m_weights[face[1]][a_idx] += 0.5*cot;
        m_weights[face[0]][b_idx] += 0.5*cot;

        v1 = a - b;
        v2 = c - b;
        theta = acos( v1.dot(v2)/( v1.norm()*v2.norm() ) );
        cot = 1.0 / tan(theta);  cot = fabs(cot);

        a_idx = findNeighborIdx(face[0],face[2]);
        b_idx = findNeighborIdx(face[2],face[0]);
        m_weights[face[0]][a_idx] += 0.5*cot;
        m_weights[face[2]][b_idx] += 0.5*cot;

        v1 = b - a;
        v2 = c - a;
        theta = acos( v1.dot(v2)/( v1.norm()*v2.norm() ) );
        cot = 1.0 / tan(theta);  cot = fabs(cot);

        a_idx = findNeighborIdx(face[1],face[2]);
        b_idx = findNeighborIdx(face[2],face[1]);
        m_weights[face[1]][a_idx] += 0.5*cot;
        m_weights[face[2]][b_idx] += 0.5*cot;
    }
}


// calculate the rotation accroding edge_old and edge_new
void LaplacianDeformer::computeRotations()
{
    m_rotations.clear();
    m_rotations.resize(m_numVerts, Eigen::Matrix3d::Identity());
    for(int i=0; i<m_numVerts; i++)
    {
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

        for(int k=0; k<m_src.neighbors[i].size(); k++){
            int j = m_src.neighbors[i][k];
            // edge before and after
            Eigen::Vector3d edge = m_vertices_init[i]-m_vertices_init[j];
            Eigen::Vector3d edge_update = m_vertices_updated[i] - m_vertices_updated[j];

            double weight = m_weights[i][k];
            covariance += weight * edge * edge_update.transpose();
        }


        Eigen::Matrix3d eps;
        for(int p=0; p<3; p++)
            for(int q=0; q<3; q++)
                eps(p,q) = 1e-6;
        covariance += eps;


        Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);


        Eigen::Matrix3d u = svd.matrixU();
        Eigen::Matrix3d v = svd.matrixV();
        Eigen::Matrix3d rotation = v * u.transpose();

        if(rotation.determinant() <= 0)
        {
            Eigen::Vector3d singularValues = svd.singularValues();
            int smallestColumn = 0;
            double smallestValue = 1e10;

            for(int k=0; k<3; k++)
            {
                if(singularValues[k]<smallestValue)
                {
                    smallestValue = singularValues[k];
                    smallestColumn = k;
                }
            }

            u.col(smallestColumn) *= -1;
            rotation = v * u.transpose();
        }

        if(rotation.determinant() < 0) cout<<"Deterimant should be positive.";
        m_rotations[i] = rotation;
    }

    cout<<"computeRotations done"<<endl;
}

// Laplace–Beltrami operator
void LaplacianDeformer::computeLaplacianBeltrami(){
    m_laplacianBeltrami.resize(m_numBarycentric + m_numVerts + m_numAnchors, m_numVerts);
    m_laplacianBeltrami.setZero();

    std::vector<Eigen::Triplet<double> > Trip;
    for(int i=0; i<m_numBarycentric; i++){
        for(int k=0; k<3; k++){
            Trip.push_back(m_barycentric_B[3*i+k]);
        }
    }
    //std::vector<Eigen::Triplet<double> > Trip = m_barycentric_B;

    for(int i=0; i<m_numVerts; i++){

        double weight_sum = 0;

        for(int k=0; k<m_src.neighbors[i].size(); k++)
        {
            int j = m_src.neighbors[i][k];
            double weight = m_weights[i][k];

            Trip.push_back(Eigen::Triplet<double>(m_numBarycentric + i, j, -1 * weight));

            weight_sum += weight;
        }
        Trip.push_back(Eigen::Triplet<double>(m_numBarycentric + i, i, weight_sum));
    }

    // anchor points
    for(int i=0; i<m_numAnchors; i++){
        int idx = m_anchor_indexs[i];
        Trip.push_back(Eigen::Triplet<double>(m_numBarycentric + m_numVerts + i, idx, m_anchor_weights[i]));
    }


    m_laplacianBeltrami.setFromTriplets(Trip.begin(), Trip.end());
}




void LaplacianDeformer::solve()
{
    Eigen::MatrixXd last_solution = Eigen::MatrixXd::Zero(m_numVerts, 3);
    int iter = 0;

    Eigen::SparseMatrix<double> AtA = (m_laplacianBeltrami.transpose() * m_laplacianBeltrami).pruned();  //m_numVerts * m_numVerts
    Eigen::SparseMatrix<double> eye(m_numVerts, m_numVerts);
    const double epsilon = 1e-50;
    for(int j=0; j<m_numVerts; j++) eye.insert(j,j) = epsilon;
    AtA += eye;

    m_system_solver.compute(AtA);

    while(true)
    {
        iter ++;
        computeRotations();

        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(m_numBarycentric + m_numVerts + m_numAnchors, 3);
        for(int i=0; i < m_numBarycentric; i++){
            b.row(i) = m_barycentric_C[i];
        }


        // 1. laplacian energy
        for(int i=0; i < m_numVerts; i++)
        {
            for(size_t k=0; k<m_src.neighbors[i].size(); k++)
            {
                int j = m_src.neighbors[i][k];
                double weight = m_weights[i][k];

                Eigen::Vector3d edge = m_vertices_init[i] - m_vertices_init[j];
                b.row(m_numBarycentric + i) += weight * 0.5 * (m_rotations[i] + m_rotations[j]) * edge;
            }
        }

        // 2. anchor energy
        for(int i=0; i< m_numAnchors; i++){
            b.row(m_numBarycentric + m_numVerts + i) = m_anchor_weights[i] * m_anchor_coords[i];
        }


        // solve
        // m_laplacianBeltrami (m_numVerts + m_numAnchors, m_numVerts)
        // b (m_numVerts + m_numAnchors, 3)
        Eigen::MatrixXd Atb = m_laplacianBeltrami.transpose() * b;
        Eigen::MatrixXd solution = m_system_solver.solve(Atb);

        if(m_system_solver.info() != Eigen::Success) return;
        for(int i = 0; i < m_numVerts; i++){
            m_vertices_updated[i] = solution.row(i);
        }

        //saveResult(std::to_string(iter)+".ply");

        // check convergence
        float solution_change = (solution - last_solution).norm() / m_numVerts;
        cout << "Iter: " << iter << ",   X change: " << solution_change << endl;
        if(solution_change < 1.0 || iter >= 10) return;  //solution_change < 1

        last_solution = solution;
    }
}

// compute energy
double LaplacianDeformer::computeEnergy(){
    std::vector<double> energies(m_numVerts, 0);

    for(int i=0; i<m_numVerts; i++)
    {
        for(size_t k=0; k< m_src.neighbors[i].size(); k++)
        {
            int j = m_src.neighbors[i][k];
            Eigen::Vector3d edge_init = m_vertices_init[i] - m_vertices_init[j];
            Eigen::Vector3d edge_update = m_vertices_updated[i] - m_vertices_updated[j];

            Eigen::Vector3d vec = edge_update - m_rotations[i] * edge_init;

            double weight = m_weights[i][k];
            energies[i] += weight * vec.squaredNorm();

        }
    }

    double total_energy = 0.0;
    for(int i=0; i<m_numVerts; i++)
    {
        total_energy += energies[i];
    }
    return total_energy;
}


void LaplacianDeformer::saveResult(const string& filename)
{
    trimesh::TriMesh ret = m_src;
    for(int i=0; i<m_numVerts; i++){
        ret.vertices[i] = trimesh::vec3(m_vertices_updated[i][0], m_vertices_updated[i][1], m_vertices_updated[i][2]);
    }
    ret.write(filename);
}


trimesh::TriMesh LaplacianDeformer::getResult(){
    trimesh::TriMesh ret = m_src;
    for(int i=0; i<m_numVerts; i++){
        ret.vertices[i] = trimesh::vec3(m_vertices_updated[i][0], m_vertices_updated[i][1], m_vertices_updated[i][2]);
    }
    return ret;
}


struct PointResidual{
    PointResidual(double x, double y, double z, int idx, const std::vector<trimesh::TriMesh> &dB)
        : m_x(x), m_y(y), m_z(z), m_idx(idx), m_dB(dB){}
    
    template <typename T>
    bool operator()(const T* const alpha, T* residual) const{
        T p[3];  
        p[0] = T(0); p[1] = T(0); p[2] = T(0);
        
        for(int i=0; i<NUM_SHAPES-1; i++){
            trimesh::vec3 pt = m_dB[i].vertices[m_idx];
            p[0] += T(pt[0]) * alpha[i];
            p[1] += T(pt[1]) * alpha[i];
            p[2] += T(pt[2]) * alpha[i];
        }
        
        residual[0] = T(m_x) - p[0];
        residual[1] = T(m_y) - p[1];
        residual[2] = T(m_z) - p[2];
        return true;
    }
    
private:
    const double m_x, m_y, m_z;   // S-B0
    const int    m_idx;           // vertex index
    const std::vector<trimesh::TriMesh> &m_dB;
};


struct PriorResidual{
    PriorResidual(double* prior):m_prior(prior){}
    
    template <typename T>
    bool operator()(const T* const alpha, T* residual) const{
        for(int i=0; i<NUM_SHAPES-1; i++){
            residual[i] = T(m_prior[i]) - alpha[i];
        }
        return true;
    }
private:
    const double* m_prior;
};



std::vector<double> BlendshapeGenerator::estimateWeights(const trimesh::TriMesh& S, 
                                                         const trimesh::TriMesh& B0,
                                                         const std::vector<trimesh::TriMesh>& dB,
                                                         const std::vector<double>& w0_vec,   //46 init weight
                                                         const std::vector<double>& wp_vec,   //46 prior weight
                                                         bool  prior_weight_flag)
{    
    
    ColorStream(ColorOutput::Blue)<<"estimate weight ...";
    
    double w[NUM_SHAPES-1];
    for(int i=0; i<NUM_SHAPES-1; i++) w[i] = w0_vec[i];   
    
    double wp[NUM_SHAPES-1];
    for(int i=0; i<NUM_SHAPES-1; i++) wp[i] = wp_vec[i];
    
    ceres::Problem problem;
    
    int numVerts = S.vertices.size();
    for(int i=0; i<numVerts; i++){
        trimesh::vec3 vS  = S.vertices[i];    // S
        trimesh::vec3 vB0 = B0.vertices[i];   // B0
        double dx = vS[0] - vB0[0];           // S- B0
        double dy = vS[1] - vB0[1];
        double dz = vS[2] - vB0[2];
        
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PointResidual, 3, NUM_SHAPES-1>(new PointResidual(dx, dy, dz, i, dB));
        problem.AddResidualBlock(costFunc, NULL, w);
    }
    
    if(prior_weight_flag){
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PriorResidual, NUM_SHAPES-1, NUM_SHAPES-1>(new PriorResidual(wp));
        problem.AddResidualBlock(costFunc, NULL, w);
    }
    for(int i=0; i<NUM_SHAPES-1; i++){
        problem.SetParameterLowerBound(w, i, 0.0);
        problem.SetParameterUpperBound(w, i, 1.0);
    }
    
    // solve 
    ceres::Solver::Options options;
    options.max_num_iterations  = 10;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    options.initial_trust_region_radius = 1.0;
    options.min_lm_diagonal = 1.0;
    options.max_lm_diagonal = 1.0;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
    std::cout<<summary.BriefReport()<<std::endl;
    
    std::vector<double> ret;
    std::cout<<"    done";
    for(int i=0; i<NUM_SHAPES-1; i++){
        //std::cout<<w[i]<<" ";
        ret.push_back(w[i]);
    }
    //std::cout<<std::endl;
    return ret;
}

std::vector<trimesh::Color> partscolor = {  trimesh::Color(255,192,203), //0
                                            trimesh::Color(0,0,255),trimesh::Color(0,255,0),trimesh::Color(100,100,100),  //1-3
                                            trimesh::Color(0,255,255),trimesh::Color(255,255,0),trimesh::Color(255,0,255), //4-6
                                            trimesh::Color(40,150,100),trimesh::Color(255,100,40),trimesh::Color(160,100,255), //7-9
                                            trimesh::Color(0,100,100),trimesh::Color(254,206,180),trimesh::Color(90,200,50), //10-12
                                            trimesh::Color(200,100,100),trimesh::Color(255,255,80),trimesh::Color(140,200,170), //13-15
                                            trimesh::Color(160,82,45),trimesh::Color(0,100,160)  };

std::vector<double> BlendshapeGenerator::estimateWeightsMask(const trimesh::TriMesh& S,
                                                         const trimesh::TriMesh& B0,
                                                         const std::vector<trimesh::TriMesh>& dB,
                                                         const std::vector<double>& w0_vec,   //46 init weight
                                                         const std::vector<double>& wp_vec,   //46 prior weight
                                                         const trimesh::TriMesh& m_mask,
                                                         bool  prior_weight_flag)
{

    //colors:1,2,3,4,5,7,(8,9)face,10,14,15
    ColorStream(ColorOutput::Blue)<<"estimate weight ...";

    double w[NUM_SHAPES-1];
    for(int i=0; i<NUM_SHAPES-1; i++) w[i] = w0_vec[i];

    double wp[NUM_SHAPES-1];
    for(int i=0; i<NUM_SHAPES-1; i++) wp[i] = wp_vec[i];
    ceres::Problem problem;
    //16-

    int numVerts = S.vertices.size();
    cout << numVerts << endl;
    for(int i=0; i<numVerts; i++){
        /*
        if(m_mask.colors[i] != partscolor[1] && m_mask.colors[i] != partscolor[2] && m_mask.colors[i] != partscolor[3]
        && m_mask.colors[i] != partscolor[4] && m_mask.colors[i] != partscolor[5] && m_mask.colors[i] != partscolor[7] &&
        m_mask.colors[i] != partscolor[10] && m_mask.colors[i] != partscolor[14] && m_mask.colors[i] != partscolor[15]) continue;
        */
        if(m_mask.colors[i] != trimesh::Color(0,0,0)) continue;
        trimesh::vec3 vS  = S.vertices[i];    // S
        trimesh::vec3 vB0 = B0.vertices[i];   // B0
        double dx = vS[0] - vB0[0];           // S- B0
        double dy = vS[1] - vB0[1];
        double dz = vS[2] - vB0[2];
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PointResidual, 3, NUM_SHAPES-1>(new PointResidual(dx, dy, dz, i, dB));
        problem.AddResidualBlock(costFunc, NULL, w);
    }

    if(prior_weight_flag){
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PriorResidual, NUM_SHAPES-1, NUM_SHAPES-1>(new PriorResidual(wp));
        problem.AddResidualBlock(costFunc, NULL, w);
    }
    for(int i=0; i<NUM_SHAPES-1; i++){
        problem.SetParameterLowerBound(w, i, 0.0);
        problem.SetParameterUpperBound(w, i, 1.0);
    }
    // solve
    ceres::Solver::Options options;
    options.max_num_iterations  = 10;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    options.initial_trust_region_radius = 1.0;
    options.min_lm_diagonal = 1.0;
    options.max_lm_diagonal = 1.0;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout<<summary.BriefReport()<<std::endl;

    std::vector<double> ret;
    //std::cout<<"    done";
    for(int i=0; i<NUM_SHAPES-1; i++){
        //std::cout<<w[i]<<" ";
        ret.push_back(w[i]);
    }
    //std::cout<<std::endl;
    return ret;
}



trimesh::TriMesh BlendshapeGenerator::reconstructMesh(const std::vector<double>& weight, 
                                                      const trimesh::TriMesh& B0,
                                                      const std::vector<trimesh::TriMesh>& dB)
{
    trimesh::TriMesh ret = B0;
    for(int i=0; i<ret.vertices.size(); i++)
    {
        for(int j=0; j<NUM_SHAPES-1; j++){
            ret.vertices[i].x += weight[j] * dB[j].vertices[i].x; 
            ret.vertices[i].y += weight[j] * dB[j].vertices[i].y;
            ret.vertices[i].z += weight[j] * dB[j].vertices[i].z;
        }
    }
    return ret;
}


/*
 * @input:
 * 
 */
std::vector<trimesh::TriMesh> BlendshapeGenerator::optimizeBlendshapes(
        const std::vector<trimesh::TriMesh>& S,                   // 20
        const std::vector<std::vector<Eigen::Matrix3d> >& Sgrad,  // 20 * 22800 * mat33
        const trimesh::TriMesh& B0,                               // B0
        const std::vector<std::vector<double> >& alpha,           // 20 * 46
        double beta, double gamma,
        const std::vector<std::vector<Eigen::Matrix3d> >& prior,  // 46 * 22800 * mat33
        const std::vector<std::vector<double> >& w_prior)         // 46 * 22800
{

    cout<<"Optimize blendshapes ..."<<endl;
    
    int nfaces = B0.faces.size();
    int nverts = B0.vertices.size();
    
    cout<<"nfaces: "<<nfaces<<"     nverts: "<<nverts<<endl;
    
    int nposes  = S.size();     // 20
    int nshapes = prior.size(); // 46
    
    
    // the triangle gradient of B0 
    std::vector<Eigen::Matrix3d> B0grad(nfaces);
    //std::vector<double> B0d(nfaces);
    for(int i=0; i<nfaces; i++){
//         std::pair<Eigen::Matrix3d, double> GD = triangleGradient2(B0, i);
//         B0grad[i] = GD.first;
//         B0d[i] = GD.second;
        B0grad[i] = triangleGradient(B0, i);
    }
    
    
    using Tripletd = Eigen::Triplet<double>;
    using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    std::vector<Tripletd> Adata_coeffs;

    // upper part of A
    // sigma (alpha_ij * M^B_i) = M^S_j - M^B_0 
    // [npose * 9, nshapes * 9]
    for(int i=0; i<nposes; i++){
        int rowoffset = 9 * i;
        for(int j=0; j<nshapes; j++){   // 46
            int coloffset = 9 * j;
            for(int k=0; k<9; k++){
                Adata_coeffs.push_back(Tripletd(rowoffset+k, coloffset+k, alpha[i][j]));
            }
        }
    }
    
    
    int nrows = (nposes + nshapes) * 9;
    int ncols = nshapes * 9;
    
    std::vector<std::vector<Eigen::Matrix3d> > M(nfaces);
    //#pragma omp parallel for
    for(int j=0; j<nfaces; j++){
        if(j%1000 == 0) std::cout<<j<<std::endl<<std::flush;
        //cout<<"face "<<j<<endl;    
        
        Eigen::VectorXd b(nrows);
        
        Eigen::Matrix3d B0j = B0grad[j];
        for(int i=0; i<nposes; i++){
            Eigen::Matrix3d Sgrad_ij = Sgrad[i][j]; //20 * 22800 * mat33
            for(int k=0; k<9; k++){
                b(i * 9 + k) = Sgrad_ij(k/3, k%3) - B0j(k/3, k%3);
            }
        }
        
        // lower part of A
        std::vector<Tripletd> A_coeffs = Adata_coeffs;
        for(int i=0; i<nshapes; i++){
            int rowoffset = (nposes + i) * 9;
            int coloffset = i * 9;
            for(int k=0; k<9; k++){
                A_coeffs.push_back(Tripletd(rowoffset+k, coloffset+k, beta * w_prior[i][j]));
            }
        }
        
        // lower part of B
        for(int i=0; i<nshapes; i++){
            for(int k=0; k<9; k++){
                b( (nposes + i) * 9 + k) = beta * w_prior[i][j] * prior[i][j](k/3, k%3);
            }
        }
        
        //cout<<"Constructing linear system done..."<<endl;
        
        Eigen::SparseMatrix<double> A(nrows, ncols);
        A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
        A.makeCompressed();
        Eigen::SparseMatrix<double> AtA = (A.transpose() * A).pruned();
        
        // epsilon
        const double epsilon = 1e-6;
        Eigen::SparseMatrix<double> eye(ncols, ncols);
        for(int j=0; j<ncols; j++) eye.insert(j,j) = epsilon;
        AtA += eye;
        
        // solve Ax = b
        Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix<double> > solver;
        solver.compute(AtA);
        
        Eigen::VectorXd x = solver.solve(A.transpose() * b);  //vectorXd //46 * 9
        
        M[j] = std::vector<Eigen::Matrix3d>(nshapes);  
        for(int i=0; i<nshapes; i++){
            for(int k=0; k<9; k++){
                M[j][i](k/3, k%3) = x(9*i + k, 0); 
            }
        }
    }
    
    // reconstruct the blendshape
    ColorStream(ColorOutput::Blue)<<"reconstruct the blendshape from local gradient frame ...";
    MeshTransfer transfer;
    transfer.setSource(B0);
    transfer.setTarget(B0);
    
    std::vector<trimesh::TriMesh> B_new(nshapes);
    for(int i=0; i<nshapes; i++){        
        std::vector<Eigen::Matrix3d> Bgrad_i(nfaces);
        for(int j=0; j<nfaces; j++){
            Eigen::Matrix3d M0j = B0grad[j];
            Eigen::Matrix3d Mij = M[j][i];
            
            Bgrad_i[j] = ((Mij + M0j) * M0j.inverse()).transpose();       
        }
        B_new[i] = transfer.transfer(Bgrad_i, true);
        B_new[i].write("B/B_" + std::to_string(i) + ".obj");
    }
    return B_new;
}


void BlendshapeGenerator::blendshapeGeneration(){
    std::string A_path = "../data/Tester_1/Blendshape/";     // 47 template blenshape
    std::string T_path = "../data/Tester_1/TrainingPose/";   // 20 template poses
    std::string S_path = "../data/Tester_101/TrainingPose/"; // 20 training poses
    std::string B_path = "../data/Tester_101/Blendshape/";   // 47 blendshape to be solved (unknown)
    
    const int nshapes = NUM_SHAPES;  //47
    const int nposes  = NUM_POSES;   //20

    // given A0-A46, T0-T19, S0-S19
    // -> B0-B46
    std::vector<trimesh::TriMesh> A(nshapes); // 47 example blendshapes
    std::vector<trimesh::TriMesh> T(nposes);  // 20 example poses
    std::vector<trimesh::TriMesh> S(nposes);  // 20 training poses
    std::vector<trimesh::TriMesh> B(nshapes); // 47 unknown variables
    
    std::vector<trimesh::TriMesh> B_new(nshapes-1);
    
    // load the template blendshapes and groundtruth blendshapes
    for(int i=0; i<nshapes; i++){
        A[i]    = *trimesh::TriMesh::read(A_path + "shape_" + std::to_string(i) + ".obj");
    }
    
    for(int i=0; i<nshapes-1; i++){
        
        //B_new[i] = *trimesh::TriMesh::read(B_path + "shape_" + std::to_string(i+1) + ".obj");
    }
    
    // load training poses
    for(int i=0; i<nposes; i++){
        T[i] = *trimesh::TriMesh::read(T_path + "pose_" + std::to_string(i) + ".obj");
        S[i] = *trimesh::TriMesh::read(S_path + "pose_" + std::to_string(i) + ".obj");
    }
    
    trimesh::TriMesh A0 = A[0];
    trimesh::TriMesh B0 = S[0];
    
    // step 1.1: initialize B blendshape
    std::cout<<"1.1 Intialize Blendshape B ... "<<std::endl;
    MeshTransfer transfer;
    transfer.setSource(A0);
    transfer.setTarget(B0);
    std::vector<trimesh::TriMesh> B_init(nshapes);
    B_init[0] = B0;
    for(int i=1; i<nshapes; i++){
        std::cout << i <<" ";
        B_init[i] = transfer.transfer(A[i]);
        B_init[i].write("initBlendshape/"+std::to_string(i)+".obj");
    }
    std::cout<<"    Done"<<std::endl;
    
    
    // step 1.2: the delta shapes
    std::vector<trimesh::TriMesh> dB(nshapes-1);  // 46
    for(int i=0; i<nshapes-1; i++){
        dB[i] = B_init[i+1];
        for(int j=0; j<dB[i].vertices.size(); j++){
            dB[i].vertices[j] -= B0.vertices[j];
        }
    }
    
    
    // step 2.1: estimate the initial weights
    std::cout<<"2.1 Estimate initial weight ..."<<std::endl;
    std::vector<double> w0_vec(nshapes-1, 0.0), wp_vec(nshapes-1, 0.0);
    w0_vec[0] = 1.0;
    std::vector<std::vector<double> > alpha_refs(nposes, std::vector<double>(nshapes-1, 0.0));  // 20 * 46
    for(int i=0; i<nposes; i++)
    {
        std::cout << i <<" ";
//        alpha_refs[i] = estimateWeights(S[i], B0, dB, w0_vec, wp_vec, true);
//         
//         trimesh::TriMesh ret = reconstructMesh(alpha_refs[i], B0, dB);
//         ret.write("reconstruct/"+std::to_string(i)+".ply");
        
    }
    std::cout<<"    Done"<<std::endl;
    
    
    // step 2.2: triangle gradient
    std::cout<<"2.2 TriangleGradient for A0 and B0 ...";
    int nfaces = A0.faces.size();
    std::vector<Eigen::Matrix3d> MA0, MB0;  // 22800 * mat33 // local frame for each triangle
    for(int j=0; j<nfaces; j++){
        Eigen::Matrix3d MA0j = triangleGradient(A0, j);
        Eigen::Matrix3d MB0j = triangleGradient(B0, j);
        MA0.push_back(MA0j);
        MB0.push_back(MB0j);
    }
    std::cout<<"    Done"<<std::endl;
    
    
    // step 3.1: prior parameters
    std::cout<<"3.1 w_prior and prior ..."<<std::endl;
    double kappa = 0.1, theta = 2;
    std::vector<std::vector<Eigen::Matrix3d> > prior(nshapes-1, std::vector<Eigen::Matrix3d>(nfaces)); // 46 * 22800 * mat33
    std::vector<std::vector<double> > w_prior(nshapes-1, std::vector<double>(nfaces, 0.0));            // 46 * 22800
    for(int i=0; i<nshapes-1; i++){   
        std::cout<<i<<" ";
        trimesh::TriMesh Ai = A[i+1];   // for A1 - A46
        
        for(int j=0; j<nfaces; j++){
            Eigen::Matrix3d MA0j = MA0[j];
            Eigen::Matrix3d MAij = triangleGradient(Ai, j);
            Eigen::Matrix3d GA0Ai = MAij * MA0j.inverse();
            Eigen::Matrix3d MB0j = MB0[j];
            prior[i][j] = GA0Ai * MB0j - MB0j;   // localframe for B
            
            // GA0Ai * MA0j - MA0j = MAij * MA0j.inverse() * MA0j - MA0j =  MAij * MA0j - MA0j
            double MAij_norm = (MAij - MA0j).norm();   
            w_prior[i][j] = pow( (1 + MAij_norm) / (kappa + MAij_norm), theta);
        }
    }
    std::cout<<"    Done"<<std::endl;
    
    
    // triangleGradient for S
    std::vector<std::vector<Eigen::Matrix3d> > Sgrad(nposes, std::vector<Eigen::Matrix3d>(nfaces));   // 20 * 22800 * mat33
    for(int i=0; i<nposes; i++){
        for(int j=0; j<nfaces; j++){
            Eigen::Matrix3d Sij = triangleGradient(S[i], j); 
            Sgrad[i][j] = Sij;
        }
    }
    
    
    /***********************************************
     *************** main loop *********************
     * ********************************************/
    bool   converged = false;
    int    maxIters = 10;//10;
    double beta_max  = 0.5,  beta_min  = 0.1;
    double gamma_max = 1000, gamma_min = 100;
    int numVerts = B[0].vertices.size();    // 11510
    
    std::vector<std::vector<double> > alpha = alpha_refs;   // 20*46
    int iters = 0;
    while(iters < maxIters && !converged){
        cout<<"Iteration: "<<iters<<endl;
        converged = false;
        
        double beta  = beta_max  - 1.0*iters/maxIters * (beta_max  - beta_min);
        double gamma = gamma_max - 1.0*iters/maxIters * (gamma_max - gamma_min);
        
        // refine blendshapes
        B_new = optimizeBlendshapes(S, Sgrad, B0, alpha, beta, gamma, prior, w_prior);
        for(int i=0; i<nshapes-1; i++) B[i+1] = B_new[i];
        
        
        // update delta shapes
        for(int i=0; i<nshapes-1; i++){
            for(int k=0; k<numVerts; k++){
                dB[i].vertices[k] = B[i+1].vertices[k] - B[0].vertices[k];
            }
        }
        
        // update weights
        for(int i=0; i<nposes; i++){
            alpha[i] = estimateWeights(S[i], B0, dB, alpha[i], wp_vec, true);
        }
        
        // update reconstruction vertices
        for(int i=0; i<nposes; i++){
            trimesh::TriMesh tmp = B0;
            for(int j=0; j<nshapes-1; j++){
                for(int k=0; k<numVerts; k++){
                    tmp.vertices[k] += alpha[i][j] * dB[j].vertices[k];
                }
            }
            S[i] = tmp;
        }
    
        
        // compute deformation gradient for each triangle of S
        for(int i=0; i<nposes; i++){
            for(int j=0; j<nfaces; j++){
                Sgrad[i][j] = triangleGradient(S[i], j);
            }
        }
    
        iters++;
        if(converged || iters == maxIters) break;
        
    }
    
    
    // save output
    for(int i=0; i<nshapes; i++){
        B[i].write("B/B_" + std::to_string(i) + ".obj");
    }
    for(int i=0; i<nposes; i++){
        S[i].write("S/S_" + std::to_string(i) + ".obj");
    }
    
    return;
    
}
