#include "Common.h"
#include "MeshTransfer.h"
const int NUM_SHAPES = 47;
struct Landmark2dResidual{
    Landmark2dResidual(double x, double y, int idx, const std::vector<std::vector<trimesh::vec2>> &dB)
            : px(x), py(y), index(idx), db_landmarks(dB){}
    template <typename T>
    bool operator()(const T* const w, T* residual) const{
        T p[2];
        p[0] = T(0); p[1] = T(0);

        for(int i=0; i<NUM_SHAPES-1; i++){
            trimesh::vec2 ld = db_landmarks[i][index];
            p[0] += T(ld[0]) * w[i];
            p[1] += T(ld[1]) * w[i];
        }
        residual[0] = T(px) - p[0];
        residual[1] = T(py) - p[1];
        return true;
    }
    private:
    const double px, py;
    const int index;
    const std::vector<std::vector<trimesh::vec2>> &db_landmarks;
};

struct Landmark3dResidual{
    Landmark3dResidual(double x, double y, double z, int idx, const std::vector<std::vector<trimesh::vec3>> &dB)
            : px(x), py(y),pz(z), index(idx), db_landmarks(dB){}
    template <typename T>
    bool operator()(const T* const alpha, T* residual) const{
        T p[2];
        p[0] = T(0); p[1] = T(0);p[2] = T(0);

        for(int i=0; i<NUM_SHAPES-1; i++){
            trimesh::vec3 ld = db_landmarks[i][index];
            p[0] += T(ld[0]) * alpha[i];
            p[1] += T(ld[1]) * alpha[i];
            p[2] += T(ld[2]) * alpha[i];
        }
        residual[0] = T(px) - p[0];
        residual[1] = T(py) - p[1];
        residual[2] = T(pz) - p[2];
        return true;
    }
private:
    const double px, py, pz;
    const int index;
    const std::vector<std::vector<trimesh::vec3>> &db_landmarks;
};

struct PriorResidual{
    PriorResidual(double* prior):m_prior(prior){}

    template <typename T>
    bool operator()(const T* const w, T* residual) const{
        for(int i=0; i<NUM_SHAPES-1; i++){
            residual[i] = T(m_prior[i]) - w[i];
        }
        return true;
    }
private:
    const double* m_prior;
};


vector<double> estimateWeights_2d(                       const std::vector<trimesh::vec2>& marks,
                                                         const std::vector<int>& landmark_index,
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
    std::vector<std::vector<trimesh::vec2>> db_landmarks;
    for(int i = 0;i<dB.size();i++)
    {
        for(int j = 0;j<marks.size();j++)
        {
            trimesh::vec3 a = dB[i].vertices[landmark_index[j]];
            trimesh::vec2 b = trimesh::vec2(a.x,a.z);
            db_landmarks[i].push_back(b);
        }
    }

    for(int i=0; i<marks.size(); i++){
        trimesh::vec2 inputlandmark  = marks[i];    // S
        trimesh::vec2 B0_landmark = db_landmarks[0][i];   // B0
        double dx = inputlandmark[0] - B0_landmark[0];           // S- B0
        double dy = inputlandmark[1] - B0_landmark[1];
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<Landmark2dResidual, 2, 46>(new Landmark2dResidual(dx, dy, i, db_landmarks));
        problem.AddResidualBlock(costFunc, NULL, w);
    }

    if(prior_weight_flag){
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PriorResidual, 46, 46>(new PriorResidual(wp));
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
	options.minimizer_progress_to_stdout = false;
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    options.initial_trust_region_radius = 1.0;
    options.min_lm_diagonal = 1.0;
    options.max_lm_diagonal = 1.0;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
    //std::cout<<summary.BriefReport()<<std::endl;

    std::vector<double> ret;
    std::cout<<"    done";
    for(int i=0; i<NUM_SHAPES-1; i++){
        //std::cout<<w[i]<<" ";
        ret.push_back(w[i]);
    }
    //std::cout<<std::endl;
    return ret;
}

vector<double> estimateWeights_3d(                       const trimesh::TriMesh& S,
                                                         const std::vector<int>& landmark_index,
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
    std::vector<std::vector<trimesh::vec3>> db_landmarks(dB.size());
    std::vector<trimesh::vec3> source_marks;
    for(int i = 0;i<landmark_index.size();i++)
        source_marks.push_back(S.vertices[landmark_index[i]]);
    //cout << "1" << endl;
    for(int i = 0;i<dB.size();i++)
    {
        for(int j = 0;j<landmark_index.size();j++)
        {
            trimesh::vec3 a = dB[i].vertices[landmark_index[j]];
            //trimesh::vec3 b = trimesh::vec3(a.x,a.y,a.z);
            db_landmarks[i].push_back(a);
        }
    }
    //cout << "2" << endl;
    for(int i=0; i<landmark_index.size(); i++){
        trimesh::vec3 inputlandmark  = source_marks[i];    // S
        trimesh::vec3 B0_landmark = B0.vertices[landmark_index[i]];   // B0
        double dx = inputlandmark[0] - B0_landmark[0];           // S - B0
        double dy = inputlandmark[1] - B0_landmark[1];
        double dz = inputlandmark[2] - B0_landmark[2];
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<Landmark3dResidual, 3, 46>(new Landmark3dResidual(dx, dy, dz, i, db_landmarks));
        problem.AddResidualBlock(costFunc, NULL, w);
    }
    //cout << "3" << endl;
    if(prior_weight_flag){
        ceres::CostFunction *costFunc = new AutoDiffCostFunction<PriorResidual, 46, 46>(new PriorResidual(wp));
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
    cout << "Solving" << endl;
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

void blending_shapes(trimesh::TriMesh &blended_shape,
                     const trimesh::TriMesh& B0,
                     const std::vector<trimesh::TriMesh>& dB,
                     const std::vector<double>& weights)
{
    //trimesh::TriMesh blended_shape = B0;
    for(int i = 0;i<weights.size();i++)
    {
        for(int v = 0;v<B0.vertices.size();v++)
        {
            blended_shape.vertices[v].x += (float)weights[i] * (dB[i].vertices[v].x - B0.vertices[v].x);
            blended_shape.vertices[v].y += (float)weights[i] * (dB[i].vertices[v].y - B0.vertices[v].y);
            blended_shape.vertices[v].z += (float)weights[i] * (dB[i].vertices[v].z - B0.vertices[v].z);
        }
    }
    return;
}