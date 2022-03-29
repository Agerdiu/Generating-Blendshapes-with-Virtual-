#include "TestCases.h"
#include <dirent.h>
#include <omp.h>
#include <cstdlib>

void TestCases::testCeres()
{
	double init_x = 5.0;
	double x = init_x;
	
	//build the problem
	ceres::Problem problem;
	
	// set up the cost function 
	// use the autodifferentation to obtain the derivate
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
	problem.AddResidualBlock(cost_function, NULL, &x);
	
	// run the solver
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	
	std::cout<<summary.BriefReport()<<std::endl;
	std::cout<<"init x:"<<init_x<<"   ->  "<<x<<std::endl;
	
	return;
}


void TestCases::testEigenMatrix()
{
	/*
	2 -1 0 0 0
	-1 2 -1 0 0
	0 -1 2 -1 0
	0 0 -1 2 -1
	0 0 0 -1 2
	*/
	Eigen::SparseMatrix<double> A(5, 5);
	
	std::vector<Eigen::Triplet<double> > _A;
	_A.push_back(Eigen::Triplet<double>(0, 0, 2));
	_A.push_back(Eigen::Triplet<double>(0, 1,-1));
	_A.push_back(Eigen::Triplet<double>(1, 0,-1));
	_A.push_back(Eigen::Triplet<double>(1, 1, 2));
	_A.push_back(Eigen::Triplet<double>(1, 2,-1));
	_A.push_back(Eigen::Triplet<double>(2, 1,-1));
	_A.push_back(Eigen::Triplet<double>(2, 2, 2));
	_A.push_back(Eigen::Triplet<double>(2, 3,-1));
	_A.push_back(Eigen::Triplet<double>(3, 2,-1));
	_A.push_back(Eigen::Triplet<double>(3, 3, 2));
	_A.push_back(Eigen::Triplet<double>(3, 4,-1));
	_A.push_back(Eigen::Triplet<double>(4, 3,-1));
	_A.push_back(Eigen::Triplet<double>(4, 4, 2));
	_A.push_back(Eigen::Triplet<double>(2, 1, 2));
	
	A.setFromTriplets(_A.begin(), _A.end());

	
	auto AtA = A.transpose() * A;
	
	Eigen::Matrix<double,5,1> b;
	for(int i=0; i<5; i++) b(i,0)=1;
	
	Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix<double> > solver;
	solver.compute(AtA);
	Eigen::VectorXd x = solver.solve(A.transpose()*b);
	
	for(int i=0; i<5; i++) std::cout<<x(i,0)<<" ";
	std::cout<<std::endl;
	
	Eigen::VectorXd b1 = A * x;
	for(int i=0; i<5; i++) std::cout<<b1(i,0)<<" ";
	std::cout<<std::endl;
}

void TestCases::testTriangleGradient()
{
	trimesh::TriMesh* mesh = trimesh::TriMesh::read("../emily-23686.obj");
	Eigen::Matrix3d tmp = triangleGradient(*mesh, 0);
    cout<<tmp * tmp.inverse()<<endl;
	
	pair<Eigen::Matrix3d, double> ret = triangleGradient2(*mesh, 0);
	cout<<ret.first<<endl;
	cout<<ret.second<<endl;
	
	return;
}


void TestCases::testMeshTransfer()
{
    MeshTransfer transfer;
    trimesh::TriMesh S0 = *trimesh::TriMesh::read("../data/Tester_1/Blendshape/shape_0.obj");
    trimesh::TriMesh S1 = *trimesh::TriMesh::read("../data/Tester_1/Blendshape/shape_1.obj");
    trimesh::TriMesh T0 = *trimesh::TriMesh::read("../data/Tester_101/Blendshape/shape_0.obj");
    transfer.setSource(S0);
    transfer.setTarget(S0);
    transfer.transfer(S1);
}

void laplacianDeformation(
        const trimesh::TriMesh& src,
        trimesh::TriMesh& dst,
        const std::vector<int>& landmarks_3d_idx,
        const std::vector<Eigen::Vector3d>& landmarks_3d,
        const std::string saveName = "deformed.ply")
{
    LaplacianDeformer laplacianDeformer;
    laplacianDeformer.setSource(src);

    // test cases
    std::vector<int>             anchor_indexs;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > anchor_coords;
    std::vector<double>          anchor_weights;

    for(int i=0; i<landmarks_3d_idx.size(); i++){

        anchor_indexs.push_back(landmarks_3d_idx[i]);
        anchor_coords.push_back(landmarks_3d[i]);
        anchor_weights.push_back(0.8);
    }

    laplacianDeformer.setAnchors(anchor_indexs, anchor_coords, anchor_weights); // user-specific
    laplacianDeformer.preCompute();
    laplacianDeformer.solve();
    trimesh::TriMesh ret = laplacianDeformer.getResult();
    ret.write(saveName);



    dst = src;

    cout << ret.vertices.size() << endl;
    cout << dst.vertices.size() << endl;

    for(int i = 0;i<dst.vertices.size();i++)
    {
        dst.vertices[i] = ret.vertices[i];
    }
    ret.clear();

    cout << "done!" << endl;

    return;
}


void TestCases::testBlendshapeGeneration()
{
    BlendshapeGenerator bg;
    bg.blendshapeGeneration();
}
void Aligning_non_toplogic(trimesh::TriMesh &aligning_mesh,trimesh::TriMesh &mesh_aligned,string result_name)
{

    trimesh::TriMesh template_mesh = mesh_aligned;
    //trimesh::TriMesh tracking = *trimesh::TriMesh::read("../data/bahe/tracking_23686.obj");
    //orc : 11237,11191,3981
    //1 15603 17 8538 28 5746
    //tomcat: 806 245 862
    //warpping:
    //left eye 11337 right eye 5936 center 2454
    //fat: 3538,18428,3726
    //female: 9696,4886,845
    vector <int> aligning_indexs = {1137,364,653};
    vector <int> template_indexs = {1137,364,653};
    //vector <int> template_indexs = {15603,8538,5746};
    trimesh::vec3 v1 = aligning_mesh.vertices[aligning_indexs[0]] - aligning_mesh.vertices[aligning_indexs[1]];
    trimesh::vec3 v2 = template_mesh.vertices[template_indexs[0]] - template_mesh.vertices[template_indexs[1]];
    trimesh::vec3 aligning = template_mesh.vertices[template_indexs[2]];

    double d = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) / sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] *= d;
    }

    trimesh::vec3 v3 = template_mesh.vertices[template_indexs[0]] - aligning_mesh.vertices[aligning_indexs[0]];

    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] += v3;
    }
    //origin : template_index[0]

    trimesh::vec3 tempmove1 = template_mesh.vertices[template_indexs[0]];

    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] -= tempmove1;
    }

    for(int i = 0;i<template_mesh.vertices.size();i++)
    {
        template_mesh.vertices[i] -= tempmove1;
    }
    cout << v1 << endl;
    cout << v2 << endl;
    cout << v3 << endl;

    //aligning_mesh.write("../newblendshapes/" + meshname + "mesh_step1.obj");
    //template_mesh.write("../newblendshapes/" + meshname + "temp_step1.obj");



    v1 = aligning_mesh.vertices[aligning_indexs[1]];
    v2 = template_mesh.vertices[template_indexs[1]];

    double n1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);

    v1 = v1/sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    v2 = v2/sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

    double theta = acos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))/2;

    trimesh::vec3 crossv;

    crossv[0] = v1[1] * v2[2] - v1[2] * v2[1];
    crossv[1] = v1[2] * v2[0] - v1[0] * v2[2];
    crossv[2] = v1[0] * v2[1] - v1[1] * v2[0];
    crossv /= sqrt(crossv[0] * crossv[0] + crossv[1] * crossv[1] + crossv[2] * crossv[2]);


    double quo[4];

    quo[0] = cos(theta);
    quo[1] = sin(theta) * crossv[0];
    quo[2] = sin(theta) * crossv[1];
    quo[3] = sin(theta) * crossv[2];

    Eigen::Matrix<double,3,3> R1;

    R1(0,0) = 2*quo[0]*quo[0] + 2*quo[1]*quo[1] - 1;
    R1(0,1) = 2*(quo[1] * quo[2] + quo[0] * quo[3]);
    R1(0,2) = 2*(quo[1] * quo[3] - quo[0] * quo[2]);

    R1(1,0) = 2*(quo[1] * quo[2] - quo[0] * quo[3]);
    R1(1,1) = 2*quo[0]*quo[0] + 2*quo[2]*quo[2] - 1;
    R1(1,2) = 2*(quo[2] * quo[3] + quo[0] * quo[1]);

    R1(2,0) = 2*(quo[1] * quo[3] + quo[0] * quo[2]);
    R1(2,1) = 2*(quo[2] * quo[3] - quo[0] * quo[1]);
    R1(2,2) = 2*quo[0]*quo[0] + 2*quo[3]*quo[3] - 1;


    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        Eigen::Matrix<double,1,3> t(aligning_mesh.vertices[i][0],aligning_mesh.vertices[i][1],aligning_mesh.vertices[i][2]);

        t = t * R1;
        if(isnan(t[0]))
        {
            //cout << "Is nan !" << endl;
            aligning_mesh.vertices[i][0] = aligning_mesh.vertices[i][0];
            aligning_mesh.vertices[i][1] = aligning_mesh.vertices[i][1];
            aligning_mesh.vertices[i][2] = aligning_mesh.vertices[i][2];
        }
        else {
            aligning_mesh.vertices[i][0] = t(0, 0);
            aligning_mesh.vertices[i][1] = t(0, 1);
            aligning_mesh.vertices[i][2] = t(0, 2);
        }
    }

    //aligning_mesh.write("../newblendshapes/" + meshname + "mesh_step2.obj");
    //template_mesh.write("../newblendshapes/" + meshname + "temp_step2.obj");

    trimesh::vec3 tempmove2 = (aligning_mesh.vertices[aligning_indexs[1]]+aligning_mesh.vertices[aligning_indexs[0]]) / 2;

    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] -= tempmove2;
    }
    for(int i = 0;i<template_mesh.vertices.size();i++)
    {
        template_mesh.vertices[i] -= tempmove2;
    }

    //origin: (0+1)/2
    //aligning_mesh.write("../newblendshapes/" + meshname + "mesh_step3.obj");
    //template_mesh.write("../newblendshapes/" + meshname + "temp_step3.obj");

    v1 = aligning_mesh.vertices[aligning_indexs[2]] - (aligning_mesh.vertices[aligning_indexs[1]]+aligning_mesh.vertices[aligning_indexs[0]]) / 2;
    v2 = template_mesh.vertices[template_indexs[2]] - (template_mesh.vertices[template_indexs[1]]+template_mesh.vertices[template_indexs[0]]) / 2;

    v1 = v1/sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    v2 = v2/sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

    theta = acos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))/2;

    crossv[0] = v1[1] * v2[2] - v1[2] * v2[1];
    crossv[1] = v1[2] * v2[0] - v1[0] * v2[2];
    crossv[2] = v1[0] * v2[1] - v1[1] * v2[0];
    crossv /= sqrt(crossv[0] * crossv[0] + crossv[1] * crossv[1] + crossv[2] * crossv[2]);

    quo[0] = cos(theta);
    quo[1] = sin(theta) * crossv[0];
    quo[2] = sin(theta) * crossv[1];
    quo[3] = sin(theta) * crossv[2];

    Eigen::Matrix<double,3,3> R2;

    R2(0,0) = 2*quo[0]*quo[0] + 2*quo[1]*quo[1] - 1;
    R2(0,1) = 2*(quo[1] * quo[2] + quo[0] * quo[3]);
    R2(0,2) = 2*(quo[1] * quo[3] - quo[0] * quo[2]);

    R2(1,0) = 2*(quo[1] * quo[2] - quo[0] * quo[3]);
    R2(1,1) = 2*quo[0]*quo[0] + 2*quo[2]*quo[2] - 1;
    R2(1,2) = 2*(quo[2] * quo[3] + quo[0] * quo[1]);

    R2(2,0) = 2*(quo[1] * quo[3] + quo[0] * quo[2]);
    R2(2,1) = 2*(quo[2] * quo[3] - quo[0] * quo[1]);
    R2(2,2) = 2*quo[0]*quo[0] + 2*quo[3]*quo[3] - 1;

    for(int i = 0;i<aligning_mesh.vertices.size();i++) {
        Eigen::Matrix<double, 1, 3> t(aligning_mesh.vertices[i][0], aligning_mesh.vertices[i][1], aligning_mesh.vertices[i][2]);
        t = t * R2;
        if(isnan(t[0]))
        {
            aligning_mesh.vertices[i][0] = aligning_mesh.vertices[i][0];
            aligning_mesh.vertices[i][1] = aligning_mesh.vertices[i][1];
            aligning_mesh.vertices[i][2] = aligning_mesh.vertices[i][2];
        }
        else {
            aligning_mesh.vertices[i][0] = t(0, 0);
            aligning_mesh.vertices[i][1] = t(0, 1);
            aligning_mesh.vertices[i][2] = t(0, 2);
        }
    }

    //aligning_mesh.write("../newblendshapes/" + meshname + "mesh_step4.obj");
    //template_mesh.write("../newblendshapes/" + meshname + "temp_step4.obj");

    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] += tempmove1;
        aligning_mesh.vertices[i] += tempmove2;
    }
    aligning_mesh.write("../result/Non_toplogic_aligned.obj");
    aligning_mesh.write("../result/" + result_name);

    aligning = aligning - aligning_mesh.vertices[aligning_indexs[2]];

    for(int i = 0;i<aligning_mesh.vertices.size();i++)
    {
        aligning_mesh.vertices[i] += aligning;
    }

    /*
      const int nshapes = 47;  //47

    std::vector<trimesh::TriMesh> A(nshapes);

	for(int i=0; i<nshapes; i++){
        A[i] = *trimesh::TriMesh::read("../data/bahe/out_" + std::to_string(i) + ".obj");
    }

	for(int i=0; i<A.size();i++)
	{
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] *= d;
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] += v3;
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] -= tempmove1;
		for(int j=0;j<A[i].vertices.size();j++)
		{
			if(j == 15603) continue;
			Eigen::Matrix<double,1,3> t(A[i].vertices[j][0],A[i].vertices[j][1],A[i].vertices[j][2]);
			t = t * R1;
			A[i].vertices[j][0] = t(0,0);
			A[i].vertices[j][1] = t(0,1);
			A[i].vertices[j][2] = t(0,2);
		}
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] -= tempmove2;
		for(int j=0;j<A[i].vertices.size();j++)
		{
			Eigen::Matrix<double,1,3> t(A[i].vertices[j][0],A[i].vertices[j][1],A[i].vertices[j][2]);
			t = t * R2;
			A[i].vertices[j][0] = t(0,0);
			A[i].vertices[j][1] = t(0,1);
			A[i].vertices[j][2] = t(0,2);
		}
		for(int j=0;j<A[i].vertices.size();j++)
		{
			A[i].vertices[j] += tempmove1;
			A[i].vertices[j] += tempmove2;
		}
		A[i].write("../data/bahe/blendshape_aligned_" + std::to_string(i) + ".obj");
	}
     */

}
void AligningBlendshapes(trimesh::TriMesh &B0,trimesh::TriMesh tracking)
{
	//trimesh::TriMesh B0 = *trimesh::TriMesh::read("../data/bahe/out_0.obj");
    //trimesh::TriMesh tracking = *trimesh::TriMesh::read("../data/bahe/tracking_23686.obj");
	//1 15603 17 8538 28 5746
	trimesh::vec3 v1 = B0.vertices[15603] - B0.vertices[8538];
	trimesh::vec3 v2 = tracking.vertices[15603] - tracking.vertices[8538];
    trimesh::vec3 aligning = tracking.vertices[5746];

	double d = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) / sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	for(int i = 0;i<B0.vertices.size();i++)
	{
		B0.vertices[i] *= d;
	}

	trimesh::vec3 v3 = tracking.vertices[15603] - B0.vertices[15603];

	for(int i = 0;i<B0.vertices.size();i++)
	{
		B0.vertices[i] += v3;
	}

	trimesh::vec3 tempmove1 = B0.vertices[15603];
	
	for(int i = 0;i<B0.vertices.size();i++)
	{
		B0.vertices[i] -= tempmove1;
		tracking.vertices[i] -= tempmove1;
	}

	v1 = B0.vertices[8538];
	v2 = tracking.vertices[8538];
	
	double n1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	
	v1 = v1/sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	v2 = v2/sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
	
	double theta = acos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))/2;

	trimesh::vec3 crossv;
	
	crossv[0] = v1[1] * v2[2] - v1[2] * v2[1];
	crossv[1] = v1[2] * v2[0] - v1[0] * v2[2];
	crossv[2] = v1[0] * v2[1] - v1[1] * v2[0];
	crossv /= sqrt(crossv[0] * crossv[0] + crossv[1] * crossv[1] + crossv[2] * crossv[2]);

	
	double quo[4];

	quo[0] = cos(theta);
	quo[1] = sin(theta) * crossv[0];
	quo[2] = sin(theta) * crossv[1];
	quo[3] = sin(theta) * crossv[2];

	Eigen::Matrix<double,3,3> R1;
	
	R1(0,0) = 2*quo[0]*quo[0] + 2*quo[1]*quo[1] - 1;
	R1(0,1) = 2*(quo[1] * quo[2] + quo[0] * quo[3]);
	R1(0,2) = 2*(quo[1] * quo[3] - quo[0] * quo[2]);
	
	R1(1,0) = 2*(quo[1] * quo[2] - quo[0] * quo[3]);
	R1(1,1) = 2*quo[0]*quo[0] + 2*quo[2]*quo[2] - 1;
	R1(1,2) = 2*(quo[2] * quo[3] + quo[0] * quo[1]);

	R1(2,0) = 2*(quo[1] * quo[3] + quo[0] * quo[2]);
	R1(2,1) = 2*(quo[2] * quo[3] - quo[0] * quo[1]);
	R1(2,2) = 2*quo[0]*quo[0] + 2*quo[3]*quo[3] - 1;

	
	for(int i = 0;i<B0.vertices.size();i++)
	{
		if(i == 15603) continue;
		Eigen::Matrix<double,1,3> t(B0.vertices[i][0],B0.vertices[i][1],B0.vertices[i][2]);
		t = t * R1;
		B0.vertices[i][0] = t(0,0);
		B0.vertices[i][1] = t(0,1);
		B0.vertices[i][2] = t(0,2);
	}
	
	trimesh::vec3 tempmove2 = B0.vertices[8538] / 2;

	for(int i = 0;i<B0.vertices.size();i++)
	{
		B0.vertices[i] -= tempmove2;
		tracking.vertices[i] -= tempmove2;
	}
	

	v1 = B0.vertices[5746];
	v2 = tracking.vertices[5746];
	
	v1 = v1/sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	v2 = v2/sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
	
	theta = acos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))/2;
	
	crossv[0] = v1[1] * v2[2] - v1[2] * v2[1];
	crossv[1] = v1[2] * v2[0] - v1[0] * v2[2];
	crossv[2] = v1[0] * v2[1] - v1[1] * v2[0];
	crossv /= sqrt(crossv[0] * crossv[0] + crossv[1] * crossv[1] + crossv[2] * crossv[2]);

	quo[0] = cos(theta);
	quo[1] = sin(theta) * crossv[0];
	quo[2] = sin(theta) * crossv[1];
	quo[3] = sin(theta) * crossv[2];

	Eigen::Matrix<double,3,3> R2;
	
	R2(0,0) = 2*quo[0]*quo[0] + 2*quo[1]*quo[1] - 1;
	R2(0,1) = 2*(quo[1] * quo[2] + quo[0] * quo[3]);
	R2(0,2) = 2*(quo[1] * quo[3] - quo[0] * quo[2]);
	
	R2(1,0) = 2*(quo[1] * quo[2] - quo[0] * quo[3]);
	R2(1,1) = 2*quo[0]*quo[0] + 2*quo[2]*quo[2] - 1;
	R2(1,2) = 2*(quo[2] * quo[3] + quo[0] * quo[1]);

	R2(2,0) = 2*(quo[1] * quo[3] + quo[0] * quo[2]);
	R2(2,1) = 2*(quo[2] * quo[3] - quo[0] * quo[1]);
	R2(2,2) = 2*quo[0]*quo[0] + 2*quo[3]*quo[3] - 1;

	for(int i = 0;i<B0.vertices.size();i++) {
        Eigen::Matrix<double, 1, 3> t(B0.vertices[i][0], B0.vertices[i][1], B0.vertices[i][2]);
        t = t * R2;
        B0.vertices[i][0] = t(0, 0);
        B0.vertices[i][1] = t(0, 1);
        B0.vertices[i][2] = t(0, 2);
    }

	for(int i = 0;i<B0.vertices.size();i++)
	{
		B0.vertices[i] += tempmove1;
		B0.vertices[i] += tempmove2;
	}
    aligning = aligning - B0.vertices[5746];

    for(int i = 0;i<B0.vertices.size();i++)
    {
        B0.vertices[i] += aligning;
    }

	B0.write("../result/Data_aligned.obj");

    /*
      const int nshapes = 47;  //47

    std::vector<trimesh::TriMesh> A(nshapes);

	for(int i=0; i<nshapes; i++){
        A[i] = *trimesh::TriMesh::read("../data/bahe/out_" + std::to_string(i) + ".obj");
    }

	for(int i=0; i<A.size();i++)
	{
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] *= d;
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] += v3;
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] -= tempmove1;
		for(int j=0;j<A[i].vertices.size();j++)
		{
			if(j == 15603) continue;
			Eigen::Matrix<double,1,3> t(A[i].vertices[j][0],A[i].vertices[j][1],A[i].vertices[j][2]);
			t = t * R1;
			A[i].vertices[j][0] = t(0,0);
			A[i].vertices[j][1] = t(0,1);
			A[i].vertices[j][2] = t(0,2);
		}
		for(int j=0;j<A[i].vertices.size();j++)
			A[i].vertices[j] -= tempmove2;
		for(int j=0;j<A[i].vertices.size();j++)
		{
			Eigen::Matrix<double,1,3> t(A[i].vertices[j][0],A[i].vertices[j][1],A[i].vertices[j][2]);
			t = t * R2;
			A[i].vertices[j][0] = t(0,0);
			A[i].vertices[j][1] = t(0,1);
			A[i].vertices[j][2] = t(0,2);
		}
		for(int j=0;j<A[i].vertices.size();j++)
		{
			A[i].vertices[j] += tempmove1;
			A[i].vertices[j] += tempmove2;
		}
		A[i].write("../data/bahe/blendshape_aligned_" + std::to_string(i) + ".obj");
	}
     */

}
void get_files_list(vector<string> &files,const string PATH)
{
    struct dirent *ptr;
    DIR *dir;
    //string PATH = "./file";
    dir=opendir(PATH.c_str());
    while((ptr=readdir(dir))!=NULL)
    {
        if(ptr->d_name[0] == '.')
            continue;
        //cout << ptr->d_name << endl;
        files.push_back(ptr->d_name);
    }
    closedir(dir);
}

vector<trimesh::Vec<3,int>> building_edges(trimesh::TriMesh &shape,const int &f1, const int &f2)
{
    cout << "building edges for " << f1 << " & " << f2 << endl;

    vector<trimesh::Vec<3,int>> results;
    int A1 = shape.faces[f1][0];
    int A2 = shape.faces[f1][1];
    int A3 = shape.faces[f1][2];

    int B1 = shape.faces[f2][0];
    int B2 = shape.faces[f2][1];
    int B3 = shape.faces[f2][2];

    vector<float> Dists(6);
    Dists[0] = trimesh::dist(shape.vertices[A1],shape.vertices[B1]) + trimesh::dist(shape.vertices[A2],shape.vertices[B2])+ trimesh::dist(shape.vertices[A3],shape.vertices[B3]);
    Dists[1] = trimesh::dist(shape.vertices[A1],shape.vertices[B1]) + trimesh::dist(shape.vertices[A2],shape.vertices[B3])+ trimesh::dist(shape.vertices[A3],shape.vertices[B2]);
    Dists[2] = trimesh::dist(shape.vertices[A1],shape.vertices[B2]) + trimesh::dist(shape.vertices[A2],shape.vertices[B1])+ trimesh::dist(shape.vertices[A3],shape.vertices[B3]);
    Dists[3] = trimesh::dist(shape.vertices[A1],shape.vertices[B3]) + trimesh::dist(shape.vertices[A2],shape.vertices[B2])+ trimesh::dist(shape.vertices[A3],shape.vertices[B1]);
    Dists[4] = trimesh::dist(shape.vertices[A1],shape.vertices[B3]) + trimesh::dist(shape.vertices[A2],shape.vertices[B1])+ trimesh::dist(shape.vertices[A3],shape.vertices[B2]);
    Dists[5] = trimesh::dist(shape.vertices[A1],shape.vertices[B2]) + trimesh::dist(shape.vertices[A2],shape.vertices[B3])+ trimesh::dist(shape.vertices[A3],shape.vertices[B1]);

    float mindist = Dists[0];
    int choice = 0;
    for(int i = 0;i<Dists.size();i++)
    {
        if(mindist < Dists[i])
        {
            mindist = Dists[i];
            choice = i;
        }
    }
    int T1 = B1;
    int T2 = B2;
    int T3 = B3;

    switch (choice) {
        case 0: break;
        case 1: { B2 = T3; B3 = T2; break; }
        case 2: { B2 = T1; B1 = T2; break; }
        case 3: { B1 = T3; B3 = T1; break; }
        case 4: { B1 = T3; B2 = T1; B3 = T2; break; }
        case 5: { B1 = T2; B2 = T3; B3 = T1; break; }
        default: break;
    }
    //cout << "The choice is " << choice << endl;
    //cout << A1 << "-" << B1 << endl;
    //cout << A2 << "-" << B2 << endl;
    //cout << A3 << "-" << B3 << endl;
    double D[3];
    D[0] = trimesh::dist(shape.vertices[A1],shape.vertices[B1]);
    D[1] = trimesh::dist(shape.vertices[A2],shape.vertices[B2]);
    D[2] = trimesh::dist(shape.vertices[A3],shape.vertices[B3]);

    double mind = D[0];
    double maxd = D[0];
    for(int i = 0;i<=2;i++)
    {
        if(D[i] > maxd) maxd = D[i];
        if(D[i] < mind) mind = D[i];
    }

    //if(maxd * 0.7 > mind) return results;

    results.emplace_back(trimesh::Vec<3,int>(A1,B1,B3));
    cout << "tri: " << A1 << "-" << B1 << "-" << B3 << endl;
    results.emplace_back(trimesh::Vec<3,int>(A1,A3,B3));
    cout << "tri: " << A1 << "-" << A3 << "-" << B3 << endl;
    results.emplace_back(trimesh::Vec<3,int>(A1,B1,A2));
    cout << "tri: " << A1 << "-" << B1 << "-" << A2 << endl;
    results.emplace_back(trimesh::Vec<3,int>(B1,A2,B2));
    cout << "tri: " << B1 << "-" << A2 << "-" << B2 << endl;
    results.emplace_back(trimesh::Vec<3,int>(A2,B2,A3));
    cout << "tri: " << A2 << "-" << B2 << "-" << A3 << endl;
    results.emplace_back(trimesh::Vec<3,int>(B2,A3,B3));
    cout << "tri: " << B2 << "-" << A3 << "-" << B3 << endl;
    return results;
}

void matching_faces(trimesh::TriMesh &shape, trimesh::TriMesh &deform_shape, string name, int index_out, int index_in)
{
    //  For each vertex, all neighboring vertices
    //::std::vector< ::std::vector<int> > neighbors;
    //  For each vertex, all neighboring faces
    //::std::vector< ::std::vector<int> > adjacentfaces;
    //  For each face, the three faces attached to its edges
    //  (for example, across_edge[3][2] is the number of the face
    //   that's touching the edge opposite vertex 2 of face 3)
    //::std::vector<Face> across_edge;

    int original_face_num = shape.faces.size();
    deform_shape.write("../result/"+name+"_deform_source.obj");
    cout << "start matching" << endl;
    shape.need_adjacentfaces();
    shape.need_neighbors();
    shape.need_across_edge();

    set <int> nearest_indexes;
    nearest_indexes.insert(index_in);
    for(int t = 4;t>=0;t--)
    {
        set <int> temp_neighobrs;
        for(int i : nearest_indexes)
        {
            //cout << "vi : " << i << endl;
            for(int temp : shape.neighbors[i])
            {
                //cout << "neighbors : " << temp << endl;
                if(nearest_indexes.count(temp) == 0) temp_neighobrs.insert(temp);
            }
        }
        for(int temp : temp_neighobrs) nearest_indexes.insert(temp);
        temp_neighobrs.clear();
    }

    float mindist = trimesh::dist(shape.vertices[index_in],shape.vertices[index_out]);
    for(int index : nearest_indexes)
    {
        if(trimesh::dist(shape.vertices[index],shape.vertices[index_out]) < mindist)
        {
            //index_in = index;
            mindist = trimesh::dist(shape.vertices[index],shape.vertices[index_out]);
        }
    }
    nearest_indexes.clear();
    //cout << "Nearest Point : " << index_in << endl;
    //cout << "starting counting inside faces: " << endl;
    set <int> reigns_inside;
    set <int> faces;
    vector<int> adj = shape.adjacentfaces[index_in];
    for(int i = 0;i<adj.size();i++)
    {
        //cout << adj[i] << endl;
        faces.insert(adj[i]);
    }
    int t = 4;
    set <int> temp;
    for(int i = 0;i<t;i++)
    {
        //cout << "Turn : " << i << endl;
        temp.clear();
        for(int f : faces)
        {

            for(int adj_f : shape.across_edge[f])
            {
                temp.insert(adj_f);
                reigns_inside.insert(adj_f);
                //cout << adj_f << endl;
            }
        }
        faces = temp;
    }
    cout << "Inside Result: " << endl;
    for(int r : reigns_inside) cout << "fi == " << r << " || ";
    unordered_map <int,trimesh::point> face_cords_in;
    for(int f : reigns_inside)
    {
        //cout << f << endl;
        face_cords_in[f] = shape.vertices[shape.faces[f][0]] + shape.vertices[shape.faces[f][1]] + shape.vertices[shape.faces[f][2]];
        //cout << face_cords_in[f] << endl;
        face_cords_in[f] /= 3;
        //cout << face_cords_in[f] << endl;
    }

    adj.clear();
    faces.clear();
    //cout << "starting counting outside faces: " << endl;
    set <int> reigns_outside;
    adj = shape.adjacentfaces[index_out];
    for(int i = 0;i<adj.size();i++)
    {
        //cout << adj[i] << endl;
        faces.insert(adj[i]);
    }
    t+=1;
    temp.clear();
    for(int i = 0;i<t;i++)
    {
        //cout << "Turn : " << i << endl;
        temp.clear();
        for(int f : faces)
        {

            for(int adj_f : shape.across_edge[f])
            {
                temp.insert(adj_f);
                reigns_outside.insert(adj_f);
                //cout << adj_f << endl;
            }
        }
        faces = temp;
    }
    cout << "Outside Result: " << endl;
    for(int r : reigns_outside) cout << "fi == " << r << " || ";
    unordered_map <int,trimesh::point> face_cords_out;
    set <int> outside_indexes;
    for(int f : reigns_outside)
    {
        outside_indexes.insert(shape.faces[f][0]);
        outside_indexes.insert(shape.faces[f][1]);
        outside_indexes.insert(shape.faces[f][2]);
        //cout << f << endl;
        face_cords_out[f] = shape.vertices[shape.faces[f][0]] + shape.vertices[shape.faces[f][1]] + shape.vertices[shape.faces[f][2]];
        //cout << face_cords_out[f] << endl;
        face_cords_out[f] /= 3;
        //cout << face_cords_out[f] << endl;
    }

    unordered_map <int,int> match_out_in;
    set <int> inside_indexes;
    for(int fin : reigns_inside)
    {
        inside_indexes.insert(shape.faces[fin][0]);
        inside_indexes.insert(shape.faces[fin][1]);
        inside_indexes.insert(shape.faces[fin][2]);

        mindist = 99999;
        int match = -1;
        for(int fout : reigns_outside)
        {
            if(trimesh::dist(face_cords_in[fin],face_cords_out[fout]) < mindist)
            {
                match = fout;
                mindist = trimesh::dist(face_cords_in[fin],face_cords_out[fout]);
            }
        }
        if(match_out_in[match] == 0) match_out_in[match] = fin;
        else
        {
            if(mindist < trimesh::dist(face_cords_out[match],face_cords_in[match_out_in[match]]))
            {
                match_out_in[match] = fin;
            }
            else match_out_in[match] = match_out_in[match];
        }
    }
    unordered_map <int,int> match_in_out;

    for(int face : reigns_outside)
    {
        if(match_out_in[face]!=0)
        {
            match_in_out[match_out_in[face]] = face;
        }
    }

    //vector<trimesh::Vec<3,int>> Alledges;

    //vector<trimesh::Vec<3,int>> minused_faces;
    //for(int i = 0;i<shape.faces.size();i++)
    //{
        //if(inside_indexes.count(shape.faces[i][0]) == 0 &&inside_indexes.count(shape.faces[i][1]) == 0 && inside_indexes.count(shape.faces[i][2]) == 0)
        //{
            //minused_faces.emplace_back(shape.faces[i]);
        //}
    //}
    //shape.faces = minused_faces;
    //shape.write("../result/edge_minused.obj");

    set <int> vertices_to_add;
    vector<trimesh::Vec<3,int>> faces_to_add;
    for(int fout : reigns_outside)
    {
        if(match_out_in[fout] != 0)
        {
            //cout << fout << " : " << match_out_in[fout] << endl;
            trimesh::Vec<3,int> newface = shape.faces[0];
            vector<trimesh::Vec<3,int>> newedges = building_edges(shape,fout, match_out_in[fout]);
            for(int u = 0;u<newedges.size();u++)
            {
                 if(vertices_to_add.count(newedges[u][0]) == 0 || vertices_to_add.count(newedges[u][1]) == 0 ||vertices_to_add.count(newedges[u][2]) == 0)
                {
                    faces_to_add.emplace_back(newedges[u]);
                    vertices_to_add.insert(newedges[u][0]);
                    vertices_to_add.insert(newedges[u][1]);
                    vertices_to_add.insert(newedges[u][2]);
                }
            }
        }
    }

    for(int i = 0;i < faces_to_add.size() ;i++)
    {
       shape.faces.emplace_back(faces_to_add[i]);
    }

    shape.write("../result/"+name+"_edge_added.obj");

    unordered_map<int,int> weights_factor;

    for(int in : inside_indexes)
    {
        for(int face : shape.adjacentfaces[in])
        {
            if(match_in_out[face]!=0)
            {
                weights_factor[in] += 1;
            }
        }
    }




    //shape.faces = minused_faces;
    //shape.write("../result/edge_added.obj");
    /*
    vector<int> anchor_list = {7146,7147,6714,6712,6715,6291,5880,6293,5884,5887,5886,5489,5489,5889,5492,5493,5491,5893,5496,5497,5897,5898,5501,5903,5901,
                                  5902,5507,5909,5905,6322,8969,8088,8089,7672,8110,8113,8554,8104,8547,8548,8096,8539,8983,8978,8977,9414,9415,9842,10260,10680,
                                  11119,11121,11550,11970,12404,12817,13230,13283,12875,12469,12055,11693,11692,11689,12097,12098,12514,12924,13335,13336,14113,14523,
                                  14915,15287,15637,15953,15978,15979,16264,16534,16796,16540,16800,17032,17034,16976,16747,16491,16489,16219,15604,15602,15602,15253,
                                  14489,14079,13680,13681,13681,13271,13272,13269,13682,13687,12856,13688,13275,13276,13276,13693,13692,13361,12953,12954,12951,12546,
                                  12131,11726,11726,11724,11314,10905,10500,10101,9702,9297,9295,8879,8014,8015,7582,7581,14569,13749,12916,12499,11666,10835,10424,9608,
                                  8772,9612,20736,20733,21785,21991,21931,21586,7233,7683,8583,9027,9605,8314,7867,7461,7021,18855,18852,19904,9190,8590,7688,7238,21370,
                                  21349,21494,21478,20827,21063,21024,21659,21718,21122,21123,21857,21138,19488,19452,19468,19613,19597,18945,19143,19267,19236,19249,
                                  19974,19961,19251,6588,6167,6170,5763,6176,6521,6958,6835,7271,9590,8307,6805,5534,4892,6553,4947,1951,1461,1957,2271,2588,2905,3561,
                                  3898,19757,6136,5353,4609,4205,4564,5307,5706,6437,4446,3741,2753,2435,1601,1395,1186,996,539,93,66,266,1330,1800,2078,2996,3649,3991,
                                  4692,5445,6254,6682,7550,6467,3758,3091,2157,1619,1386,978,815,529,306,143,52,173,761,1109,1781,2060,2663,3303,3971,5034,5040,5823,6663,
                                  10332,2726,7892,2804,5864,10027,21332,21333,21346,21347,21348,21352,22070,21495,21474,21473,21473,21475,21492,21492,21493,21493,20826,20825,
                                  21069,21067,21034,21030,21032,21027,20360,20357,21662,21660,21698,21698,21697,21697,21141,21145,21142,21143,21120,21120,21121,21121,21130,
                                  21128,21877,21877,21855,21837,21394,21389,19486,19486,19451,19452,19465,19466,19466,19467,19472,19469,20189,19614,19593,19592,19592,19594,
                                  19611,19612,18946,18944,19188,19182,19149,19146,18476,18474,19774,19270,19260,19258,19258,19235,19237,19238,19247,19247,19996,19972,19976,
                                  19975,19957,19957,19958,19958,19512,19507,19499,19495};
    */
    /* female
    vector<int> anchor_list = { 2117, 1545, 1378, 1528, 1395, 1369, 2148, 1526,//left_eye
                                8002, 8000, 8676, //right eye up
                                9078,9178,9094,9117,9075,9076,
                               3774,3144,9631,2834,3078,9567,9931,3490,5118,12540,3834,4982,11380,4023,10427,5329,11718,4698,11097,5327,4044,10449,6346,12685};//nose&face;
    vector<int> anchor_fix = {8748, 7855, 7835, 7845, 12102}; //right eye down

    */
    set <int> anchor_indexes;
    /*
     vector<int> anchor_list = {7035, 1614, 2948, 4027, 7234, 1600, 2958, 8553, 8550, 2956, 6748, 1098,
                               5565, 7270, 11034, 2904, 4148, 4196, 9676, 4127, 9738, 4190, 9813, 4272,
                               1060, 6657, 6819, 993, 909, 266, 6513, 1037};
     * */
    //orc
    vector<int> anchor_list = {21006, 7915, 15318, 10544, 15572, 2323, 813, 13764, 459, 21455, 380, 1524, 21506, 8508, 1589, 22964, 9925};

    for(int f : outside_indexes) anchor_indexes.insert(f);
    for(int i = 0;i<anchor_list.size();i++) anchor_indexes.insert(anchor_list[i]);


    vector<int> landmarks;
    vector<Eigen::Vector3d> landmarks_3d;
    //int anchor_point = 9050;
    //trimesh::point move = deform_shape.vertices[anchor_point] - shape.vertices[anchor_point];

    for(int anc : anchor_indexes)
    {
        Eigen::Vector3d temp;
        temp[0] = deform_shape.vertices[anc][0];
        temp[1] = deform_shape.vertices[anc][1];
        temp[2] = deform_shape.vertices[anc][2];
        landmarks.emplace_back(anc);
        landmarks_3d.emplace_back(temp);
    }

    trimesh::TriMesh deformedMesh;

    LaplacianDeformer laplacianDeformer;

    laplacianDeformer.setSource(shape);

    // test cases
    std::vector<int>             anchor_indexs;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > anchor_coords;
    std::vector<double>          anchor_weights;

    for(int i=0; i<landmarks.size(); i++){
        cout << "vi == " << landmarks[i] << " || ";
        anchor_indexs.push_back(landmarks[i]);
        anchor_coords.push_back(landmarks_3d[i]);
        anchor_weights.push_back(1.0);
    }

    cout << anchor_indexs.size() << endl;
    cout << anchor_coords.size() << endl;
    cout << anchor_weights.size() << endl;

    cout << "Anchoring finished" << endl;
    laplacianDeformer.setAnchors(anchor_indexs, anchor_coords, anchor_weights); // user-specific
    cout << "Anchor setted" << endl;
    laplacianDeformer.preCompute();

    cout << "Starting Solve" << endl;
    laplacianDeformer.solve();
    trimesh::TriMesh ret = laplacianDeformer.getResult();
    ret.write("../result/"+name+"_edge_deformed.obj");
    deformedMesh = ret;
    //cout << "done!" << endl;
    trimesh::TriMesh source = deform_shape;
    for(int face : reigns_outside)
    {
        if(match_out_in[face] != 0)
        {
            trimesh::point movement;
            trimesh::point newcenter = deformedMesh.vertices[deformedMesh.faces[face][0]] + deformedMesh.vertices[deformedMesh.faces[face][1]] + deformedMesh.vertices[deformedMesh.faces[face][2]];
            newcenter /= 3;
            movement = newcenter - face_cords_out[face];

            for(int index : shape.faces[match_out_in[face]])
            {
                shape.vertices[index] += (movement/weights_factor[index]);
            }

        }
    }

    shape.write("../result/"+name+"_edge_warpped.obj");

    for(int index : inside_indexes)
    {
        if(weights_factor[index]!=0)
        {
            anchor_indexs.push_back(index);
            Eigen::Vector3d temp;
            temp[0] = shape.vertices[index][0];
            temp[1] = shape.vertices[index][1];
            temp[2] = shape.vertices[index][2];
            anchor_coords.push_back(temp);
            anchor_weights.push_back(0.4);
        }
    }
    //cout << "next!" << endl;
    LaplacianDeformer l2;
    l2.setSource(source);
    l2.setAnchors(anchor_indexs, anchor_coords, anchor_weights); // user-specific
    l2.preCompute();
    l2.solve();
    ret = l2.getResult();
    ret.write("../result/"+name+"_final_deformed.obj");
    ret.faces.resize(original_face_num);
    ret.write("../result/"+name+"_final_deformed_noedges.obj");
    return;
}


trimesh::TriMesh mix_mesh(trimesh::TriMesh &B0,trimesh::TriMesh &Bm,double ratio)
{
    trimesh::TriMesh db = Bm;
    for(int j=0; j<db.vertices.size(); j++){
        db.vertices[j] -= B0.vertices[j];
    }
    trimesh::TriMesh mix_result = B0;
    for(int j=0; j<mix_result.vertices.size(); j++){
        mix_result.vertices[j] += ratio * db.vertices[j];
    }
    mix_result.write("../result/mesh_mixed.obj");
    return mix_result;
}

trimesh::TriMesh double_mix_mesh(trimesh::TriMesh &B0,trimesh::TriMesh &Bm1,trimesh::TriMesh &Bm2,double ratio)
{
    trimesh::TriMesh db = Bm1;
    for(int j=0; j<db.vertices.size(); j++){
        db.vertices[j] -= B0.vertices[j];
    }
    trimesh::TriMesh mix_result = B0;
    for(int j=0; j<mix_result.vertices.size(); j++){
        mix_result.vertices[j] += ratio * db.vertices[j];
    }
    db = Bm2;
    for(int j=0; j<db.vertices.size(); j++){
        db.vertices[j] -= B0.vertices[j];
    }
    for(int j=0; j<mix_result.vertices.size(); j++){
        mix_result.vertices[j] += ratio * db.vertices[j];
    }
    mix_result.write("../result/mesh_mixed.obj");
    return mix_result;
}

void searching_paths(trimesh::TriMesh &shape,int face,int start,int step,int limit,vector<int> &visited)
{
    //shape.need_across_edge();
    shape.need_adjacentfaces();
    if(step >= limit) return;
    else if(visited[face] != 0 && visited[face] <= step) return;
    cout << "seraching paths: " << start << " - " << face << endl;
    //cout << visited.size() << endl;
    visited[face] = step;
    //cout << shape.across_edge[face][0] << endl;
    //cout << shape.across_edge[face][1] << endl;
    //cout << shape.across_edge[face][2] << endl;
    for(int p : shape.faces[face])
    {
        for(int adj_face : shape.adjacentfaces[p])
        {
            searching_paths(shape,adj_face,start,step + 1,limit,visited);
        }
    }
    return;
}
double distance(vector<double> A,vector<double> B)
{
    double result = (A[0] - B[0])*(A[0] - B[0]) + (A[1] - B[1])*(A[1] - B[1]) + (A[2] - B[2])*(A[2] - B[2]);
    result = sqrt(result);
    return result;
}
bool is_inside(int face,trimesh::TriMesh &mask)
{
    //cout << mask.colors[6489] << endl;
    //cout << mask.colors[20786] << endl;
    //cout << mask.colors[18198] << endl;
    if(mask.colors[mask.faces[face][0]] == mask.colors[4039]) return false;
    return true;
    if(mask.colors[mask.faces[face][0]] == mask.colors[6489] || mask.colors[mask.faces[face][0]] == mask.colors[20786] ||
            mask.colors[mask.faces[face][0]] == mask.colors[18198]) return true;
    return false;
}
void density_control(vector<vector<int>> &labels, vector<int> &outside_faces,vector<int> &result, int density, int cur)
{
    cout << "Density Control: " << outside_faces[cur] << endl;
    int mindist = 999;
    int next = -1;
    for(int i = 0;i < outside_faces.size();i++)
    {
        if(result[outside_faces[i]] != 0) continue;
        if((labels[outside_faces[cur]][outside_faces[i]]) <= density && i != cur && labels[outside_faces[cur]][outside_faces[i]] != 0)
        {
            result[outside_faces[i]] = -1;
        }
        else if(labels[outside_faces[cur]][outside_faces[i]] != 0)
        {
            if(labels[outside_faces[cur]][outside_faces[i]] < mindist)
            {
                mindist = labels[outside_faces[cur]][outside_faces[i]];
                next = i;
            }
        }
    }
    result[outside_faces[cur]] = 1;
    if(mindist != 999) density_control(labels, outside_faces, result, density, next);
    else
    {
        for(int i = 0;i<outside_faces.size();i++)
        {
            if(result[outside_faces[i]] == 0)
            {
                next = i;
                break;
            }
        }
        if(next != -1) density_control(labels, outside_faces, result, density, next);
        else return;
    }
}
void auto_matching_faces_by_color(trimesh::TriMesh &shape,double length)
{
    omp_set_num_threads(8);
    trimesh::TriMesh mask = *trimesh::TriMesh::read("../Mask_lzj2.ply");
    vector<int> outside_faces;
    vector<int> inside_faces;
    for(int i = 0;i<shape.faces.size();i++)
    {
        if(mask.colors[mask.faces[i][0]] == mask.colors[4039] || mask.colors[mask.faces[i][1]] == mask.colors[4039] || mask.colors[mask.faces[i][2]] == mask.colors[4039])
            outside_faces.push_back(i);
        if(mask.colors[mask.faces[i][0]] == mask.colors[1477] || mask.colors[mask.faces[i][1]] == mask.colors[1477] || mask.colors[mask.faces[i][2]] == mask.colors[1477])
            inside_faces.push_back(i);
    }
    vector<vector<double>> face_cords;
    face_cords.resize(shape.faces.size());
    for(int i = 0;i<shape.faces.size();i++)
    {
        int A = shape.faces[i][0];
        int B = shape.faces[i][1];
        int C = shape.faces[i][2];
        face_cords[i].push_back((shape.vertices[A][0] + shape.vertices[B][0] + shape.vertices[C][0])/3);
        face_cords[i].push_back((shape.vertices[A][1] + shape.vertices[B][1] + shape.vertices[C][1])/3);
        face_cords[i].push_back((shape.vertices[A][2] + shape.vertices[B][2] + shape.vertices[C][2])/3);
    }
    cout << "cords done" << endl;
    set <pair<int,int>> face_pairs;
    #pragma omp parallel for
    for(int i = 0;i<outside_faces.size();i++)
    {
        double mindist = 10000;
        int nearest_face = -1;
        for(int j = 0;j<inside_faces.size();j++)
        {
            double temp_d = distance(face_cords[outside_faces[i]], face_cords[inside_faces[j]]);
            if (temp_d < mindist) {
                mindist = temp_d;
                nearest_face = inside_faces[j];
            }
        }
        if(mindist < length) face_pairs.insert(pair<int,int>(outside_faces[i],nearest_face));
    }
    vector<set<int>> outside_faces_neighobrs;
    outside_faces_neighobrs.resize(shape.faces.size());
    shape.need_adjacentfaces();
    int density = 3;
    for(int i = 0;i<outside_faces.size();i++)
    {
        cout << "Controling Density " << outside_faces[i] << endl;
        std::queue<int> temp;
        temp.push(outside_faces[i]);
        int counter = 0;
        while(counter < density)
        {
            vector<int> currentfaces;
            while(!temp.empty())
            {
                currentfaces.push_back(temp.front());
                temp.pop();
            }
            for(int adj_face : currentfaces)
            {
                outside_faces_neighobrs[outside_faces[i]].insert(adj_face);
                for(int vertex : shape.faces[adj_face])
                for(int neighbors : shape.adjacentfaces[vertex])
                {
                    temp.push(neighbors);
                }
            }
            counter++;
        }
        for(int c : outside_faces_neighobrs[outside_faces[i]])
            cout << c << " ";
        cout << endl;
    }
    cout << "Next Step" << endl;
    vector<int> face_flags(shape.faces.size());
    for(int i = 0;i<shape.faces.size();i++) face_flags[i] = 0;
    set <int> available_outside_faces;
    int p = outside_faces[0];
    while(true)
    {
        cout << p << endl;
        if(face_flags[p] == 0)
        {
            available_outside_faces.insert(p);
            cout << "Insert: " << p << endl;
            for(int n : outside_faces_neighobrs[p])
            {
                cout << "Cut Down: " << n << endl;
                face_flags[n] = 1;
            }
        }
        int next = -1;
        double mindist = 10000;
        for(int p2 : outside_faces)
        {
            if(distance(face_cords[p],face_cords[p2]) < mindist && face_flags[p2] == 0)
            {
                mindist = distance(face_cords[p],face_cords[p2]);
                next = p2;
            }
        }
        if(next != -1)
        {
            p = next;
        }
        else break;
    }
    ofstream data("../result/density.txt");
    int total = 0;
    for(pair<int,int> p : face_pairs)
    {
        if(available_outside_faces.count(p.first) > 0)
        {
            vector<trimesh::Vec<3, int>> new_faces = building_edges(shape, p.first, p.second);
            data << p.first;
            data << "-";
            data << p.second;
            data << endl;
            for (int i = 0; i < new_faces.size(); i++)
            {
                shape.faces.push_back(new_faces[i]);
            }
            total++;
        }
    }
    cout << total * 6 << endl;
    shape.write("../result/mesh_edge_added.obj");
}
void auto_matching_faces(trimesh::TriMesh &shape,int limit,double length)
{
    //  For each vertex, all neighboring vertices
    //::std::vector< ::std::vector<int> > neighbors;
    //  For each vertex, all neighboring faces
    //::std::vector< ::std::vector<int> > adjacentfaces;
    //  For each face, the three faces attached to its edges
    //  (for example, across_edge[3][2] is the number of the face
    //   that's touching the edge opposite vertex 2 of face 3)
    //::std::vector<Face> across_edge;
    omp_set_num_threads(8);
    //trimesh::TriMesh mask = *trimesh::TriMesh::read("../S/emily-mask-23686.ply");
    trimesh::TriMesh mask = *trimesh::TriMesh::read("../Mask.ply");
    cout << mesh_center_of_mass(&shape) << endl;
    double dx = mesh_center_of_mass(&shape)[0];
    double dy = mesh_center_of_mass(&shape)[1];
    double dz = mesh_center_of_mass(&shape)[2];
    //return;
    for(int i = 0;i<shape.vertices.size();i++)
    {
        shape.vertices[i][0] -= dx;
        shape.vertices[i][1] -= dy;
        shape.vertices[i][2] -= dz;
    }
    //shape.write("../result/mesh_centered.obj");
    //return;
    vector<int> reigons(shape.faces.size());
    vector<vector<int>> reigons_faces(8);
    set <pair<int,int>> face_pairs;
    shape.need_adjacentfaces();
    shape.need_neighbors();

    vector<vector<int>> labels;
    labels.resize(shape.faces.size());
    #pragma omp parallel for
    for(int i = 0;i<shape.faces.size();i++)
    {
        cout << "searching face: " << i << endl;
        vector<int> visited;
        visited.resize(shape.faces.size());
        searching_paths(shape,i, i,0,limit,visited);
        labels[i] = visited;
        visited.clear();
    }
    cout << "mapping done" << endl;
    for(int i = 0;i<labels.size();i++)
    {
        //cout << "center: " << i << endl;
        for(int j = 0;j<labels[i].size();j++)
        {
            //if(labels[i][j] == 1) cout << "fi == " << j << " || ";
        }
        //cout << endl;
    }
    //return;


    vector<vector<double>> face_cords;
    face_cords.resize(shape.faces.size());
    for(int i = 0;i<shape.faces.size();i++)
    {
        int A = shape.faces[i][0];
        int B = shape.faces[i][1];
        int C = shape.faces[i][2];
        face_cords[i].push_back((shape.vertices[A][0] + shape.vertices[B][0] + shape.vertices[C][0])/3);
        face_cords[i].push_back((shape.vertices[A][1] + shape.vertices[B][1] + shape.vertices[C][1])/3);
        face_cords[i].push_back((shape.vertices[A][2] + shape.vertices[B][2] + shape.vertices[C][2])/3);
        //cout << face_cords[i][0] << " " << face_cords[i][1] << face_cords[i][2] << endl;
        if(face_cords[i][0] <= 0 && face_cords[i][1] <= 0 && face_cords[i][2] <= 0) reigons[i] = 0;
        if(face_cords[i][0] <= 0 && face_cords[i][1] <= 0 && face_cords[i][2] >= 0) reigons[i] = 1;
        if(face_cords[i][0] <= 0 && face_cords[i][1] >= 0 && face_cords[i][2] <= 0) reigons[i] = 2;
        if(face_cords[i][0] <= 0 && face_cords[i][1] >= 0 && face_cords[i][2] >= 0) reigons[i] = 3;
        if(face_cords[i][0] >= 0 && face_cords[i][1] <= 0 && face_cords[i][2] <= 0) reigons[i] = 4;
        if(face_cords[i][0] >= 0 && face_cords[i][1] <= 0 && face_cords[i][2] >= 0) reigons[i] = 5;
        if(face_cords[i][0] >= 0 && face_cords[i][1] >= 0 && face_cords[i][2] <= 0) reigons[i] = 6;
        if(face_cords[i][0] >= 0 && face_cords[i][1] >= 0 && face_cords[i][2] >= 0) reigons[i] = 7;

        reigons_faces[reigons[i]].push_back(i);
    }
    cout << "cords done" << endl;
    //return;
    #pragma omp parallel for
    for(int i = 0;i<shape.faces.size();i++)
    {
        cout << "matching: " << i << endl;
        double mindist = 10000;
        int minface = -1;
        //cout << "reigon: " << reigons[i] << endl;
        //cout << "reigon size: " << reigons_faces[reigons[i]].size() << endl;
        for (int t = 0; t < reigons_faces[reigons[i]].size(); t++)
        {

            int j = reigons_faces[reigons[i]][t];
            if(labels[i][j] == 1 || labels[j][i] == 1 || i == j) continue;
            else {
                double temp_d = distance(face_cords[i], face_cords[j]);
                if (temp_d < mindist) {
                    mindist = temp_d;
                    minface = j;
                }
            }
        }
        if(mindist < length)
        {
            cout << "Dist: " << mindist << endl;
            cout << "Length: " << length << endl;
            cout << "base-target" << i << " - " << minface << endl;
            cout << labels[i][minface] << endl;
            cout << labels[minface][i] << endl;
            if(i < minface)
            {
                face_pairs.insert(pair<int,int>(i,minface));
            }
            else if(minface < i)
            {
                face_pairs.insert(pair<int,int>(minface,i));
            }
        }
    }
    cout << "matching done" << endl;
    vector<int> matches(shape.faces.size());
    for(int i = 0;i<matches.size();i++) matches[i] = 0;
    for(pair<int,int> unit : face_pairs)
    {
        //if((unit.first - unit.second) > -500 && (unit.first - unit.second) < 500) continue;
        if(is_inside(unit.first,mask) && is_inside(unit.second,mask)) continue;
        else if(!is_inside(unit.first,mask) && !is_inside(unit.second,mask)) continue;
        cout << unit.first << "-" << unit.second << endl;
        if(is_inside(unit.first,mask))
        {
            if(matches[unit.first])
            {
                if(distance(face_cords[unit.first],face_cords[unit.second]) < distance(face_cords[unit.first],face_cords[matches[unit.first]]))
                    matches[unit.first] = unit.second;
            }
            else matches[unit.first] = unit.second;
        }
        else
        {
            if(matches[unit.second])
            {
                if(distance(face_cords[unit.first],face_cords[unit.second]) < distance(face_cords[unit.second],face_cords[matches[unit.second]]))
                    matches[unit.second] = unit.first;
            }
            else matches[unit.second] = unit.first;
        }
    }
    vector<int> outside_faces;
    for(int i = 0;i < shape.faces.size();i++)
    {
        //outside_faces.push_back(matches[i]);
        if(matches[i] != 0)
        {
            outside_faces.push_back(matches[i]);
        }
    }
    vector<int> density_control_result(shape.faces.size());
    cout << "Starting Control" << endl;
    cout << outside_faces.size() << endl;
    density_control(labels,outside_faces,density_control_result,1,0);
    for(int i = 0;i<matches.size();i++)
    {
        if(density_control_result[matches[i]] != 1) matches[i] = 0;
    }

    ofstream data("../result/density.txt");
    for(int i = 0;i<matches.size();i++)
    {
        if(matches[i]) {
            vector<trimesh::Vec<3, int>> new_faces = building_edges(shape, i, matches[i]);
            data << i;
            data << "-";
            data << matches[i];
            data << endl;
            for (int i = 0; i < new_faces.size(); i++)
            {
                shape.faces.push_back(new_faces[i]);
            }
        }
    }
    shape.write("../result/mesh_edge_added.obj");
    return;
}
void mix_by_step(trimesh::TriMesh &B0,trimesh::TriMesh &target)
{
    trimesh::TriMesh db = target;
    for(int j=0; j<db.vertices.size(); j++){
        db.vertices[j] -= B0.vertices[j];
    }
    trimesh::TriMesh mix_result;
    int counter = 0;
    for(double ratio = 0;ratio<=1.2;ratio+=0.01)
    {
        mix_result = B0;
        for(int j=0; j<mix_result.vertices.size(); j++){
            mix_result.vertices[j] += ratio * db.vertices[j];
        }
        mix_result.write("../mix_result/" + to_string(counter) + ".obj");
        counter++;
    }
    return;
}
int main()
{
    //trimesh::TriMesh source = *trimesh::TriMesh::read("../15.obj");
    trimesh::TriMesh mask = *trimesh::TriMesh::read("../basis.obj");
    //Aligning_non_toplogic(source,t,"15.obj");
    //return 0;
    //trimesh::TriMesh target = *trimesh::TriMesh::read("../B/Cheeks_Inhaled_Left_aligned.obj");
    //mix_by_step(source,target);
    //return 0;

    //TestCases::testBlendshapeGeneration();
    //return 0;
    //omp_set_num_threads(4);
    //trimesh::TriMesh source = *trimesh::TriMesh::read("../lzj01.OBJ");
    //trimesh::TriMesh target = *trimesh::TriMesh::read("../B/Mesh_aligned.obj");
    //Aligning_non_toplogic(source,target,"WARPPING.obj");
    //trimesh::TriMesh target = *trimesh::TriMesh::read("../Mesh_aligned.obj");
    //Aligning_non_toplogic(source,target,"lzj01.obj");
    //auto_matching_faces(source,10,8);
    //auto_matching_faces_by_color(source,10);
    //return 0;
    vector<string> blendshapenames;
    string blendshape_path = "../wy_blendshapes/";
    string source_path = "../FacialWarpping/";
    string mesh_path = "../result/";
    string file_path = "../result/";
    string MaskFilename = "MeshMask.ply";
    std::vector<double> weights(51);
    get_files_list(blendshapenames,blendshape_path);
    //trimesh::TriMesh base = *trimesh::TriMesh::read("../Warpping/Mesh_aligned.obj");
    int index = 1;
    const int nshapes = 52;
    BlendshapeGenerator bg;

    std::vector<trimesh::TriMesh> B(nshapes);
    for(int i=0; i<nshapes; i++){
        B[i] = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        if(blendshapenames[i] == "basis.obj")
        {
            B[i] = B[0];
            B[0] = *trimesh::TriMesh::read(blendshape_path + "basis.obj");
            blendshapenames[i] = blendshapenames[0];
            blendshapenames[0] = "basis.obj";
        }
    }
    for(int i = 0;i<blendshapenames.size();i++)
    {
        trimesh::TriMesh b = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        Aligning_non_toplogic(b,B[0],blendshape_path + blendshapenames[i]);
    }
    return 0;

    std::vector<trimesh::TriMesh> dB(nshapes-1);
    for(int i=0; i<nshapes-1; i++){
        dB[i] = B[i+1];
        for(int j=0; j<dB[i].vertices.size(); j++){
            dB[i].vertices[j] -= B[0].vertices[j];
        }
    }
    while(index < 6)
    {
        ifstream in("../pose_weights/0000000"+ to_string(index)+".txt");
        unordered_map <string,double> blendshape_weights;
        string line;
        if ( in.fail() )
        {
            cout << "open file error" <<endl;
            return -1;
        }
        double total_weights = 0;
        while(getline (in, line))
        {
            cout << line <<endl ;
            for(int i = 0;i<line.size();i++)
            {
                if(line[i] == ':')
                {
                    blendshape_weights[line.substr(0,i-1)] = atof(line.substr(i+1,line.size()).c_str());
                }
            }
        }
        in.close();
        trimesh::TriMesh result = B[0];
        for(int i=0; i<nshapes-1; i++){
            cout << blendshapenames[i] << endl;
            cout << blendshape_weights[blendshapenames[i]] << endl;
        }
        for(int i = 0;i<dB.size();i++)
        {
            double dbw = blendshape_weights[blendshapenames[i+1]];
            for(int t = 0;t<dB[i].vertices.size();t++)
            {
                result.vertices[t] += dB[i].vertices[t]*dbw;
            }
        }
        result.write("../ebfr_pose/pose_"+ to_string(index)+".obj");
        //return 0;
        //cout << "Total: " << total_weights << endl;
        ofstream data("../ebfr_weights/00000000"+to_string(index)+".txt");
        for(int i = 0;i<weights.size();i++)
        {
            data << blendshapenames[i+1];
            data << " : ";
            data << blendshape_weights[blendshapenames[i+1]] << setprecision(2);
            data << endl;
        }
        index += 1;
    }
    return 0;
    /*
    for(int i = 0;i<blendshapenames.size();i++)
    {
        trimesh::TriMesh b = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        Aligning_non_toplogic(b,base,blendshape_path + blendshapenames[i]);
    }
    */

    //trimesh::TriMesh mask = *trimesh::TriMesh::read(source_path + MaskFilename);
    // load the template blendshapes and groundtruth blendshapes
    for(int i=0; i<nshapes; i++){
        B[i] = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        if(blendshapenames[i] == "Mesh_aligned.obj")
        {
            B[i] = B[0];
            B[0] = *trimesh::TriMesh::read(blendshape_path + "Mesh_aligned.obj");
            blendshapenames[i] = blendshapenames[0];
            blendshapenames[0] = "Mesh_aligned.obj";
        }
    }

    //std::vector<trimesh::TriMesh> dB(nshapes-1);
    for(int i=0; i<nshapes-1; i++){
        dB[i] = B[i+1];
        for(int j=0; j<dB[i].vertices.size(); j++){
            dB[i].vertices[j] -= B[0].vertices[j];
        }
    }


    const int frames = 3000;
    for(int i = 1;i<=frames;i++)
    {
        string framename = std::to_string(i) + ".obj";
        while(framename.size() < 12) framename = "0" + framename;
        string weightfile = std::to_string(i) + ".txt";
        while(weightfile.size() < 12) weightfile = "0" + weightfile;
        framename = "Frame" + framename;
        trimesh::TriMesh S = *trimesh::TriMesh::read(source_path + framename);
        Aligning_non_toplogic(S,B[0],framename);
        S = *trimesh::TriMesh::read(mesh_path + framename);
        //AligningBlendshapes(S,B[0]);
        //
        std::vector<double> w0_vec(nshapes-1, 0.0), wp_vec(nshapes-1, 0.0);

        weights = bg.estimateWeightsMask(S, B[0], dB, w0_vec, wp_vec, mask, true);
        // = bg.estimateWeightsMask(S, B[0], dB, w0_vec, wp_vec, mask, true);
        for(int i = 0;i<weights.size();i++)
        {
            cout << blendshapenames[i+1] << ":" << weights[i] << endl;
            w0_vec[i] = weights[i];
            wp_vec[i] = weights[i];
        }
        ofstream data(file_path + weightfile);
        for(int i = 0;i<weights.size();i++)
        {
            data << blendshapenames[i+1];
            data << " : ";
            data << weights[i];
            data << endl;
        }
        trimesh::TriMesh ret = bg.reconstructMesh(weights, B[0], dB);
        ret.write(file_path + framename);

        std::cout<< framename + " Done" <<std::endl;
    }
    /*
    string new_blendshape_path = "../newblendshapes/";
    trimesh::TriMesh base = *trimesh::TriMesh::read(blendshape_path + "Mesh_aligned.obj");
    for(int i=1; i<nshapes; i++){
        //AligningBlendshapes(B[i],B[0]);
        trimesh::TriMesh Bi = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        Aligning_non_toplogic(Bi,base,new_blendshape_path + blendshapenames[i],blendshapenames[i]);
        //Bi.write(blendshape_path +"Aligned_"+ blendshapenames[i]);
    }
    base.write(new_blendshape_path + "Mesh_aligned.obj");
    return 0;

    return 0;
    /*
    vector<string> blendshapenames;
    string blendshape_path = "../Zws/";
    string source_path = "../S/";
    string mesh_path = "../result/";
    string file_path = "../result/";
    string MaskFilename = "emily-mask-23686.ply";
    std::vector<double> weights(157);
    get_files_list(blendshapenames,blendshape_path);
    ifstream in("../result_weights/00000013.txt");
    unordered_map <string,double> blendshape_weights;
    string line;
    if ( in.fail() )
    {
        cout << "open file error" <<endl;
        return -1;
    }
    while(getline (in, line))
    {
        cout << line <<endl ;
        for(int i = 0;i<line.size();i++)
        {
            if(line[i] == ':')
            {
                //cout << line.substr(0,i-1) << endl;
                //string w = line.substr(i+1,line.size());
                //cout << w << endl;
                //float temp = atof(w.c_str());
                //cout << temp << endl;
                blendshape_weights[line.substr(0,i-1)] = atof(line.substr(i+1,line.size()).c_str());
            }
        }
    }
    in.close();
    */


    //trimesh::TriMesh base = *trimesh::TriMesh::read("../Fat_aligned.obj");
    //trimesh::TriMesh mix1 = *trimesh::TriMesh::read("../Eyes_Opened_Max_Left_aligned.obj");
    //trimesh::TriMesh mix2 = *trimesh::TriMesh::read("../Eyes_Opened_Max_Right_aligned.obj");

    //trimesh::TriMesh src = *trimesh::TriMesh::read("../Fat_right_aligned.obj");
    //Aligning_non_toplogic(src,base);
    //src = mix_mesh(base,src,1.2);
    //matching_faces(base,src,"fat_right",9887,7750);
    //AligningBlendshapes(src,target);
    //Aligning_non_toplogic(src,target);
    //return 0;
    //src = mix_mesh(no_edge,src,1.2);
    //matching_faces(no_edge, src, "orc_left_inhealed" , 6654, 10247);
    //matching_faces(no_edge, src, "orc_right_inhealed" , 1002, 4735);
    //return 0;
    //Aligning_non_toplogic(src,no_edge);
    //return 0;
    //matching_faces(no_edge, src, "bahe_left_inhealed" , 3691, 2024);
    //matching_faces(no_edge, src, "bahe_right_inhealed" , 12444, 9139);
    //matching_faces(no_edge,src,"female",9114,9138);
    //matching_faces(no_edge,src,"lzj_nose",1963,2245);
    //Aligning_non_toplogic(source,target);

    //female01:
    //left eye 11337 right eye 5936 center 2454
    //return 0;
    /*

	BlendshapeGenerator bg;
	const int nshapes = 158;
    std::vector<trimesh::TriMesh> B(nshapes);
    trimesh::TriMesh mask = *trimesh::TriMesh::read(source_path + MaskFilename);
    // load the template blendshapes and groundtruth blendshapes
    for(int i=0; i<nshapes; i++){
        B[i] = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        if(blendshapenames[i] == "Mesh_aligned.obj")
        {
            B[i] = B[0];
            B[0] = *trimesh::TriMesh::read(blendshape_path + "Mesh_aligned.obj");
            blendshapenames[i] = blendshapenames[0];
            blendshapenames[0] = "Mesh_aligned.obj";
        }
    }
    /*
    string new_blendshape_path = "../newblendshapes/";
    trimesh::TriMesh base = *trimesh::TriMesh::read(blendshape_path + "Mesh_aligned.obj");
    for(int i=1; i<nshapes; i++){
        //AligningBlendshapes(B[i],B[0]);
        trimesh::TriMesh Bi = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        Aligning_non_toplogic(Bi,base,new_blendshape_path + blendshapenames[i],blendshapenames[i]);
        //Bi.write(blendshape_path +"Aligned_"+ blendshapenames[i]);
    }
    base.write(new_blendshape_path + "Mesh_aligned.obj");
    return 0;
    */
    /*
    for(int i=1; i<nshapes; i++){
        B[i] = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
    }
    */
    /*
    for(int i=0; i<nshapes; i++){
        trimesh::TriMesh temp = *trimesh::TriMesh::read(blendshape_path + blendshapenames[i]);
        AligningBlendshapes(temp,B[0]);
        temp.write(blendshape_path + "aligned_" + blendshapenames[i]);
        cout << blendshapenames[i] << endl;
    }
    return 0;
    */
    /*

    trimesh::TriMesh result = B[0];
    for(int i=0; i<nshapes-1; i++){
        cout << blendshapenames[i] << endl;
        cout << blendshape_weights[blendshapenames[i]] << endl;
    }
    for(int i = 0;i<dB.size();i++)
    {
        double dbw = blendshape_weights[blendshapenames[i+1]];
        for(int t = 0;t<dB[i].vertices.size();t++)
        {
            result.vertices[t] += dB[i].vertices[t]*dbw;
        }
    }
    result.write("../final_result/transfered.obj");
    return 0;




    const int frames = 10;

    //w0_vec[0] = 0.0;


    for(int i = 1;i<=frames;i++)
    {
        string framename = std::to_string(i) + ".obj";
        while(framename.size() < 12) framename = "0" + framename;
        string weightfile = std::to_string(i) + ".txt";
        while(weightfile.size() < 12) weightfile = "0" + weightfile;
        trimesh::TriMesh S = *trimesh::TriMesh::read(source_path + framename);

        AligningBlendshapes(S,B[0]);
        //
        weights = bg.estimateWeightsMask(S, B[0], dB, w0_vec, wp_vec, mask, true);
        // = bg.estimateWeightsMask(S, B[0], dB, w0_vec, wp_vec, mask, true);
        for(int i = 0;i<weights.size();i++)
        {
            cout << blendshapenames[i+1] << ":" << weights[i] << endl;
            w0_vec[i] = weights[i];
            wp_vec[i] = weights[i];
        }
        ofstream data(file_path + weightfile);
        for(int i = 0;i<weights.size();i++)
        {
            data << blendshapenames[i+1];
            data << " : ";
            data << weights[i];
            data << endl;
        }
        trimesh::TriMesh ret = bg.reconstructMesh(weights, B[0], dB);
        ret.write(file_path + framename);

        std::cout<< framename + " Done" <<std::endl;
    }
    */
}
