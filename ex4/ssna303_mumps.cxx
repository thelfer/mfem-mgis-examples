/*!
 * \file   ssna303.cxx
 * \brief
 * \author Thomas Helfer
 * \date   14/12/2020
 */

#include <memory>
#include <cstdlib>
#include <iostream>
#include "mfem/general/optparser.hpp"
#include "mfem/linalg/solvers.hpp"
#include "mfem/linalg/hypre.hpp"

#ifdef MFEM_USE_PETSC
#include "mfem/linalg/petsc.hpp"
#endif /* MFEM_USE_PETSC */

#ifdef MFEM_USE_MUMPS
#include "mfem/linalg/mumps.hpp"
#endif /* MFEM_USE_MUMPS */

#include "mfem/fem/datacollection.hpp"
#include "MGIS/Raise.hxx"
#include "MFEMMGIS/Material.hxx"
#include "MFEMMGIS/Profiler.hxx"
#include "MFEMMGIS/UniformDirichletBoundaryCondition.hxx"
#include "MFEMMGIS/NonLinearEvolutionProblem.hxx"
#include "MFEMMGIS/NonLinearEvolutionProblemImplementation.hxx"
#include "MFEMMGIS/LinearSolverFactory.hxx"

#define PRINT_DEBUG (std::cout << __FILE__ << ":" << __LINE__ << std::endl)

int main(int argc, char** argv) {
  using namespace mfem_mgis;
  //
  auto ctx = Context{};
  auto or_die = ctx.getFatalFailureHandler();
  // ctx.enableProfiling(true);
  initialize(argc, argv);
  constexpr const auto dim = size_type{3};
  const char* mesh_file = "ssna303_3d.msh";
  const char* behaviour = "FiniteStrainPlasticity";
  const char* library = "src/libBehaviour.so";
  const char* petscrc_file = "";
  bool use_fbar = false;
  auto parallel = int{1};
  auto ref_para = 0;
  auto ref_seq = 0;
  auto order = 1;

  // options treatment
  mfem::OptionsParser args(argc, argv);
  declareDefaultOptions(args);
  args.AddOption(&parallel, "-p", "--parallel",
                 "Perform parallel computations.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
#ifdef MGIS_HAVE_TFEL
  args.AddOption(&use_fbar, "", "--use-fbar", "", "--no-use-fbar",
                 "Use Fbar formulation.");
#endif /* MGIS_HAVE_TFEL */
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return EXIT_FAILURE;
  }
  args.PrintOptions(std::cout);

  // loading the mesh
  NonLinearEvolutionProblem problem(
      ctx, {{"MeshFileName", mesh_file},
            {"Materials", dict{{"NotchedBeam", 1}}},
            {"FiniteElementFamily", "H1"},
            {"FiniteElementOrder", order},
            {"UnknownsSize", dim},
            {"Hypothesis", "Tridimensional"},
            {"Parallel", true},
            {"NumberOfUniformRefinements", parallel ? ref_para : ref_seq}});

  // 2 1 "Volume"
#ifdef MGIS_HAVE_TFEL
  if (use_fbar) {
    problem.addBehaviourIntegrator(
        ctx, "Mechanics", "NotchedBeam", library, behaviour,
        {{"Regularization", dict{{"FBar", dict{}}}}}) |
        or_die;
  } else {
    problem.addBehaviourIntegrator(ctx, "Mechanics", "NotchedBeam", library,
                                   behaviour) |
        or_die;
  }
#else /* MGIS_HAVE_TFEL */
  problem.addBehaviourIntegrator("Mechanics", "NotchedBeam", library,
                                 behaviour);
#endif /* MGIS_HAVE_TFEL */
  // materials
  auto& m1 = problem.getMaterial(ctx, "NotchedBeam", 0) | or_die;
  mgis::behaviour::setExternalStateVariable(m1.s0, "Temperature", 293.15);
  mgis::behaviour::setExternalStateVariable(m1.s1, "Temperature", 293.15);
  // boundary conditions

  // 3 LowerBoundary
  problem.addBoundaryCondition(
      ctx, std::make_unique<UniformDirichletBoundaryCondition>(
               problem.getFiniteElementDiscretizationPointer(), 3, 1)) |
      or_die;
  // 4 SymmetryPlane1
  problem.addBoundaryCondition(
      ctx, std::make_unique<UniformDirichletBoundaryCondition>(
          problem.getFiniteElementDiscretizationPointer(), 4, 0)) |
      or_die;
  // 5 SymmetryPlane2
  problem.addBoundaryCondition(
      ctx, std::make_unique<UniformDirichletBoundaryCondition>(
               problem.getFiniteElementDiscretizationPointer(), 5, 2)) |
      or_die;
  // 2 UpperBoundary
  problem.addBoundaryCondition(
      ctx, std::make_unique<UniformDirichletBoundaryCondition>(
               problem.getFiniteElementDiscretizationPointer(), 2, 1,
               [](const auto t) {
                 const auto u = 6e-3 * t;
                 return u;
               })) |
      or_die;

  // solving the problem without petsc
  if (!usePETSc()) {
    problem.setPredictionPolicy(
        {.strategy =
             mfem_mgis::PredictionStrategy::BEGINNING_OF_TIME_STEP_PREDICTION});
    problem.setSolverParameters(ctx, {{"VerbosityLevel", 2},
                                      {"RelativeTolerance", 1e-6},
                                      {"AbsoluteTolerance", 0.},
                                      {"MaximumNumberOfIterations", 10}}) |
        or_die;
    if (parallel) {
      problem.setLinearSolver(ctx, "MUMPSSolver", {}) | or_die;
    } else {
      problem.setLinearSolver(ctx, "UMFPackSolver", {}) | or_die;
    }
  }

  // vtk export
  problem.addPostProcessing(
      ctx, "ParaviewExportResults",
      {{"OutputFileName", std::string("ssna303-displacements")}}) |
      or_die;
  problem.addPostProcessing(
      ctx, "ComputeResultantForceOnBoundary",
      {{"Boundary", 2}, {"OutputFileName", "force.txt"}}) |
      or_die;

  // loop over time step
  const auto nsteps = size_type{50};
  const auto dt = real{1} / nsteps;
  auto t = real{0};
  auto iteration = size_type{};
  for (size_type i = 0; i != nsteps; ++i) {
    std::cout << "iteration " << iteration << " from " << t << " to " << t + dt
              << '\n';
    // resolution
    auto ct = t;
    auto dt2 = dt;
    auto nsteps = size_type{1};
    auto niter = size_type{0};
    while (nsteps != 0) {
      bool converged = problem.solve(ctx, ct, dt2);
      if (converged) {
        --nsteps;
        ct += dt2;
        problem.update();
      } else {
        std::cout << "\nsubstep: " << niter << '\n';
        nsteps *= 2;
        dt2 /= 2;
        ++niter;
        problem.revert();
        if (niter == 10) {
          mgis::raise("maximum number of substeps");
        }
      }
    }
    problem.executePostProcessings(ctx, t, dt);
    t += dt;
    ++iteration;
    std::cout << '\n';
  }
  // Profiler::OutputManager::printTimeTable(ctx);
  return EXIT_SUCCESS;
}
