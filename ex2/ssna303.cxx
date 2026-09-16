/*!
 * \file   ssna3030.cxx
 * \brief
 * \author Thomas Helfer, Guillaume Latu
 * \date   06/04/2021
 */

#include <memory>
#include <cstdlib>
#include <iostream>
#include "mfem/general/optparser.hpp"
#include "MFEMMGIS/Material.hxx"
#include "MFEMMGIS/NonLinearEvolutionProblem.hxx"

int main(int argc, char** argv) {
  using namespace mfem_mgis;
  constexpr const auto dim = size_type{2};
  //
  auto ctx = Context{};
  auto or_die = ctx.getFatalFailureHandler();
  // Initialize mfem_mgis (it includes a call to MPI_Init)
  initialize(argc, argv);

  const char* mesh_file = "ssna303.msh";
  const char* behaviour = "Plasticity";
  const char* library = "src/libBehaviour.so";
  bool use_fbar = false;
#if defined(MFEM_USE_MUMPS) && defined(MFEM_USE_MPI)
  bool parallel = true;
#else
  bool parallel = false;
#endif
  auto order = 1;
  // options treatment
  mfem::OptionsParser args(argc, argv);
  declareDefaultOptions(args);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&parallel, "-p", "--parallel", "-no-p", "--no-parallel",
                 "Perform parallel computations.");
#ifdef MGIS_HAVE_TFEL
  args.AddOption(&use_fbar, "", "--use-fbar", "", "--no-use-fbar",
                 "Use Fbar formulation.");
#endif /* MGIS_HAVE_TFEL */
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    abort(EXIT_FAILURE);
  }
  args.PrintOptions(std::cout);
  // the non linear problem
  NonLinearEvolutionProblem problem(
      ctx, dict{{"MeshFileName", mesh_file},
                {"FiniteElementFamily", "H1"},
                {"FiniteElementOrder", order},
                {"UnknownsSize", dim},
                {"Materials", dict{{"NotchedBeam", 1}}},
                {"Boundardies", dict{{"LowerBoundary", 3},
                                     {"SymmetryAxis", 4},
                                     {"UpperBoundary", 2}}},
                {"Hypothesis", "PlaneStrain"},
                {"Parallel", parallel}}); 
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
  problem.addBehaviourIntegrator(ctx, "Mechanics", "NotchedBeam", library,
                                 behaviour) |
      or_die;
#endif /* MGIS_HAVE_TFEL */
  // materials
  auto& m1 = problem.getMaterial(ctx, "NotchedBeam", 0) | or_die;
  mgis::behaviour::setExternalStateVariable(ctx, m1.s0, "Temperature", 293.15) |
      or_die;
  mgis::behaviour::setExternalStateVariable(ctx, m1.s1, "Temperature", 293.15) |
      or_die;
  // boundary conditions
  problem.addUniformDirichletBoundaryCondition(
      {{"Boundary", "LowerBoundary"}, {"Component", 1}});
  problem.addUniformDirichletBoundaryCondition(
      {{"Boundary", "SymmetryAxis"}, {"Component", 0}});
  problem.addUniformDirichletBoundaryCondition(
      {{"Boundary", "UpperBoundary"},
       {"Component", 1},
       {"LoadingEvolution", [](const auto t) {
          const auto u = 6e-3 * t;
          return u;
        }}});
  // solving the problem
  if (!usePETSc()) {
    problem.setPredictionPolicy(
        {.strategy =
             mfem_mgis::PredictionStrategy::BEGINNING_OF_TIME_STEP_PREDICTION});
    problem.setSolverParameters(ctx, {{"VerbosityLevel", 0},
                                      {"RelativeTolerance", 1e-6},
                                      {"AbsoluteTolerance", 0.},
                                      {"MaximumNumberOfIterations", 10}}) |
        or_die;
    // selection of the linear solver
    if (parallel) {
      problem.setLinearSolver(ctx, "MUMPSSolver", {}) | or_die;
    } else {
      problem.setLinearSolver(ctx, "UMFPackSolver", {}) | or_die;
    }
  }
  // post-processings
  problem.addPostProcessing(
      ctx, "ComputeResultantForceOnBoundary",
      {{"Boundary", 2}, {"OutputFileName", "force.txt"}}) |
      or_die;
  problem.addPostProcessing(ctx, "ParaviewExportResults",
                            {{"OutputFileName", "ssna303-displacements"}}) |
      or_die;
  problem.addPostProcessing(ctx, "ParaviewExportIntegrationPointResultsAtNodes",
                            {{{"Results", "FirstPiolaKirchhoffStress"},
                              {"OutputFileName", "ssna303-stress"}}}) |
      or_die;
  problem.addPostProcessing(
      ctx, "ParaviewExportIntegrationPointResultsAtNodes",
      {{{"Results", "EquivalentPlasticStrain"},
        {"OutputFileName", "ssna303-equivalent-plastic-strain"}}}) |
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
    auto nsubsteps = size_type{0};
    while (nsteps != 0) {
      auto converged = problem.solve(ctx, ct, dt2);
      if (converged) {
        --nsteps;
        ct += dt2;
        problem.update();
      } else {
        std::cout << "\nsubstep: " << nsubsteps << '\n';
        nsteps *= 2;
        dt2 /= 2;
        ++nsubsteps;
        problem.revert();
        if (nsubsteps == 10) {
          raise("maximum number of substeps");
        }
      }
    }
    problem.executePostProcessings(ctx, t, dt);
    t += dt;
    ++iteration;
    std::cout << '\n';
  }
  return EXIT_SUCCESS;
}
