// Stress harness for the planner's degenerate / edge-case regimes.
//
// WHY THIS EXISTS
// ---------------
// The release build defines NDEBUG, which compiles out every Eigen bounds assertion in the
// planner -- so an out-of-range .block()/.segment()/.col() does not fault, it silently
// corrupts the heap and the process dies later somewhere unrelated. One such bug
// (get_fast_cascaded_D_matrix overflowing its S_cascaded matrix whenever a trajectory was
// short enough to need only 3 segments) survived indefinitely because normal flight never
// produced a short enough path. It only appeared after the planning horizon was reduced.
//
// Build this WITHOUT -DNDEBUG so those assertions are live, and with sanitizers:
//
//   g++ -std=c++17 -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
//       tools/stress_planner.cpp $(ls src/*.cpp | grep -v python_bindings) \
//       -Iinclude -I/usr/include/eigen3 -I/usr/local/include/osqp \
//       -L/usr/local/lib -losqp -o stress_planner
//
// (osqp.h is included without an osqp/ prefix, hence .../include/osqp; python_bindings.cpp
// needs pybind11 and is excluded.)
//
// Run one case per process so a crash isolates rather than ending the sweep:
//   N=$(./stress_planner --count); for i in $(seq 0 $((N-1))); do ./stress_planner $i; done
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include "voxel_grid.hpp"
#include "trajectory_planner.hpp"

using namespace trajectory_planner;

namespace {

bool solid(const Eigen::Vector3d& p) {
  if (p.x()>=12&&p.x()<=12.5&&p.y()>=-3&&p.y()<=3&&p.z()>=-6&&p.z()<=0) return true;
  if (p.x()>=6&&p.x()<=7&&p.y()>=-1&&p.y()<=1&&p.z()>=-5&&p.z()<=-3) return true;
  double dx=p.x()-5, dy=p.y()-3;
  if (std::sqrt(dx*dx+dy*dy)<=0.5 && p.z()>=-8 && p.z()<=0) return true;
  dx=p.x()-8; dy=p.y()+2;
  if (std::sqrt(dx*dx+dy*dy)<=0.75 && p.z()>=-6 && p.z()<=0) return true;
  return false;
}

std::vector<Eigen::Vector3d> scan_at(const Eigen::Vector3d& p) {
  std::vector<Eigen::Vector3d> h;
  for (int iv=0; iv<20; ++iv) {
    double el=(-7+iv*3)*M_PI/180;
    for (int ih=0; ih<120; ++ih) {
      double az=(-180+ih*3)*M_PI/180;
      Eigen::Vector3d d(std::cos(el)*std::cos(az), std::cos(el)*std::sin(az), -std::sin(el));
      d.normalize();
      double t=70.0;
      for (double s=0.25; s<40; s+=0.25) {
        Eigen::Vector3d q=p+s*d;
        if (q.z()>=0.0 || solid(q)) { t=s; break; }
      }
      if (t>=0.1 && t<40) h.push_back(p+t*d);
    }
  }
  return h;
}

struct Case {
  std::string name;
  Eigen::Vector3d start, goal, vel, acc;
  double horizon;
  int max_sfc;
  int prior_scans;
};

std::vector<Case> build_cases() {
  std::vector<Case> c;
  const Eigen::Vector3d Z = Eigen::Vector3d::Zero();
  auto add=[&](std::string n, Eigen::Vector3d s, Eigen::Vector3d g, Eigen::Vector3d v,
               Eigen::Vector3d a, double h, int m, int ps){ c.push_back({n,s,g,v,a,h,m,ps}); };

  // --- degenerate goals -------------------------------------------------------------
  add("goal == start",            {0,0,-3}, {0,0,-3},      {0,0,0}, Z, 10, -1, 20);
  add("goal 0.05 m away",         {0,0,-3}, {0.05,0,-3},   {0,0,0}, Z, 10, -1, 20);
  add("goal 0.5 m away",          {0,0,-3}, {0.5,0,-3},    {1,0,0}, Z, 10, -1, 20);
  add("goal 1 m, moving fast",    {0,0,-3}, {1,0,-3},      {3,0,0}, Z, 10, -1, 20);
  add("goal 100 m away",          {0,0,-3}, {100,0,-3},    {1,0,0}, Z, 10, -1, 20);
  add("goal behind vehicle",      {0,0,-3}, {-8,0,-3},     {3,0,0}, Z, 10, -1, 20);
  add("goal inside a wall",       {0,0,-3}, {12.25,0,-3},  {1,0,0}, Z, 15, -1, 20);
  add("goal inside pillar",       {0,0,-3}, {5,3,-4},      {1,0,0}, Z, 10, -1, 20);
  add("goal below ground",        {0,0,-3}, {6,0,2.0},     {1,0,0}, Z, 10, -1, 20);
  add("purely vertical goal",     {0,0,-3}, {0,0,-8},      {0,0,0}, Z, 10, -1, 20);

  // --- degenerate starts ------------------------------------------------------------
  add("start inside obstacle",    {6.5,0,-4}, {14,0,-3},   {1,0,0}, Z, 10, -1, 20);
  add("start on the ground",      {0,0,-0.2}, {8,0,-3},    {0,0,0}, Z, 10, -1, 20);
  add("start above map ceiling",  {0,0,-14},  {8,0,-3},    {0,0,0}, Z, 10, -1, 20);

  // --- kinematic extremes -----------------------------------------------------------
  add("velocity at v_max",        {0,0,-3}, {12,0,-3},     {3,0,0},    Z, 10, -1, 20);
  add("velocity above v_max",     {0,0,-3}, {12,0,-3},     {6,0,0},    Z, 10, -1, 20);
  add("velocity backwards",       {0,0,-3}, {12,0,-3},     {-3,0,0},   Z, 10, -1, 20);
  add("large acceleration",       {0,0,-3}, {12,0,-3},     {2,0,0}, {0,0,-8}, 10, -1, 20);
  add("velocity straight down",   {0,0,-3}, {12,0,-3},     {0,0,3},    Z, 10, -1, 20);

  // --- horizon / corridor-count extremes --------------------------------------------
  for (double h : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0, 50.0})
    add("horizon " + std::to_string(h), {0,0,-3}, {30,0,-3}, {1,0,0}, Z, h, -1, 20);
  for (int m : {0, 1, 2, 3, 4, 8})
    add("max_sfc " + std::to_string(m), {0,0,-3}, {30,0,-3}, {1,0,0}, Z, 15, m, 20);

  // --- map states -------------------------------------------------------------------
  add("EMPTY map (no scans)",     {0,0,-3}, {10,0,-3},     {1,0,0}, Z, 10, -1, 0);
  add("single scan only",         {0,0,-3}, {10,0,-3},     {1,0,0}, Z, 10, -1, 1);
  add("heavily accumulated",      {0,0,-3}, {10,0,-3},     {1,0,0}, Z, 10, -1, 120);

  // --- short paths that still bend (the regime that found the snap-stencil bug) ------
  for (double sx = 3.0; sx <= 11.0; sx += 1.0)
    for (double gd : {1.0, 2.0, 3.0, 4.0})
      add("short bend x=" + std::to_string(sx) + " d=" + std::to_string(gd),
          {sx,0,-3}, {sx+gd,0.5,-3}, {0.5,0,0}, Z, 10, -1, 20);
  return c;
}

} // namespace

int main(int argc, char** argv) {
  auto cases = build_cases();
  if (argc > 1 && std::string(argv[1]) == "--count") {
    std::cout << cases.size() << "\n";
    return 0;
  }
  int idx = (argc > 1) ? std::atoi(argv[1]) : 0;
  if (idx < 0 || idx >= (int)cases.size()) { std::cerr << "bad index\n"; return 2; }
  const Case& k = cases[idx];

  const double VOX = 0.5, R = 0.4905;
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(VOX, R);
  for (int n = k.prior_scans; n > 0; --n) {
    Eigen::Vector3d p = k.start - Eigen::Vector3d(0.22 * n, 0, 0);
    grid->update_from_scan(p, scan_at(p));
    (void)grid->get_unknown_frontier();
  }
  if (k.prior_scans > 0) grid->update_from_scan(k.start, scan_at(k.start));

  FrontEndConfig cfg;
  cfg.drone_physical_radius = R;
  cfg.voxel_resolution = VOX;
  cfg.map_bounds = Eigen::Vector3d(100, 100, 15);
  cfg.sensor_range = 70.0;
  cfg.min_altitude = R + 0.1;
  cfg.max_sfc_count = k.max_sfc;
  cfg.planning_horizon = k.horizon;

  TrajectoryPlanner planner(cfg, grid, 3.0, 2.0);
  auto res = planner.plan_mission(k.start, k.goal, k.vel, k.acc);

  // Sanity-check anything we hand downstream: the manager evaluates this spline blindly.
  bool sane = true;
  if (res.success) {
    if (res.control_points.size() % 3 != 0) sane = false;
    if (res.control_points.size() / 3 <= res.degree) sane = false;
    if (res.knots.size() <= res.degree + 1) sane = false;
    if (!res.control_points.allFinite() || !res.knots.allFinite()) sane = false;
  }
  std::cout << "CASE " << idx << " [" << k.name << "] success=" << res.success
            << " corridors=" << res.corridors.size()
            << " pts=" << res.total_control_points
            << (sane ? "  OK" : "  *** INSANE RESULT ***") << "\n";
  return sane ? 0 : 3;
}
