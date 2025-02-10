/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(wall/ghost/region, FixWallGhostRegion);
#else

#ifndef LMP_FIX_WALL_GHOST_REGION_H
#define LMP_FIX_WALL_GHOST_REGION_H

#include "fix.h"
#include "region.h"
#include <unordered_map>
#include <unordered_set>

namespace LAMMPS_NS {

class FixWallGhostRegion : public Fix {
 public:
  FixWallGhostRegion(class LAMMPS *, int, char **);
  ~FixWallGhostRegion() override;
  int setmask() override;
  void init() override;
  //Functions called before each run
  void setup(int) override;
  void setup_pre_force(int) override;
  void min_setup(int) override;
  //Functions called before or after each force evaluation
  void pre_force(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  bool (FixWallGhostRegion::*check_across)(double, double, double, double, double, double);

 protected:
  LAMMPS *lmp;
  double cutoff; //Cutoff for wall particle interactions
  int every; //Number of timesteps after which the wall neighborlist is reevaluated
  int ilevel_respa;
  int append_flag;
  int ghost_dimensions;

  int inum; //Number of atoms in process neighbor list
  int *ilist; //Array of local ids stored in process neighbor list
  int *numneigh; //Array with number of neghbors for each local atom
  int **firstneigh; //2D array with neighbors of each local atom

  char *idregion;
  Region *region;

  double **x; //2D array with atomic positions
  int *mask; //Atomic mask array
  bigint initial_timestep; //Timestep on which a run began
  std::unordered_map<int, std::unordered_set<int>>* wall_save;
  std::unordered_map<int, std::unordered_set<int>> rel2wall;
  int get_ghost_offset(double *); //Returns a unique identifier of the location of the atom copy in the 3x3x3 ghost atom simulation region multicell
  bool filled_before; //Flag that tells whether the wall memory array has been filled before

  bool across_region(double, double, double, double, double, double);
  bool across_partial_region(double, double, double, double, double, double);


};

}    // namespace LAMMPS_NS

#endif
#endif
