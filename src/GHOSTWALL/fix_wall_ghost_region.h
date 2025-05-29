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
#include <memory>
#include <vector>

namespace LAMMPS_NS {

using RelMap = std::unordered_map<int, std::unordered_set<int>>;

class FixWallGhostRegion : public Fix {
 public:
  FixWallGhostRegion(class LAMMPS *, int, char **);
  ~FixWallGhostRegion() override;
  int setmask() override;
  void init() override;

  //Functions called before each run
  void setup_pre_force(int) override;

  //Functions called before or after each force evaluation
  void pre_force(int) override;
  void min_pre_force(int) override;

  //Called before each reneighbor
  void pre_neighbor() override;
  void min_pre_neighbor() override;

  void setup_post_neighbor() override;

  bool (FixWallGhostRegion::*check_across)(double, double, double, double, double, double);


 protected:
  LAMMPS *lmp;
  int ghost_dimensions;
  int rebuild_ts;
  int last_ts_built;

  bool filled_before; //Flag that tells whether the wall memory array has been filled before
  int from_restart;
  bool scanned_restart;
  bool has_parent_wall;

  int inum; //Number of atoms in process neighbor list
  int *ilist, *jlist; //Arrays of local ids stored in process neighbor list
  int *numneigh; //Array with number of neghbors for each local atom
  int **firstneigh; //2D array with neighbors of each local atom

  char *idregion, *idparent;
  Region *region;

  FixWallGhostRegion *parent_wall;

  double **x; //2D array with atomic positions
  int *mask; //Atomic mask array
  bigint initial_timestep; //Timestep on which a run began
  std::shared_ptr<RelMap> rel2wall;
  RelMap local_rel2wall;
  int get_ghost_offset(double *); //Returns a unique identifier of the location of the atom copy in the 3x3x3 ghost atom simulation region multicell

  bool across_region(double, double, double, double, double, double);
  bool across_partial_region(double, double, double, double, double, double);

  //restart functions
  int maxsize_restart();
  int size_restart(int);
  int pack_restart(int, double *);
  void unpack_restart(int, int);

  //Helper functions
  RelMap deserializeMap(const std::vector<int> &);
  void mergeMaps(std::shared_ptr<RelMap>, const RelMap &);
  std::vector<int> serializeMap(const std::shared_ptr<RelMap>);


};

}    // namespace LAMMPS_NS

#endif
#endif
