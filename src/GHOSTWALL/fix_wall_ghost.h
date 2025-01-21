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
FixStyle(wall/ghost, FixWallGhost);
#else

#ifndef LMP_FIX_WALL_GHOST_H
#define LMP_FIX_WALL_GHOST_H

#include "fix.h"
#include <mpi.h>

namespace LAMMPS_NS {

class FixWallGhost : public Fix {
 public:
  int nwall; //Counter storing the index of a wall while parsing command args
  int wallwhich[6]; //Array with identifiers of wich side of the simulation region each wall refers to
  double coord0[6]; //Array storing an offset for a specified side for each wall
  int xstyle[6]; //Array storing type of coordinate specification: variable, constant or edge
  int xindex[6]; //Array storing variable identifiers for future lookup of wall position values
  char *xstr[6]; //Names of variables specifying position of each wall
  enum { NONE = 0, EDGE, CONSTANT, VARIABLE };

  FixWallGhost(class LAMMPS *, int, char **);
  ~FixWallGhost() override;
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

  virtual void wall_particle(int, double); //Function called to evaluate each wall's effect on the particles in the system

 protected:
  LAMMPS *lmp;
  double cutoff[6]; //Cutoffs for wall particle interactions for each wall
  int every[6]; //Array storing number of timesteps after which the wall neighborlists are reevaluated
  double xscale, yscale, zscale;
  int varflag;    // 1 if wall position is a variable
  int ilevel_respa;

  int inum; //Number of atoms in process neighbor list
  int *ilist; //Array of local ids stored in process neighbor list
  int *numneigh; //Array with number of neghbors for each local atom
  int **firstneigh; //2D array with neighbors of each local atom

  double **x; //2D array with atomic positions
  int *mask; //Atomic mask array
  bigint initial_timestep; //Timestep on which a run began
  bool *rel2wall[6]; //Array of pointers to boolean arrays that store information about the relative position of each atom (and its ghost atom copies) to the wall
  int get_ghost_offset(double *); //Returns a unique identifier of the location of the atom copy in the 3x3x3 ghost atom simulation region multicell
  bool filled_before; //Flag that tells whether the wall memory array has been filled before
  static void bitwise_or(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype); // Declare static
  MPI_Op mpi_bor_custom;


};

}    // namespace LAMMPS_NS

#endif
#endif
