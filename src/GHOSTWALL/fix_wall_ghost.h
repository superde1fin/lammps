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

namespace LAMMPS_NS {

class FixWallGhost : public Fix {
 public:
  int nwall;
  int wallwhich[6];
  double coord0[6];
  int xstyle[6];
  int xindex[6];
  char *xstr[6];
  enum { NONE = 0, EDGE, CONSTANT, VARIABLE };

  FixWallGhost(class LAMMPS *, int, char **);
  ~FixWallGhost() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void min_setup(int) override;
  void pre_force(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  virtual void precompute(int){}
  virtual void wall_particle(int, int, double);

 protected:
  double cutoff[6];
  double xscale, yscale, zscale;
  int varflag;    // 1 if wall position is a variable
  int ilevel_respa;
};

}    // namespace LAMMPS_NS

#endif
#endif
