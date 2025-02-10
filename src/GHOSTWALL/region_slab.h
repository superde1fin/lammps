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

#ifdef REGION_CLASS
// clang-format off
RegionStyle(slab,RegSlab);
// clang-format on
#else

#ifndef LMP_REGION_SLAB_H
#define LMP_REGION_SLAB_H

#include "region.h"

namespace LAMMPS_NS {

class RegSlab : public Region {
 public:
  RegSlab(class LAMMPS *, int, char **);
  ~RegSlab() override;
  int inside(double, double, double) override;
  int surface_interior(double *, double) override;
  int surface_exterior(double *, double) override;
  bool across_partial(double, double, double, double, double, double);
  double x_vec[3];
  double y_vec[3];
  double z_vec[3];

 protected:
  double xp, yp, zp, side1, side2;
  double * coord_transform(double, double, double);
  double *point;
  double get_norm(double *);
  double dot(double *, double *);
  void cross(double *, double *, double*);
};

}    // namespace LAMMPS_NS

#endif
#endif
