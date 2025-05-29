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
  void init() override;
  int inside(double, double, double) override;
  int surface_interior(double *, double) override;
  int surface_exterior(double *, double) override;

  int within(double, double, double);

  bool across_partial(double, double, double, double, double, double);
  bool across_partial(double, double, double, double, double, double, double, double, double);
  void shape_update() override;


 protected:
  double *coord_transform(double, double, double);
  double *point;
  double get_norm(double *);
  double dot(double *, double *);
  void cross(double *, double *, double*);
  void setup_vectors();


  //Center point of the slab
  double xp, yp, zp;
  int xstyle, xvar;
  int ystyle, yvar;
  int zstyle, zvar;
  char *xstr, *ystr, *zstr;

  //Vector defining first side of the slab
  double side1;
  double x_vec[3];
  int s1xstyle, s1xvar;
  int s1ystyle, s1yvar;
  int s1zstyle, s1zvar;
  char *s1xstr, *s1ystr, *s1zstr;

  //Vector defining second side of the slab
  double side2;
  double y_vec[3];
  int s2xstyle, s2xvar;
  int s2ystyle, s2yvar;
  int s2zstyle, s2zvar;
  char *s2xstr, *s2ystr, *s2zstr;

  //Normal vector to the slab plane
  double z_vec[3];

  void variable_check();

};

}    // namespace LAMMPS_NS

#endif
#endif
