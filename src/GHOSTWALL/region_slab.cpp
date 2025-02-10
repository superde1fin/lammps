/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "region_slab.h"
#include "domain.h"

#include "error.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RegSlab::RegSlab(LAMMPS *lmp, int narg, char **arg) : Region(lmp, narg, arg), point(nullptr)
{
  options(narg - 14, &arg[14]);

  int fields_covered = 0;
  int iarg = 2;
  int norm;
  full_volume = 0;

  while (iarg < 14){
    if (!strcmp(arg[iarg], "center")){
      xp = utils::numeric(FLERR, arg[++iarg], false, lmp);
      yp = utils::numeric(FLERR, arg[++iarg], false, lmp);
      zp = utils::numeric(FLERR, arg[++iarg], false, lmp);
    } else if (!strcmp(arg[iarg], "side1")){
      x_vec[0] = utils::numeric(FLERR, arg[++iarg], false, lmp);
      x_vec[1] = utils::numeric(FLERR, arg[++iarg], false, lmp);
      x_vec[2] = utils::numeric(FLERR, arg[++iarg], false, lmp);
    } else if (!strcmp(arg[iarg], "side2")){
      y_vec[0] = utils::numeric(FLERR, arg[++iarg], false, lmp);
      y_vec[1] = utils::numeric(FLERR, arg[++iarg], false, lmp);
      y_vec[2] = utils::numeric(FLERR, arg[++iarg], false, lmp);
    } else {
      error->all(FLERR, "Illegal region slab keyword {}", arg[iarg]);
    }
    iarg++;
  }

  //Check that input vectors are orthogonal
  if (dot(x_vec, y_vec)){
    error->all(FLERR, "Vectors defining region slab have to be orthogonal");
  }

  // Calculate unit normal vector by crossing x_vec into y_vec and writing results into z_vec
  cross(x_vec, y_vec, z_vec);

  norm = get_norm(z_vec);
  z_vec[0] /= norm;
  z_vec[1] /= norm;
  z_vec[2] /= norm;


  //Calculate side lengths
  side1 = get_norm(x_vec);
  x_vec[0] /= side1;
  x_vec[1] /= side1;
  x_vec[2] /= side1;

  side2 = get_norm(y_vec);
  y_vec[0] /= side2;
  y_vec[1] /= side2;
  y_vec[2] /= side2;


  // Slab has no bounding box
  bboxflag = 0;
  cmax = 1;
  contact = new Contact[cmax];
  tmax = 1;
}

//Transform coordinates of any point into a rotated and shifted system of axis centered at the slab center, with x' and y' along slab edges and z' being normal to the plane
double *RegSlab::coord_transform(double x, double y, double z){
  double *res = new double[3];

  res[0] = (x - xp) * x_vec[0] + (y - yp) * x_vec[1] + (z - zp) * x_vec[2];
  res[1] = (x - xp) * y_vec[0] + (y - yp) * y_vec[1] + (z - zp) * y_vec[2];
  res[2] = (x - xp) * z_vec[0] + (y - yp) * z_vec[1] + (z - zp) * z_vec[2];

  return res;

}

/* ---------------------------------------------------------------------- */

RegSlab::~RegSlab()
{
  if (copymode) return;
  delete[] contact;
  delete[] point;
}


int RegSlab::inside(double x, double y, double z)
{
  point = coord_transform(x, y, z);

  if ((point[2] >= 0) && (point[0] >= -side1/2) && (point[0] <= side1/2) && (point[1] >= -side2/2) && (point[1] <= side2/2)) return 1;
  else if ((point[2] < 0) && (point[0] >= -side1/2) && (point[0] <= side1/2) && (point[1] >= -side2/2) && (point[1] <= side2/2)) return 0;
  else return -1;

}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from normal side of plane
   no contact if on other side (possible if called from union/intersect)
   delxyz = vector from nearest projected point on plane to x
------------------------------------------------------------------------- */

int RegSlab::surface_interior(double *x, double cutoff)
{
  double dot = (x[0] - xp) * z_vec[0] + (x[1] - yp) * z_vec[1] + (x[2] - zp) * z_vec[2];
  //printf("POSI: (%f, %f, %f) with dot %f and inside %d\n", x[0], x[1], x[2], dot, inside(x[0], x[1], x[2]));
  if (dot < cutoff && dot >= 0.0 && inside(x[0], x[1], x[2]) != -1) {
    contact[0].r = dot;
    contact[0].delx = dot * z_vec[0];
    contact[0].dely = dot * z_vec[1];
    contact[0].delz = dot * z_vec[2];
    contact[0].radius = 0;
    contact[0].iwall = 0;
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from non-z_vec side of plane
   no contact if on other side (possible if called from union/intersect)
   delxyz = vector from nearest projected point on plane to x
------------------------------------------------------------------------- */

int RegSlab::surface_exterior(double *x, double cutoff)
{
  double dot = (x[0] - xp) * z_vec[0] + (x[1] - yp) * z_vec[1] + (x[2] - zp) * z_vec[2];
  dot = -dot;
  //printf("POSE: (%f, %f, %f) with dot %f and inside %d\n", x[0], x[1], x[2], dot, inside(x[0], x[1], x[2]));
  if (dot < cutoff && dot >= 0.0 && inside(x[0], x[1], x[2]) != -1) {
    contact[0].r = dot;
    contact[0].delx = -dot * z_vec[0];
    contact[0].dely = -dot * z_vec[1];
    contact[0].delz = -dot * z_vec[2];
    contact[0].radius = 0;
    contact[0].iwall = 0;
    return 1;
  }
  return 0;
}

bool RegSlab::across_partial(double x1, double y1, double z1, double x2, double y2, double z2){
  double* point1 = coord_transform(x1, y1, z1);
  double* point2 = coord_transform(x2, y2, z2);
  double t, x_int, y_int; //Parameter of a line from tranformed p1 to transformed p2
  bool res;
  //printf("Two points (%f, %f, %f) and (%f, %f, %f) | (%f, %f, %f) and (%f, %f, %f)\n", x1, y1, z1, x2, y2, z2, point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]);
  if (point1[2] == point2[2]){
   res = 0; //Line from p1 to p2 is parallel to the slab plane
  } else {
    t = point1[2]/(point1[2] - point2[2]);
    if (t < 0 || t > 1) {
      res = 0; //Intersection of line from p1 to p2 lies outside of the segment between the points
    } else {
      x_int = point1[0] + t * (point2[0] - point1[0]);
      y_int = point1[1] + t * (point2[1] - point1[1]);
      res = (x_int >= -side1/2) && (x_int <= side1/2) && (y_int >= -side2/2) && (y_int <= side2/2);
    }
  }
  delete[] point1;
  delete[] point2;
  return res;
}

double RegSlab::get_norm(double* vec){
  return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

void RegSlab::cross(double* x_vec, double* y_vec, double* z_vec){
  z_vec[0] = x_vec[1] * y_vec[2] - x_vec[2] * y_vec[1];
  z_vec[1] = x_vec[2] * y_vec[0] - x_vec[0] * y_vec[2];
  z_vec[2] = x_vec[0] * y_vec[1] - x_vec[1] * y_vec[0];
}

double RegSlab::dot(double* x_vec, double* y_vec){
  return x_vec[0] * y_vec[0] + x_vec[1] * y_vec[1] + x_vec[2] * y_vec[2];
}
