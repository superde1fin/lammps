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
#include "input.h"
#include "update.h"
#include "variable.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RegSlab::RegSlab(LAMMPS *lmp, int narg, char **arg) :
        Region(lmp, narg, arg), point(nullptr), xstr(nullptr), ystr(nullptr), zstr(nullptr),
        s1xstr(nullptr), s1ystr(nullptr), s1zstr(nullptr), s2xstr(nullptr), s2ystr(nullptr),
        s2zstr(nullptr)
{
  xvar = yvar = zvar = s1xvar = s1yvar = s1zvar = s2xvar = s2yvar = s2zvar = 0.0;

  options(narg - 14, &arg[14]);

  full_volume = 0;


  int iarg = 2;

  while (iarg < 14){
    if (!strcmp(arg[iarg], "center")){
      if (utils::strmatch(arg[++iarg], "^v_")) {
        xstr = utils::strdup(arg[iarg] + 2);
        xp = 0;
        xstyle = VARIABLE;
        varshape = 1;
      } else {
        xp = utils::numeric(FLERR, arg[iarg], false, lmp);
        xstyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        ystr = utils::strdup(arg[iarg] + 2);
        yp = 0;
        ystyle = VARIABLE;
        varshape = 1;
      } else {
        yp = utils::numeric(FLERR, arg[iarg], false, lmp);
        ystyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        zstr = utils::strdup(arg[iarg] + 2);
        zp = 0;
        zstyle = VARIABLE;
        varshape = 1;
      } else {
        zp = utils::numeric(FLERR, arg[iarg], false, lmp);
        zstyle = CONSTANT;
      }
    } else if (!strcmp(arg[iarg], "side1")){
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s1xstr = utils::strdup(arg[iarg] + 2);
        x_vec[0] = 0;
        s1xstyle = VARIABLE;
        varshape = 1;
      } else {
        x_vec[0] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s1xstyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s1ystr = utils::strdup(arg[iarg] + 2);
        x_vec[1] = 0;
        s1ystyle = VARIABLE;
        varshape = 1;
      } else {
        x_vec[1] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s1ystyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s1zstr = utils::strdup(arg[iarg] + 2);
        x_vec[2] = 0;
        s1zstyle = VARIABLE;
        varshape = 1;
      } else {
        x_vec[2] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s1zstyle = CONSTANT;
      }
    } else if (!strcmp(arg[iarg], "side2")){
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s2xstr = utils::strdup(arg[iarg] + 2);
        y_vec[0] = 0;
        s2xstyle = VARIABLE;
        varshape = 1;
      } else {
        y_vec[0] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s2xstyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s2ystr = utils::strdup(arg[iarg] + 2);
        y_vec[1] = 0;
        s2ystyle = VARIABLE;
        varshape = 1;
      } else {
        y_vec[1] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s2ystyle = CONSTANT;
      }
      if (utils::strmatch(arg[++iarg], "^v_")) {
        s2zstr = utils::strdup(arg[iarg] + 2);
        y_vec[2] = 0;
        s2zstyle = VARIABLE;
        varshape = 1;
      } else {
        y_vec[2] = utils::numeric(FLERR, arg[iarg], false, lmp);
        s2zstyle = CONSTANT;
      }
    } else {
      error->all(FLERR, "Illegal region slab keyword {}", arg[iarg]);
    }
    iarg++;
  }

  if (varshape) {
    variable_check();
    RegSlab::shape_update();
  } else {
    setup_vectors();
  }


  // Slab has no bounding box
  bboxflag = 0;
  cmax = 1;
  contact = new Contact[cmax];
  tmax = 1;
}

void RegSlab::setup_vectors()
{
  //Check that input vectors are orthogonal
  if (dot(x_vec, y_vec)){
    error->all(FLERR, "Vectors defining region slab have to be orthogonal");
  }

  // Calculate unit normal vector by crossing x_vec into y_vec and writing results into z_vec
  cross(x_vec, y_vec, z_vec);

  int norm;
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
}

void RegSlab::shape_update()
{
  if (xstyle == VARIABLE) xp = input->variable->compute_equal(xvar);
  if (ystyle == VARIABLE) yp = input->variable->compute_equal(yvar);
  if (zstyle == VARIABLE) zp = input->variable->compute_equal(zvar);

  if (s1xstyle == VARIABLE) x_vec[0] = input->variable->compute_equal(s1xvar);
  if (s1ystyle == VARIABLE) x_vec[1] = input->variable->compute_equal(s1yvar);
  if (s1zstyle == VARIABLE) x_vec[2] = input->variable->compute_equal(s1zvar);

  if (s2xstyle == VARIABLE) y_vec[0] = input->variable->compute_equal(s2xvar);
  if (s2ystyle == VARIABLE) y_vec[1] = input->variable->compute_equal(s2yvar);
  if (s2zstyle == VARIABLE) y_vec[2] = input->variable->compute_equal(s2zvar);

  setup_vectors();

}

void RegSlab::variable_check()
{

  if (xstyle == VARIABLE) {
    xvar = input->variable->find(xstr);
    if (xvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", xstr);
    if (!input->variable->equalstyle(xvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", xstr);
  }

  if (ystyle == VARIABLE) {
    yvar = input->variable->find(ystr);
    if (yvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", ystr);
    if (!input->variable->equalstyle(yvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", ystr);
  }

  if (zstyle == VARIABLE) {
    zvar = input->variable->find(zstr);
    if (zvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", zstr);
    if (!input->variable->equalstyle(zvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", zstr);
  }

  if (s1xstyle == VARIABLE) {
    s1xvar = input->variable->find(s1xstr);
    if (s1xvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s1xstr);
    if (!input->variable->equalstyle(s1xvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s1xstr);
  }

  if (s1ystyle == VARIABLE) {
    s1yvar = input->variable->find(s1ystr);
    if (s1yvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s1ystr);
    if (!input->variable->equalstyle(s1yvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s1ystr);
  }

  if (s1zstyle == VARIABLE) {
    s1zvar = input->variable->find(s1zstr);
    if (s1zvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s1zstr);
    if (!input->variable->equalstyle(s1zvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s1zstr);
  }

  if (s2xstyle == VARIABLE) {
    s2xvar = input->variable->find(s2xstr);
    if (s2xvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s2xstr);
    if (!input->variable->equalstyle(s2xvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s2xstr);
  }

  if (s2ystyle == VARIABLE) {
    s2yvar = input->variable->find(s2ystr);
    if (s2yvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s2ystr);
    if (!input->variable->equalstyle(s2yvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s2ystr);
  }

  if (s2zstyle == VARIABLE) {
    s2zvar = input->variable->find(s2zstr);
    if (s2zvar < 0) error->all(FLERR, "Variable {} for region plane does not exist", s2zstr);
    if (!input->variable->equalstyle(s2zvar))
      error->all(FLERR, "Variable {} for region plane is invalid style", s2zstr);
  }
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

  delete[] xstr;
  delete[] ystr;
  delete[] zstr;

  delete[] s1xstr;
  delete[] s1ystr;
  delete[] s1zstr;

  delete[] s2xstr;
  delete[] s2ystr;
  delete[] s2zstr;
}

void RegSlab::init()
{
  Region::init();
  if (varshape) variable_check();
}


int RegSlab::inside(double x, double y, double z)
{
  point = coord_transform(x, y, z);

  return (point[2] >= 0) && (point[0] >= -side1/2) && (point[0] <= side1/2) && (point[1] >= -side2/2) && (point[1] <= side2/2);

}

int RegSlab::within(double x, double y, double z)
{
  point = coord_transform(x, y, z);

  return ((point[2] >= 0) && (point[0] >= -side1/2) && (point[0] <= side1/2) && (point[1] >= -side2/2) && (point[1] <= side2/2))
          &&
  ((point[2] < 0) && (point[0] >= -side1/2) && (point[0] <= side1/2) && (point[1] >= -side2/2) && (point[1] <= side2/2));

}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from normal side of plane
   no contact if on other side (possible if called from union/intersect)
   delxyz = vector from nearest projected point on plane to x
------------------------------------------------------------------------- */

int RegSlab::surface_interior(double *x, double cutoff)
{
  double dot = (x[0] - xp) * z_vec[0] + (x[1] - yp) * z_vec[1] + (x[2] - zp) * z_vec[2];
  if (dot < cutoff && dot >= 0.0 && within(x[0], x[1], x[2])) {
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
  if (dot < cutoff && dot >= 0.0 && within(x[0], x[1], x[2])) {
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

bool RegSlab::across_partial(double x1, double y1, double z1, double x2, double y2, double z2, double xpn, double ypn, double zpn){
  double origin_save[3] = {xp, yp, zp};
  xp = xpn;
  yp = ypn;
  zp = zpn;
  double* point1 = coord_transform(x1, y1, z1);
  double* point2 = coord_transform(x2, y2, z2);
  double t, x_int, y_int;
  bool res;
  xp = origin_save[0];
  yp = origin_save[1];
  zp = origin_save[2];
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

bool RegSlab::across_partial(double x1, double y1, double z1, double x2, double y2, double z2){
  if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp, zp)) return true; //13
  double x_side = domain->boxhi[0] - domain->boxlo[0];
  double y_side = domain->boxhi[1] - domain->boxlo[1];
  double z_side = domain->boxhi[2] - domain->boxlo[2];
  if (!domain->boundary[0][0] && !domain->boundary[1][0] && !domain->boundary[2][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp - y_side, zp)) return true; //11
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp)) return true; //14
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp)) return true; //16
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp + y_side, zp)) return true; //17
    if (across_partial(x1, y1, z1, x2, y2, z2, xp - x_side, yp - y_side, zp + z_side)) return true; //18
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp - y_side, zp + z_side)) return true; //19
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp - y_side, zp + z_side)) return true; //20
    if (across_partial(x1, y1, z1, x2, y2, z2, xp - x_side, yp, zp + z_side)) return true; //21
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp, zp + z_side)) return true; //22
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp + z_side)) return true; //23
    if (across_partial(x1, y1, z1, x2, y2, z2, xp - x_side, yp + y_side, zp + z_side)) return true; //24
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp + z_side)) return true; //25
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp + y_side, zp + z_side)) return true; //26
  } else if (!domain->boundary[0][0] && !domain->boundary[1][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp - y_side, zp)) return true; //11
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp)) return true; //14
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp)) return true; //16
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp + y_side, zp)) return true; //17
  } else if (!domain->boundary[0][0] && !domain->boundary[2][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp)) return true; //16
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp - y_side, zp + z_side)) return true; //19
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp, zp + z_side)) return true; //22
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp + z_side)) return true; //25
  } else if (!domain->boundary[1][0] && !domain->boundary[2][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp)) return true; //14
    if (across_partial(x1, y1, z1, x2, y2, z2, xp - x_side, yp, zp + z_side)) return true; //21
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp, zp + z_side)) return true; //22
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp + z_side)) return true; //23
  } else if (!domain->boundary[0][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp + y_side, zp)) return true; //16
  } else if (!domain->boundary[1][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp + x_side, yp, zp)) return true; //14
  } else if (!domain->boundary[2][0]) {
    if (across_partial(x1, y1, z1, x2, y2, z2, xp, yp, zp + z_side)) return true; //22
  }
  return false;
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


