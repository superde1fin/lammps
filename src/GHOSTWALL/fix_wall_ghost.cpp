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

#include "fix_wall_ghost.h"

#include "domain.h"
#include "error.h"
#include "input.h"
#include "lattice.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "variable.h"
#include "pair.h"
#include "force.h"
#include "atom.h"
#include "neigh_list.h"
#include "memory.h"


#include <cstring>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

enum { XLO = 0, XHI = 1, YLO = 2, YHI = 3, ZLO = 4, ZHI = 5 };

static const char *wallpos[] = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};

// Static or global bitwise OR function
void FixWallGhost::bitwise_or(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    unsigned char *in = static_cast<unsigned char *>(invec);
    unsigned char *inout = static_cast<unsigned char *>(inoutvec);

    for (int i = 0; i < *len; i++) {
        inout[i] |= in[i]; // Perform bitwise OR
    }
}

/* ---------------------------------------------------------------------- */

FixWallGhost::FixWallGhost(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), nwall(0), lmp(lmp)
{
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  dynamic_group_allow = 1;

  if (!domain->boundary[0][0] && !domain->boundary[1][0] && !domain->boundary[2][0]){
    ghost_dimensions = 13;
  } else if (!domain->boundary[0][0] && !domain->boundary[1][0] || !domain->boundary[0][0] && !domain->boundary[2][0] || !domain->boundary[1][0] && !domain->boundary[2][0]){
    ghost_dimensions = 4;
  } else if (!domain->boundary[0][0] || !domain->boundary[1][0] || !domain->boundary[2][0]){
    ghost_dimensions = 1;
  } else {
    ghost_dimensions = 0;
  }



  MPI_Op_create(&bitwise_or, 1, &mpi_bor_custom);

  for (int i = 0; i < modify->nfix; i++)
    printf("See fix: %s\n", modify->fix[i]->id);

  // parse args

  int scaleflag = 1;

  for (int i = 0; i < 6; i++) xstr[i] = nullptr;

  int iarg = 3, ifix;

  while (iarg < narg) {
    int wantargs = 4;

    if ((strcmp(arg[iarg], "xlo") == 0) || (strcmp(arg[iarg], "xhi") == 0) ||
        (strcmp(arg[iarg], "ylo") == 0) || (strcmp(arg[iarg], "yhi") == 0) ||
        (strcmp(arg[iarg], "zlo") == 0) || (strcmp(arg[iarg], "zhi") == 0)) {
      if (iarg + wantargs > narg) error->all(FLERR, "Missing argument for fix {} command", style);

      int newwall;
      if (strcmp(arg[iarg], "xlo") == 0) {
        newwall = XLO;
      } else if (strcmp(arg[iarg], "xhi") == 0) {
        newwall = XHI;
      } else if (strcmp(arg[iarg], "ylo") == 0) {
        newwall = YLO;
      } else if (strcmp(arg[iarg], "yhi") == 0) {
        newwall = YHI;
      } else if (strcmp(arg[iarg], "zlo") == 0) {
        newwall = ZLO;
      } else if (strcmp(arg[iarg], "zhi") == 0) {
        newwall = ZHI;
      }
      for (int m = 0; (m < nwall) && (m < 6); m++) {
        if (newwall == wallwhich[m])
          error->all(FLERR, "{} wall defined twice in fix {} command", wallpos[newwall], style);
      }
      wallwhich[nwall] = newwall;

      if (strcmp(arg[iarg + 1], "EDGE") == 0) {
        xstyle[nwall] = EDGE;
        int dim = wallwhich[nwall] / 2;
        int side = wallwhich[nwall] % 2;
        if (side == 0) {
          coord0[nwall] = domain->boxlo[dim];
        } else {
          coord0[nwall] = domain->boxhi[dim];
        }
      } else if (utils::strmatch(arg[iarg + 1], "^v_")) {
        xstyle[nwall] = VARIABLE;
        xstr[nwall] = utils::strdup(arg[iarg + 1] + 2);
      } else {
        xstyle[nwall] = CONSTANT;
        coord0[nwall] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      }

      cutoff[nwall] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      every[nwall] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      rel2wall[nwall] = nullptr;
      filled_before[nwall] = false;

      nwall++;
      iarg += wantargs;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix {} command", style);
      if (strcmp(arg[iarg + 1], "box") == 0)
        scaleflag = 0;
      else if (strcmp(arg[iarg + 1], "lattice") == 0)
        scaleflag = 1;
      else
        error->all(FLERR, "Illegal fix {} command", style);
      iarg += 2;
    } else if (strcmp(arg[iarg], "append") == 0) {
      for (ifix = 0; ifix < modify->nfix; ifix++){
        if(modify->fix[ifix] != nullptr){
          if(strcmp(arg[iarg + 1], modify->fix[ifix]->id) == 0) break;
		}
      }
      if (ifix < modify->nfix){
	    if (strcmp(modify->fix[ifix]->style, "wall/ghost") == 0){
          FixWallGhost *fixGW = dynamic_cast<FixWallGhost*>(modify->fix[ifix]);
          append_flag = 1;
          wall_save[nwall] = fixGW->rel2wall[fixGW->wallwhich[nwall]];
        } else {
          error->all(FLERR, "Initial fix has to be style wall/ghost and not {}", modify->fix[ifix]->style); 
        }
	  } else { 
        error->all(FLERR, "No fix with name {} was found", arg[iarg + 1]); 
      }
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix {} command", style);
  }


  // error checks

  if (nwall == 0) error->all(FLERR, "Illegal fix {} command: no walls defined", style);
  for (int m = 0; m < nwall; m++) {
    if (cutoff[m] <= 0.0)
      error->all(FLERR, "Fix {} cutoff <= 0.0 for {} wall", style, wallpos[wallwhich[m]]);
  }

  for (int m = 0; m < nwall; m++)
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->dimension == 2)
      error->all(FLERR, "Cannot use fix {} zlo/zhi for a 2d simulation", style);

  // scale factors for wall position for CONSTANT and VARIABLE walls

  int flag = 0;
  for (int m = 0; m < nwall; m++)
    if (xstyle[m] != EDGE) flag = 1;

  if (flag) {
    if (scaleflag) {
      xscale = domain->lattice->xlattice;
      yscale = domain->lattice->ylattice;
      zscale = domain->lattice->zlattice;
    } else
      xscale = yscale = zscale = 1.0;

    for (int m = 0; m < nwall; m++) {
      if (xstyle[m] != CONSTANT) continue;
      if (wallwhich[m] < YLO)
        coord0[m] *= xscale;
      else if (wallwhich[m] < ZLO)
        coord0[m] *= yscale;
      else
        coord0[m] *= zscale;
    }
  }

  // set varflag if any wall positions or parameters are variable

  varflag = 0;
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) varflag = 1;
  }

}

/* ---------------------------------------------------------------------- */

FixWallGhost::~FixWallGhost()
{
  if (copymode) return;

  for (int m = 0; m < nwall; m++) {
    delete[] xstr[m];
    xstr[m] = nullptr;
    if (rel2wall[m]){
      lmp->memory->destroy(rel2wall[m]);
      rel2wall[m] = nullptr;
    }
  }
  MPI_Op_free(&mpi_bor_custom);
}

/* ---------------------------------------------------------------------- */

int FixWallGhost::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;

  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::init()
{
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) {
      xindex[m] = input->variable->find(xstr[m]);
      if (xindex[m] < 0) error->all(FLERR, "Variable name for fix wall does not exist");
      if (!input->variable->equalstyle(xindex[m]))
        error->all(FLERR, "Only variables of style \"equal\" are allowed");
    }
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::setup(int vflag)
{
  initial_timestep = update->ntimestep;

  if (!utils::strmatch(update->integrate_style, "^verlet")) {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::min_setup(int vflag)
{
  post_force(vflag);
}


void FixWallGhost::pre_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::post_force(int vflag)
{
  // virial setup

  v_init(vflag);


  // coord = current position of wall
  // evaluate variables if necessary, wrap with clear/add
  // for epsilon/sigma variables need to re-invoke precompute()

  if (varflag) modify->clearstep_compute();

  double coord;
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) {
      coord = input->variable->compute_equal(xindex[m]);
      if (wallwhich[m] < YLO)
        coord *= xscale;
      else if (wallwhich[m] < ZLO)
        coord *= yscale;
      else
        coord *= zscale;
    } else
      coord = coord0[m];
    wall_particle(m, coord);
  }

  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallGhost::min_post_force(int vflag)
{
  post_force(vflag);
}


void FixWallGhost::set_wall_memory(int m, int dim, double coord){
  int all_atoms_count = (ghost_dimensions + 1)*atom->natoms; //Ensure all ghost atoms are included as well
  int i, rel_ind;
  
  if (rel2wall[m] == nullptr) lmp->memory->create(rel2wall[m], all_atoms_count, "ghostwall:rel2wall");

  std::fill(rel2wall[m], rel2wall[m] + all_atoms_count, false);

  for (i = 0; i < atom->nlocal; i++){
      if (x[i][0] > 2*domain->boxhi[0] - domain->boxlo[0] || x[i][0] < 2*domain->boxlo[0] - domain->boxhi[0] || x[i][1] > 2*domain->boxhi[0] - domain->boxlo[1] || x[i][1] < 2*domain->boxlo[1] - domain->boxhi[1] || x[i][2] > 2*domain->boxhi[2] - domain->boxlo[2] || x[i][2] < 2*domain->boxlo[2] - domain->boxhi[2]){
      error->all(FLERR, "Fix style wall/ghost/region does not support the use of second set of ghost atoms. Reduce your cutoff");
    }
	rel_ind =  (ghost_dimensions + 1)*(tag[i] - 1) + get_ghost_offset(x[i]);
    rel2wall[m][rel_ind] = (x[i][dim] - coord < 0) && (!append_flag || wall_save[m][rel_ind]);
  }

  int packed_size = (all_atoms_count + 7) / 8; // Number of bytes needed
  std::vector<unsigned char> packed(packed_size, 0);

  for (int i = 0; i < all_atoms_count; i++) {
      packed[i / 8] |= rel2wall[m][i] << (i % 8);
  }
  
  // Perform MPI_Allreduce on the packed data
  MPI_Allreduce(MPI_IN_PLACE, packed.data(), packed_size, MPI_BYTE, mpi_bor_custom, lmp->world);

  // Unpack the data back into the bool array
  for (int i = 0; i < all_atoms_count; i++) {
    rel2wall[m][i] = (packed[i / 8] >> (i % 8)) & 1;
  }

}

void FixWallGhost::wall_particle(int m, double coord)
{


  inum = force->pair->list->inum;
  ilist = force->pair->list->ilist;
  numneigh = force->pair->list->numneigh;
  firstneigh = force->pair->list->firstneigh;

  int dim = wallwhich[m] / 2;

  x = atom->x;
  mask = atom->mask;

  int i, j, ii, jj, jnum, mask_index, atm1, atm2;
  int *jlist;
  tag = lmp->atom->tag;

  bool reset_wall_memory;
  if (every[m] == -1) reset_wall_memory = !filled_before[m];
  else if (!every[m]) reset_wall_memory = true;
  else if (!((update->ntimestep - initial_timestep) % every[m])) reset_wall_memory = true;


  if (reset_wall_memory){
    set_wall_memory(m, dim, coord);
    filled_before[m] = true;
  }

  std::unordered_set<int> to_mask; //Local ids of neighboring atoms of each atom in question that will be masked because of being on an opposite side of the wall

  for (ii = 0; ii < inum; ii++) {
    to_mask.clear();
    i = ilist[ii];
    if (std::abs(x[i][dim] - coord) > cutoff[m]) continue;

    if (mask[i] & groupbit){
      jnum = numneigh[i];
      jlist = firstneigh[i];

      for (jj = 0; jj < jnum; jj++){
        j = jlist[jj];
        j &= NEIGHMASK;
        if (std::abs(x[j][dim] - coord) > cutoff[m]) continue;

        if (mask[j] & groupbit){
          atm1 = rel2wall[m][(ghost_dimensions + 1)*(tag[i] - 1) + get_ghost_offset(x[i])];
          atm2 = rel2wall[m][(ghost_dimensions + 1)*(tag[j] - 1) + get_ghost_offset(x[j])];
          if ((atm1 != -1) && (atm2 != -1) && (atm1 ^ atm2)){
            to_mask.emplace(j);
          }
        }
      }

      numneigh[i] -= to_mask.size();
      mask_index = 0;
      for (jj = 0; jj < jnum; jj++){
        if (to_mask.find(jlist[jj]) == to_mask.end()){
          jlist[mask_index++] = jlist[jj];
        }
      }
    }
  }
  
}

void FixWallGhost::setup_pre_force(int vflag) {
    initial_timestep = update->ntimestep;
    post_force(vflag);
}

int FixWallGhost::get_ghost_offset(double *atom_x){
    // Determine ghost shifts in each direction
    int dx = (atom_x[0] < domain->boxlo[0]) ? -1 : ((atom_x[0] >= domain->boxhi[0]) ? 1 : 0);
    int dy = (atom_x[1] < domain->boxlo[1]) ? -1 : ((atom_x[1] >= domain->boxhi[1]) ? 1 : 0);
    int dz = (atom_x[2] < domain->boxlo[2]) ? -1 : ((atom_x[2] >= domain->boxhi[2]) ? 1 : 0);

    // Define the valid ghost cells based on periodicity
    const int all_periodic[] = {11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
    const int xy_periodic[] = {11, 13, 14, 16, 17};
    const int xz_periodic[] = {13, 16, 19, 22, 25};
    const int yz_periodic[] = {13, 14, 21, 22, 23};
    const int x_periodic_only[] = {13, 16};
    const int y_periodic_only[] = {13, 14};
    const int z_periodic_only[] = {13, 22};

    // Choose the appropriate valid cell set
    const int *valid_cells = nullptr;

    if (!domain->boundary[0][0] && !domain->boundary[1][0] && !domain->boundary[2][0]) {
        valid_cells = all_periodic;
    } else if (!domain->boundary[0][0] && !domain->boundary[1][0]) {
        valid_cells = xy_periodic;
    } else if (!domain->boundary[0][0] && !domain->boundary[2][0]) {
        valid_cells = xz_periodic;
    } else if (!domain->boundary[1][0] && !domain->boundary[2][0]) {
        valid_cells = yz_periodic;
    } else if (!domain->boundary[0][0]) {
        valid_cells = x_periodic_only;
    } else if (!domain->boundary[1][0]) {
        valid_cells = y_periodic_only;
    } else if (!domain->boundary[2][0]) {
        valid_cells = z_periodic_only;
    } else {
        return 0; // No periodicity, no ghost atoms
    }

    // Compute the original 3D cell ID using LAMMPS-style indexing
    int original_id = (dz + 1) * 9 + (dx + 1) * 3 + (dy + 1);

    // Map the original ID to a compact index
    for (int i = 0; i <= ghost_dimensions; i++) {
        if (valid_cells[i] == original_id) {
            return i; // Return compact index (0 to num_cells - 1)
        }
    }

    return -1; // Should never happen if logic is correct
}


