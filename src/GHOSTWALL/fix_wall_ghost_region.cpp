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

#include "fix_wall_ghost_region.h"

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
#include "neighbor.h"
#include "memory.h"
#include "domain.h"

#include "region_slab.h"


#include <cstring>
#include <cmath>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixWallGhostRegion::FixWallGhostRegion(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), lmp(lmp), idregion(nullptr), region(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal fix wall/ghost/region command");

  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  dynamic_group_allow = 1;
  filled_before = false;
  wall_save = nullptr;
  if (!domain->boundary[0][0] && !domain->boundary[1][0] && !domain->boundary[2][0]){
    ghost_dimensions = 13;
  } else if (!domain->boundary[0][0] && !domain->boundary[1][0] || !domain->boundary[0][0] && !domain->boundary[2][0] || !domain->boundary[1][0] && !domain->boundary[2][0]){
    ghost_dimensions = 4;
  } else if (!domain->boundary[0][0] || !domain->boundary[1][0] || !domain->boundary[2][0]){
    ghost_dimensions = 1;
  } else {
    ghost_dimensions = 0;
  }

  append_flag = 0;

  // parse args

  region = domain->get_region_by_id(arg[3]);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", arg[3]);
  idregion = utils::strdup(arg[3]);

  if (region->full_volume) check_across = &FixWallGhostRegion::across_region;
  else check_across = &FixWallGhostRegion::across_partial_region;

  every = utils::numeric(FLERR, arg[4], false, lmp);

  if ((narg >= 7) && (strcmp(arg[5], "append") == 0)){
    auto prev_fix = modify->get_fix_by_id(arg[6]);
    if (prev_fix){
      if (strcmp(prev_fix->style, "wall/ghost/region") == 0){
        FixWallGhostRegion *fixGW = dynamic_cast<FixWallGhostRegion*>(prev_fix);
        append_flag = 1;
        wall_save = &fixGW->rel2wall;
      } else {
        error->all(FLERR, "Initial fix has to be style wall/ghost/region and not {}", prev_fix->style);
      }
    } else {
      error->all(FLERR, "No fix with name {} was found", arg[6]);
    }
  }

}

/* ---------------------------------------------------------------------- */

FixWallGhostRegion::~FixWallGhostRegion(){
}

/* ---------------------------------------------------------------------- */

int FixWallGhostRegion::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;

  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::init()
{
  region = domain->get_region_by_id(idregion);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", idregion);

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::setup(int vflag)
{
  initial_timestep = update->ntimestep;

  if (!utils::strmatch(update->integrate_style, "^verlet")) {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::min_setup(int vflag)
{
  post_force(vflag);
}


void FixWallGhostRegion::pre_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::min_post_force(int vflag)
{
  post_force(vflag);
}

void FixWallGhostRegion::post_force(int vflag)
{


  v_init(vflag);

  /*
  if (!force->pair->list) {
    error->all(FLERR, "Fix wall/ghost/region is not compatible with manybody pair styles");
  }
  */

  for(int nlid = 0; nlid < lmp->neighbor->nlist; nlid++){
    //printf("Scanning through nlist %d\n", nlid);
    inum = lmp->neighbor->lists[nlid]->inum;
    ilist = lmp->neighbor->lists[nlid]->ilist;
    numneigh = lmp->neighbor->lists[nlid]->numneigh;
    firstneigh = lmp->neighbor->lists[nlid]->firstneigh;

    int *type = atom->type;

    double cutoff;


    x = atom->x;
    mask = atom->mask;

    int i, j, ii, jj, jnum, mask_index, all_atoms_count, gl_id1, gl_id2, prev_openflag;
    int prev_neighs = 0; //Number of neighbors stored in the previous wall antilist. Only relevant when apend flag is used
    int *jlist;
    tagint *tag = lmp->atom->tag;
    std::unordered_set<int> tmp_set;

    bool reset_wall_memory;
    if (every == -1) reset_wall_memory = !filled_before;
    else if (!every) reset_wall_memory = true;
    else if (!((update->ntimestep - initial_timestep) % every)) reset_wall_memory = true;
    else reset_wall_memory = false;

    int total_AN = 0, rank, size; //Total number of antineighbors
    MPI_Comm_size(lmp->world, &size);
    MPI_Comm_rank(lmp->world, &rank);
    prev_openflag = region->openflag;
    if (reset_wall_memory){
      region->openflag = 1;
      /*
      if (append_flag) {
        rel2wall = *wall_save;
        for (const auto& [key, neigh_set] : rel2wall){
          //printf("Scanning wall memory key %d\n", key);
          prev_neighs += neigh_set.size();
        }
      }
      */
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        

        if (mask[i] & groupbit){
          jnum = numneigh[i];
          jlist = firstneigh[i];

          for (jj = 0; jj < jnum; jj++){
            j = jlist[jj];
            j &= NEIGHMASK;
            if (x[j][0] > 2*domain->boxhi[0] - domain->boxlo[0] || x[j][0] < 2*domain->boxlo[0] - domain->boxhi[0] || x[j][1] > 2*domain->boxhi[0] - domain->boxlo[1] || x[j][1] < 2*domain->boxlo[1] - domain->boxhi[1] || x[j][2] > 2*domain->boxhi[2] - domain->boxlo[2] || x[j][2] < 2*domain->boxlo[2] - domain->boxhi[2]){
              error->all(FLERR, "Fix style wall/ghost/region does not support the use of second set of ghost atoms. Reduce your cutoff");
            }

            if (mask[j] & groupbit){
              if ((this->*check_across)(x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2])){
                gl_id1 = (ghost_dimensions + 1)*(tag[i] - 1) + get_ghost_offset(x[i]);
                gl_id2 = (ghost_dimensions + 1)*(tag[j] - 1) + get_ghost_offset(x[j]);
                if (rel2wall.find(gl_id1) == rel2wall.end()){
                  rel2wall[gl_id1] = {gl_id2};
                } else {
                  rel2wall[gl_id1].insert(gl_id2);
                }
                total_AN++;
              }
            }
          }
        }
      }
      all_atoms_count = (ghost_dimensions + 1)*atom->natoms; //Ensure all ghost atoms are included as well
      int *offsets; //Array with offsets at which antineighbors will be stored
      int *num_AN; //Array with a number of antineighbors
      int *all_AN; //Array with a custom identifiers of antineighbors
      int proc_AN[size]; //Array storing total number of antineighbors accumulated from each process

      lmp->memory->create(offsets, all_atoms_count, "ghostwall:offsets");
      lmp->memory->create(num_AN, all_atoms_count, "ghostwall:num_AN");
      std::fill(offsets, offsets + all_atoms_count, 0);
      std::fill(num_AN, num_AN + all_atoms_count, 0);

      MPI_Allgather(&total_AN, 1, MPI_INT, proc_AN, 1, MPI_INT, lmp->world);
      int my_offset = 0;
      for(i = 0; i < rank; i++) my_offset += proc_AN[i];
      //printf("R%d, My offset: %d\n", i, my_offset);

      MPI_Allreduce(MPI_IN_PLACE, &total_AN, 1, MPI_INT, MPI_SUM, lmp->world);
      total_AN += prev_neighs; //Zero if append flag was not invoked
      lmp->memory->create(all_AN, total_AN, "ghostwall:all_AN");
      std::fill(all_AN, all_AN + total_AN, 0);

      int local_offset = 0;
      for (const auto& [key, neigh_set] : rel2wall){
        //printf("KEY: %d\n", key);
        if (neigh_set.size()) offsets[key] = my_offset + local_offset + 1;
        for (const auto& neigh_id : neigh_set){
          //printf("NEIGHID: %d\n", neigh_id);
          all_AN[my_offset + (local_offset++)] = neigh_id;
          num_AN[key]++;
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, all_AN, total_AN, MPI_INT, MPI_SUM, lmp->world);
      MPI_Allreduce(MPI_IN_PLACE, offsets, all_atoms_count, MPI_INT, MPI_SUM, lmp->world);
      MPI_Allreduce(MPI_IN_PLACE, num_AN, all_atoms_count, MPI_INT, MPI_SUM, lmp->world);

      for (i = 0; i < all_atoms_count; i++){
        //printf("Offset %d, of atom id %d\n", offsets[i], i);
        //if(offsets[i] >= 0 && rel2wall.find(i) == rel2wall.end()){
        if(offsets[i] > 0){
          for (local_offset = 0; local_offset < num_AN[i]; local_offset++){
            tmp_set.emplace(all_AN[offsets[i] + local_offset - 1]);
          }
          //printf("Adding %d neighbors to key %d\n", tmp_set.size(), i);
          if (rel2wall.find(i) != rel2wall.end()){
            rel2wall[i].insert(tmp_set.begin(), tmp_set.end());
          } else {
          rel2wall[i] = tmp_set;
          }
          tmp_set.clear();
        }
      }

      lmp->memory->destroy(offsets);
      lmp->memory->destroy(num_AN);
      lmp->memory->destroy(all_AN);

      filled_before = true;
    }


    /*
    for(int i = 0; i < atom->nlocal; i++){
      gl_id1 = (ghost_dimensions + 1)*(tag[i] - 1) + get_ghost_offset(x[i]);
      if(rel2wall.find(gl_id1) != rel2wall.end()){
       type[i] = 3;
      } else {
        for (const auto& [key, neigh_set] : rel2wall){
          if (neigh_set.find(gl_id1) != neigh_set.end()){
            type[i] = 3;
          }
        }
      }
    }
    for (const auto& [key, neigh_set] : rel2wall){
      printf("%d END rel2wall result key %d with size %d\n", append_flag, key, neigh_set.size());
    }
    */


    for (ii = 0; ii < inum; ii++) {
      tmp_set.clear();
      i = ilist[ii];

      if (mask[i] & groupbit){
        //printf("%d Number of neighs of at %d is %d before\n", rank, tag[i], numneigh[i]);
        jnum = numneigh[i];
        jlist = firstneigh[i];

        for (jj = 0; jj < jnum; jj++){
          j = jlist[jj];
          j &= NEIGHMASK;

          if (mask[j] & groupbit){
            gl_id1 = (ghost_dimensions + 1)*(tag[i] - 1) + get_ghost_offset(x[i]);
            gl_id2 = (ghost_dimensions + 1)*(tag[j] - 1) + get_ghost_offset(x[j]);
            if (rel2wall.find(gl_id1) != rel2wall.end()){
              if (rel2wall[gl_id1].find(gl_id2) != rel2wall[gl_id1].end()){
                tmp_set.emplace(j);
              }
            }
            if (rel2wall.find(gl_id2) != rel2wall.end()){
              if (rel2wall[gl_id2].find(gl_id1) != rel2wall[gl_id2].end()){
                tmp_set.emplace(j);
              }
            }
          } 
        }

        mask_index = 0;
        for (jj = 0; jj < jnum; jj++){
          if (tmp_set.find(jlist[jj]) == tmp_set.end()){
            jlist[mask_index++] = jlist[jj];
          } else {
            //DEBUG
            //if ((tag[i] == 36)){
            //  printf("%d Found pair in nlist %d after %d, %d\n", rank, nlid, tag[i], tag[jlist[jj]]);
            //}
            numneigh[i]--;
          }
        }
        //printf("%d Number of neighs of at %d is %d after\n", rank, tag[i], numneigh[i]);
      }
    }
  }
}

void FixWallGhostRegion::setup_pre_force(int vflag) {
    initial_timestep = update->ntimestep;
    post_force(vflag);
}

int FixWallGhostRegion::get_ghost_offset(double *atom_x) {
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


bool FixWallGhostRegion::across_region(double x1, double y1, double z1, double x2, double y2, double z2){
  return region->match(x1, y1, z1) ^ region->match(x2, y2, z2);
}

bool FixWallGhostRegion::across_partial_region(double x1, double y1, double z1, double x2, double y2, double z2){
  return region->across_partial(x1, y1, z1, x2, y2, z2);
}
