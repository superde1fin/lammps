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
#include "modify.h"
#include "update.h"
#include "atom.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "memory.h"
#include "comm.h"

#include "region_slab.h"


#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixWallGhostRegion::FixWallGhostRegion(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), lmp(lmp), idregion(nullptr), region(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal fix wall/ghost/region command");

  virial_global_flag = virial_peratom_flag = 1;
  dynamic_group_allow = 1;
  filled_before = false;
  if (domain->xperiodic && domain->yperiodic && domain->zperiodic){
    ghost_dimensions = 13;
  } else if (domain->xperiodic && domain->yperiodic || domain->xperiodic && domain->zperiodic || domain->yperiodic && domain->zperiodic){
    ghost_dimensions = 4;
  } else if (domain->xperiodic || domain->yperiodic || domain->zperiodic){
    ghost_dimensions = 1;
  } else {
    ghost_dimensions = 0;
  }

  restart_peratom = 1;
  restart_global = 0;
  from_restart = false;
  scanned_restart = false;
  has_parent_wall = false;

  rebuild_ts = 0;
  last_ts_built = -1;

  // parse args

  region = domain->get_region_by_id(arg[3]);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", arg[3]);
  idregion = utils::strdup(arg[3]);

  if (narg > 5){
    if (strcmp(arg[4], "append") == 0){
      Fix *parent_wall_candidate = modify->get_fix_by_id(arg[5]);
      if (!parent_wall_candidate) error->all(FLERR, "Wall {} for fix wall/region does not exist", arg[5]);
      if (strcmp(parent_wall_candidate->style, "wall/ghost/region") != 0){
        error->all(FLERR, "Wall {} is not a wall/region style", arg[5]);
      } else {
        parent_wall = dynamic_cast<FixWallGhostRegion*>(parent_wall_candidate);
      }
      has_parent_wall = true;
      restart_peratom = 0;
      restart_global = 1;
      idparent = utils::strdup(arg[5]);
    }
  }

  if (region->full_volume) check_across = &FixWallGhostRegion::across_region;
  else check_across = &FixWallGhostRegion::across_partial_region;

  if (has_parent_wall){
    rel2wall = parent_wall->rel2wall;
  } else {
    rel2wall = std::make_shared<RelMap>();
  }


  if (!has_parent_wall) atom->add_callback(Atom::RESTART);


}

/* ---------------------------------------------------------------------- */

FixWallGhostRegion::~FixWallGhostRegion(){
  if (rel2wall){
    rel2wall.reset();
  }
  if (!has_parent_wall) {
    atom->delete_callback(id, Atom::RESTART);
  }
}

/* ---------------------------------------------------------------------- */

int FixWallGhostRegion::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  mask |= PRE_NEIGHBOR;
  mask |= MIN_PRE_NEIGHBOR;
  mask |= POST_NEIGHBOR;
  mask |= MIN_POST_NEIGHBOR;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::init()
{
  region = domain->get_region_by_id(idregion);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", idregion);

  if (has_parent_wall) {
    Fix *parent_wall_candidate = modify->get_fix_by_id(idparent);
    if (!parent_wall_candidate) error->all(FLERR, "Wall {} for fix wall/region does not exist", idparent);
    if (strcmp(parent_wall_candidate->style, "wall/ghost/region") != 0){
      error->all(FLERR, "Wall {} is not a wall/region style", idparent);
    } else {
      parent_wall = dynamic_cast<FixWallGhostRegion*>(parent_wall_candidate);
      rel2wall = parent_wall->rel2wall;
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::pre_neighbor()
{
  rebuild_ts = update->ntimestep;
}
void FixWallGhostRegion::min_pre_neighbor()
{
  pre_neighbor();
}

void FixWallGhostRegion::setup_post_neighbor(){
  x = atom->x;
  mask = atom->mask;
  int i, j, ii, jj, jnum, all_atoms_count, min_tag, max_tag, gl_id, total_AN;
  tagint *tag = atom->tag;
  int *jlist;

  if (has_parent_wall && restart_reset) filled_before = true;

  MPI_Allreduce(MPI_IN_PLACE, &from_restart, 1, MPI_INT, MPI_SUM, lmp->world);

  for(int nlid = 0; nlid < lmp->neighbor->nlist; nlid++){
    inum = lmp->neighbor->lists[nlid]->inum;
    ilist = lmp->neighbor->lists[nlid]->ilist;
    numneigh = lmp->neighbor->lists[nlid]->numneigh;
    firstneigh = lmp->neighbor->lists[nlid]->firstneigh;


    int prev_openflag = region->openflag;
    if (!filled_before && !from_restart){
      total_AN = 0;
      region->openflag = 1;
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        

        if (mask[i] & groupbit){
          jnum = numneigh[i];
          jlist = firstneigh[i];

          for (jj = 0; jj < jnum; jj++){
            j = jlist[jj];
            j &= NEIGHMASK;
            if (x[j][0] > 2*domain->boxhi[0] - domain->boxlo[0] || x[j][0] < 2*domain->boxlo[0] - domain->boxhi[0] || x[j][1] > 2*domain->boxhi[1] - domain->boxlo[1] || x[j][1] < 2*domain->boxlo[1] - domain->boxhi[1] || x[j][2] > 2*domain->boxhi[2] - domain->boxlo[2] || x[j][2] < 2*domain->boxlo[2] - domain->boxhi[2]){
              error->all(FLERR, "Fix style wall/ghost/region does not support the use of second set of ghost atoms. Reduce your cutoff");
            }

            if (mask[j] & groupbit){
              if ((this->*check_across)(x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2])){
                if (local_rel2wall.find(i) == local_rel2wall.end()){
                  local_rel2wall[i] = {j};
                } else {
                  local_rel2wall[i].emplace(j);
                }
                total_AN++;
              }
            }
          }
        }
      }
      region->openflag = prev_openflag;


      all_atoms_count = (ghost_dimensions + 1)*atom->natoms; //Ensure all ghost atoms are included as well
      int *offsets; //Array with offsets at which antineighbors will be stored
      int *num_AN; //Array with a number of antineighbors
      int *key_tags; //Array with tags of each antineihbor key
      int *all_AN; //Array with a custom identifiers of antineighbors
      int proc_AN[comm->nprocs]; //Array storing total number of antineighbors accumulated from each process

      lmp->memory->create(offsets, all_atoms_count, "ghostwall:offsets");
      lmp->memory->create(num_AN, all_atoms_count, "ghostwall:num_AN");
      lmp->memory->create(key_tags, all_atoms_count, "ghostwall:key_tags");
      std::fill(offsets, offsets + all_atoms_count, 0);
      std::fill(num_AN, num_AN + all_atoms_count, 0);
      std::fill(key_tags, key_tags + all_atoms_count, 0);

      MPI_Allgather(&total_AN, 1, MPI_INT, proc_AN, 1, MPI_INT, lmp->world);
      int my_offset = 0;
      for(i = 0; i < comm->me; i++) my_offset += proc_AN[i];


      MPI_Allreduce(MPI_IN_PLACE, &total_AN, 1, MPI_INT, MPI_SUM, lmp->world);
      lmp->memory->create(all_AN, total_AN, "ghostwall:all_AN");
      std::fill(all_AN, all_AN + total_AN, 0);


      int local_offset = 0;
      int key;
      for (const auto& pair : local_rel2wall){
        key = pair.first;
        const std::unordered_set<int>& neigh_set = pair.second;
        gl_id = (ghost_dimensions + 1)*(tag[key] - 1) + get_ghost_offset(x[key]);
        if (neigh_set.size()){
          offsets[gl_id] = my_offset + local_offset + 1;
          key_tags[gl_id] = tag[key];
        }
        for (const auto& neigh_tag : neigh_set){
          all_AN[my_offset + (local_offset++)] = tag[neigh_tag];
          num_AN[gl_id]++;
        }
      }

      local_rel2wall.clear();

      MPI_Allreduce(MPI_IN_PLACE, all_AN, total_AN, MPI_INT, MPI_SUM, lmp->world);
      MPI_Allreduce(MPI_IN_PLACE, offsets, all_atoms_count, MPI_INT, MPI_SUM, lmp->world);
      MPI_Allreduce(MPI_IN_PLACE, num_AN, all_atoms_count, MPI_INT, MPI_SUM, lmp->world);
      MPI_Allreduce(MPI_IN_PLACE, key_tags, all_atoms_count, MPI_INT, MPI_SUM, lmp->world);

      for (i = 0; i < all_atoms_count; i++){
        if(offsets[i] > 0){
          for (local_offset = 0; local_offset < num_AN[i]; local_offset++){
            if (key_tags[i] < all_AN[offsets[i] + local_offset - 1]) {
              min_tag = key_tags[i];
              max_tag = all_AN[offsets[i] + local_offset - 1];
            } else {
              max_tag = key_tags[i];
              min_tag = all_AN[offsets[i] + local_offset - 1];
            }
            if (rel2wall->find(min_tag) != rel2wall->end()) {
              (*rel2wall)[min_tag].emplace(max_tag);
            } else {
              (*rel2wall)[min_tag] = {max_tag};
            }
          }
        }
      }


      lmp->memory->destroy(offsets);
      lmp->memory->destroy(num_AN);
      lmp->memory->destroy(all_AN);
      lmp->memory->destroy(key_tags);

    }
  }
    
  if (!scanned_restart && from_restart){
    MPI_Barrier(lmp->world);
    std::vector<int> local_serialized = serializeMap(rel2wall);
    int local_size = local_serialized.size();
    
    // Buffers for communication
    std::vector<MPI_Request> send_requests_sizes;
    std::vector<MPI_Request> send_requests_data;
    std::vector<MPI_Request> recv_requests;
    std::vector<int> recv_sizes(comm->nprocs);
    std::vector<std::vector<int>> received_data(comm->nprocs);


    //  Non-blocking receives: Receive data from all other processes
    for (int i = 0; i < comm->nprocs; ++i) {
      if (i != comm->me) {
        MPI_Request req;
        recv_sizes[i] = -1;
        MPI_Irecv(&recv_sizes[i], 1, MPI_INT, i, 0, lmp->world, &req);
        recv_requests.push_back(req);
      }
    }


    //  Non-blocking sends: Send local data sizes to all other processes
    for (int i = 0; i < comm->nprocs; ++i) {
      if (i != comm->me) {
        MPI_Request req;
        MPI_Isend(&local_size, 1, MPI_INT, i, 0, lmp->world, &req);
        send_requests_sizes.push_back(req);
      }
    }

    MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUS_IGNORE); // Ensure sizes are received
    MPI_Waitall(send_requests_sizes.size(), send_requests_sizes.data(), MPI_STATUS_IGNORE);

    recv_requests.clear();

    
    MPI_Barrier(lmp->world);  // Ensure all ranks start together
    
    // First loop: Prepare receive buffers
    for (int i = 0; i < comm->nprocs; ++i) {
        if (i != comm->me) {
            received_data[i].resize(recv_sizes[i]);
        }
    }
    
    // Second loop: Exchange data
    for (int i = 0; i < comm->nprocs; ++i) {
        if (i != comm->me) {
            MPI_Sendrecv(local_serialized.data(), local_size, MPI_INT, i, 1,  // Send to i
                         received_data[i].data(), recv_sizes[i], MPI_INT, i, 1,  // Receive from i
                         lmp->world, MPI_STATUS_IGNORE);
        }
    }
    
    MPI_Barrier(lmp->world);  // Ensure all ranks finish exchanging data

    //  Merge received data into local map
    for (int i = 0; i < comm->nprocs; ++i) {
      if (i != comm->me && !received_data[i].empty()) {
        RelMap received_map = deserializeMap(received_data[i]);
        mergeMaps(rel2wall, received_map);
      }
    }


    MPI_Barrier(lmp->world);
    scanned_restart = true;
  }
  
  from_restart = false;
  filled_before = true;
}



/* ---------------------------------------------------------------------- */

void FixWallGhostRegion::min_pre_force(int vflag)
{
  pre_force(vflag);
}


void FixWallGhostRegion::pre_force(int vflag)
{

  if (rebuild_ts == update->ntimestep && last_ts_built != update->ntimestep){
    last_ts_built = update->ntimestep;
    v_init(vflag);
    int *type = atom->type;
    x = atom->x;
    mask = atom->mask;
    int i, j, ii, jj, jnum, mask_index, min_tag, max_tag;
    tagint *tag = atom->tag;
    int *jlist;
    std::unordered_set<int> tmp_set;


    for(int nlid = 0; nlid < lmp->neighbor->nlist; nlid++){
      inum = lmp->neighbor->lists[nlid]->inum;
      ilist = lmp->neighbor->lists[nlid]->ilist;
      numneigh = lmp->neighbor->lists[nlid]->numneigh;
      firstneigh = lmp->neighbor->lists[nlid]->firstneigh;


      if (!has_parent_wall){
        for (ii = 0; ii < inum; ii++) {
          tmp_set.clear();
          i = ilist[ii];

          if (mask[i] & groupbit){
            jnum = numneigh[i];
            jlist = firstneigh[i];

            for (jj = 0; jj < jnum; jj++){
              j = jlist[jj];
              j &= NEIGHMASK;

              if (mask[j] & groupbit){
                if (tag[i] < tag[j]){
                  min_tag = tag[i];
                  max_tag = tag[j];
                } else {
                  min_tag = tag[j];
                  max_tag = tag[i];
                }
                if (rel2wall->find(min_tag) != rel2wall->end()){
                  if ((*rel2wall)[min_tag].find(max_tag) != (*rel2wall)[min_tag].end()){
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
                numneigh[i]--;
              }
            }
          }
        }
      }
    }
  }
}

void FixWallGhostRegion::setup_pre_force(int vflag) {
    rebuild_ts = update->ntimestep;
    last_ts_built = -1;
    pre_force(vflag);
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

    if (domain->xperiodic && domain->yperiodic && domain->zperiodic) {
        valid_cells = all_periodic;
    } else if (domain->xperiodic && domain->yperiodic) {
        valid_cells = xy_periodic;
    } else if (domain->xperiodic && domain->zperiodic) {
        valid_cells = xz_periodic;
    } else if (domain->yperiodic && domain->zperiodic) {
        valid_cells = yz_periodic;
    } else if (domain->xperiodic) {
        valid_cells = x_periodic_only;
    } else if (domain->yperiodic) {
        valid_cells = y_periodic_only;
    } else if (domain->zperiodic) {
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

int FixWallGhostRegion::maxsize_restart(){
  int nmax = 0;
  int sz;
  for (int i = 0; i < atom->nlocal; i++){
    sz = (*rel2wall)[atom->tag[i]].size();
    if (sz > nmax) nmax = sz;
  }
  MPI_Allreduce(MPI_IN_PLACE, &nmax, 1, MPI_INT, MPI_MAX, lmp->world);
  return nmax + 1;
}

int FixWallGhostRegion::size_restart(int i){
  if (rel2wall->find(atom->tag[i]) != rel2wall->end()){
    return (*rel2wall)[atom->tag[i]].size() + 1;
  } else {
    return 1;
  }
}

int FixWallGhostRegion::pack_restart(int i, double *buf){
  buf[0] = (*rel2wall)[atom->tag[i]].size() + 1;
  int m = 0;
  for (const int neigh : (*rel2wall)[atom->tag[i]]){
    buf[++m] = static_cast<double>(neigh);
  }
  return (*rel2wall)[atom->tag[i]].size() + 1;
}

void FixWallGhostRegion::unpack_restart(int local_ind, int saved_fix_ind){
  double **extra = atom->extra;
  int neigh_tag;

  int *proc_tags;
  int m = 0, min_tag, max_tag;
  for (int i = 0; i < saved_fix_ind; i++) m += static_cast<int>(extra[local_ind][m]);

  int num_neighs = static_cast<int>(extra[local_ind][m++]) - 1;
  for (int i = 0; i < num_neighs; i++){
    neigh_tag = static_cast<int>(extra[local_ind][m++]);
    if (atom->tag[local_ind] > neigh_tag){
      min_tag = neigh_tag;
      max_tag = atom->tag[local_ind];
    } else {
      max_tag = neigh_tag;
      min_tag = atom->tag[local_ind];
    }
    if (rel2wall->find(min_tag) != rel2wall->end()) {
      (*rel2wall)[min_tag].emplace(max_tag);
    } else {
      (*rel2wall)[min_tag] = {max_tag};
    }
  }
  from_restart = 1;
}

// Serialize unordered_map<int, unordered_set<int>> into a vector<int>
std::vector<int> FixWallGhostRegion::serializeMap(const std::shared_ptr<RelMap> local_map) {
    std::vector<int> serialized_data;
    for (const auto &entry : *local_map) {
        serialized_data.push_back(entry.first); // Key
        serialized_data.push_back(entry.second.size()); // Number of elements in set
        for (int value : entry.second) {
            serialized_data.push_back(value); // Set elements
        }
    }
    return serialized_data;
}

// Deserialize vector<int> back into unordered_map<int, unordered_set<int>>
RelMap FixWallGhostRegion::deserializeMap(const std::vector<int> &data) {
    RelMap result;
    size_t i = 0;
    while (i < data.size()) {
        int key = data[i++];
        int set_size = data[i++];
        std::unordered_set<int> values;
        for (int j = 0; j < set_size; ++j) {
            values.insert(data[i++]);
        }
        result[key] = values;
    }
    return result;
}

// Merge received map into local map
void FixWallGhostRegion::mergeMaps(std::shared_ptr<RelMap> local_map, const RelMap &received_map) {
    for (const auto &entry : received_map) {
        (*local_map)[entry.first].insert(entry.second.begin(), entry.second.end());
    }
}
