#include "virginian.h"
#include "libgpm.cuh"

/**
 * @ingroup database
 * @brief Write the tablet in a tablet slot to disk
 *
 * This function writes the tablet in a tablet slot to an area of the on-disk
 * database file. It is not thread-safe, so locking of the tablet slot array and
 * the tablet should be performed in a multi-threaded environment. This function
 * handles updating the disk slot meta information about where tablets are
 * stored.
 *
 * This function is somewhat complex because of the variable-sized disk meta
 * information. The on-disk database stores a fixed-size block of meta
 * information that corresponds to a virg_db struct. The listing of the tablet
 * IDs in the database file is of a variable size because we can have an
 * arbitrarily large number of tablets stored in the file. After the tablet
 * listing, we store actual tablets. Note that this variable-size tablet list,
 * or tablet info block, is stored in memory while the database is open and
 * written to disk when the database closes. Under this scheme, we often exceed
 * the currently allocated size of this tablet list, in which case we must
 * allocate a larger area and copy the contents to the new allocation. Problems
 * arise, however, because this new larger size may intersect with the first
 * tablet in the database file, in which case we must move this tablet to the
 * back of the list, and move on disk if it is not loaded into memory, to make
 * more room for our tablet list in the database file. A significant portion of
 * this function is devoted to this edge case.
 *
 * @param v Pointer to the state struct of the database system
 * @param slot The number of the tablet slot to be written to disk
 * @return VIRG_SUCCESS or VIRG_FAIL depending on errors during the function
 * call
 */
 

int virg_db_write_tab(virginian *v, virg_tablet_meta *tab, cudaStream_t *stream)
{    
	unsigned i;
	size_t offset = v->db.block_size;
	
    // if the tablet already has an allocated spot on disk
	if(tab->info != NULL) {
		offset += tab->info->disk_slot * VIRG_TABLET_SIZE;
	}
	// not written to disk
	else {
		// find an empty spot on disk
		for(i = 0; i < v->db.alloced_tablets; i++)
			if(v->db.tablet_info[i].used == 0)
				break;

		// if no empty spot, expand info area, disk expands implicitly
		if(i >= v->db.alloced_tablets) {
			unsigned new_alloced_tablets = v->db.alloced_tablets +
				VIRG_TABLET_INFO_INCREMENT;
			size_t size = new_alloced_tablets * sizeof(virg_tablet_info);

			// if the expanded info area still fits in the space before the
			// first tablet
			if(v->db.block_size >= size + sizeof(virg_db)) {
				// allocate a bigger area in memory
				virg_tablet_info *info = (virg_tablet_info *)malloc(size);
				// copy the tablet info in memory into the newly allocated block
				memcpy(info, v->db.tablet_info, v->db.alloced_tablets *
					sizeof(virg_tablet_info));

				// update pointers in in-memory tablets to the new tablet info
				// block
				for(i = 0; i < VIRG_MEM_TABLETS; i++)
					if(v->tablet_slot_status[i] != 0 &&
						v->tablet_slots[i]->info != NULL)
						v->tablet_slots[i]->info =
							&info[v->tablet_slots[i]->info->disk_slot];

				// free the old tablet info block
				free(v->db.tablet_info);
				v->db.tablet_info = info;

				// initialize the new area of the tablet info block
				for(i = v->db.alloced_tablets; i < new_alloced_tablets; i++) {
					v->db.tablet_info[i].used = 0;
					v->db.tablet_info[i].disk_slot = i;
#ifdef VIRG_DEBUG
					v->db.tablet_info[i].id = 0xDEADBEEF;
#endif
				}

				// use the first newly allocated slot
				i = v->db.alloced_tablets;
				v->db.alloced_tablets = new_alloced_tablets;
			}
			// the newly allocated tablet info block is big enough to intersect
			// with the first tablet stored on disk
			// resize tablet info block and move first tablet to the back
			else {
				// allocate new info block and copy with the first moved to the
				// last
				virg_tablet_info *info = (virg_tablet_info *)malloc(size);
				memcpy(info, v->db.tablet_info + 1,
					(v->db.alloced_tablets - 1) * sizeof(virg_tablet_info));
				memcpy(info + v->db.alloced_tablets - 1, v->db.tablet_info,
					sizeof(virg_tablet_info));

				// update the disk slot information because we've shifted things
				for(i = 0; i < v->db.alloced_tablets; i++)
					info[i].disk_slot = i;
				for( ; i < new_alloced_tablets; i++) {
					info[i].used = 0;
					info[i].disk_slot = i;
#ifdef VIRG_DEBUG
					info[i].id = 0xDEADBEEF;
#endif
				}

				// we must move first tablet on disk so that it isn't overwritten
				void *buff = malloc(VIRG_TABLET_SIZE);
				VIRG_CHECK(buff == NULL, "Could not allocate temporary memory");
				if(stream != NULL) {
				    cudaError_t r = cudaMemcpyAsync(buff, v->dbfd + v->db.block_size, VIRG_TABLET_SIZE,
					cudaMemcpyDeviceToHost, *stream);
				    assert(r == cudaSuccess);

				    r = cudaMemcpyAsync(v->dbfd + v->db.block_size + VIRG_TABLET_SIZE *
					    v->db.alloced_tablets, buff, VIRG_TABLET_SIZE, cudaMemcpyHostToDevice, *stream);
				    assert(r == cudaSuccess);
				}
				else {
				    cudaError_t r = cudaMemcpy(buff, v->dbfd + v->db.block_size, VIRG_TABLET_SIZE,
					cudaMemcpyDeviceToHost);
				    assert(r == cudaSuccess);

				    r = gpm_memcpy(v->dbfd + v->db.block_size + VIRG_TABLET_SIZE *
					    v->db.alloced_tablets, buff, VIRG_TABLET_SIZE, cudaMemcpyHostToDevice);
				    assert(r == cudaSuccess);
				}

				free(buff);

				// update pointers in in-memory tablets to the new tablet info
				// block
				for(i = 0; i < VIRG_MEM_TABLETS; i++)
					if(v->tablet_slot_status[i] != 0) {
						if(v->tablet_slot_ids[i] == v->db.tablet_info[0].id) {
							v->tablet_slots[i]->info = &info[v->db.alloced_tablets - 1];
							// since tablet is first, it is moved to the end
							// we must update the base_ptr to point to the new location
							v->tablet_slots[i]->base_ptr = v->dbfd + v->db.block_size + VIRG_TABLET_SIZE *
								v->db.alloced_tablets;
						}
						else if(v->tablet_slots[i]->info != NULL) {
							v->tablet_slots[i]->info = &info[v->tablet_slots[i]->info->disk_slot - 1];
						}
					}

				// free the old in-memory tablet info
				free(v->db.tablet_info);

				// use the first disk slot of the newly allocated area
				v->db.tablet_info = info;
				i = v->db.alloced_tablets;
				v->db.alloced_tablets = new_alloced_tablets;
				v->db.block_size += VIRG_TABLET_SIZE;
			}
		}

		// set out disk offset and the tablet info before writing
		offset = v->db.block_size;
		offset += i * VIRG_TABLET_SIZE;
		v->db.tablet_info[i].used = 1;
		v->db.tablet_info[i].id = tab->id;
		v->db.tablet_info[i].disk_slot = i;
		tab->info = &v->db.tablet_info[i];
	}

    cudaError_t err;
    if(stream != NULL)
	    // perform the tablet write to disk
	    err = cudaMemcpyAsync(v->dbfd + offset, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice, *stream);    
    else
	    // perform the tablet write to disk
	    err = gpm_memcpy(v->dbfd + offset, tab, sizeof(virg_tablet_meta), cudaMemcpyHostToDevice);
	// Update base pointer, in case this is first write to disk
	tab->base_ptr = v->dbfd + offset;
	VIRG_CHECK(err != cudaSuccess, "Failed to write tablet")

	return VIRG_SUCCESS;
}
 
int virg_db_write(virginian *v, unsigned slot)
{
#ifdef VIRG_DEBUG_SLOTS
	// tablet movement debug information
	fprintf(stderr, "WRITE TABLET ID %u\n", v->tablet_slots[slot]->id);
	virg_print_slots(v);
	virg_print_tablet_info(v);
#endif
    return virg_db_write_tab(v, v->tablet_slots[slot]);
}

