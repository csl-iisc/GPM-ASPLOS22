#include "virginian.h"
#include "libgpm.cuh"

/**
 * @ingroup database
 * @brief Close the currently opened database
 *
 * Close the open database. This is accomplished by clearing every single
 * main-memory tablet slot, thus ensuring the changes to every tablet are
 * reflected on disk, then writing the fixed-size virg_db struct stored in the
 * virginian state struct to the head of the database file, then writing the
 * variable-sized meta information between this fixed-size area and the start of
 * the actual tablets. The variable-size meta information is a list of all the
 * tablets in the file. This function should be called only if no tablets are
 * locked in memory.
 *
 * @param v Pointer to the state struct of the database system
 * @return VIRG_SUCCESS or VIRG_FAIL depending on errors during the function
 * call
 */
int virg_db_close(virginian *v) {
	unsigned i, r;
	size_t size;

	// check that a database is currently open
	if(v->dbfd == NULL)
		return VIRG_SUCCESS;

	// set size to the meta area of the file
	size = sizeof(virg_db);
	size += v->db.alloced_tablets * sizeof(virg_tablet_info);

	// clear every tablet slot, thus writing every in-memory tablet to disk
	for(i = 0; i < VIRG_MEM_TABLETS; i++)
		if(v->tablet_slot_status[i])
			virg_db_clear(v, i);

	// write the fixed-size meta information in the virg_db struct to disk
	r = gpm_memcpy(v->dbfd, &v->db, sizeof(virg_db), cudaMemcpyHostToDevice);
	VIRG_CHECK(r != cudaSuccess, "Problem writing db meta info");

	// write the variable-sized meta information (the tablet slot use and ids)
	// to disk
	size = v->db.alloced_tablets * sizeof(virg_tablet_info);
	r = gpm_memcpy(v->dbfd + sizeof(virg_db), v->db.tablet_info, size, cudaMemcpyHostToDevice);
	VIRG_CHECK(r != cudaSuccess, "Problem writing db meta info");

	// close the database file
	r = gpm_unmap(v->file_name, v->dbfd, VIRG_MAX_FILE_SIZE);
	VIRG_CHECK(r != cudaSuccess, "Problem closing file");
	v->dbfd = NULL;

	// free the variable size meta information block
	free(v->db.tablet_info);

	return VIRG_SUCCESS;
}

