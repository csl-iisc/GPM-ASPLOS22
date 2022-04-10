#include "virginian.h"
#include "libgpm.cuh"

/**
 * @ingroup database
 * @brief Open an existing database file
 *
 * Open a database that already exists from its location on disk. This function
 * reads the database meta information into memory, but does not load any
 * tablets into memory. Open databases should be closed with the virg_db_close()
 * function and created with the virg_db_create() function. This function sets
 * the information in the virg_db struct stored within the passed virginian
 * struct. Only one database can be open at a time, so this function should only
 * be called if there are no other open databases.
 *
 * @param v Pointer to the state struct of the database system
 * @param file Location on disk of the database file
 * @return VIRG_SUCCESS or VIRG_FAIL depending on errors during the function
 * call
 */
int virg_db_open(virginian *v, const char *file)
{
	cudaError_t r;

	// ensure that no database is currently open
	VIRG_CHECK(v->dbfd != NULL, "Database already open")

	// open the database file for reading and writing
	v->file_name = file;
	size_t len = 0;
	v->dbfd = (char *)gpm_map_file(v->file_name, len, 0);
	VIRG_CHECK(v->dbfd == NULL, "Problem opening database file")

	// read the fixed size meta information into our virg_db struct
	r = cudaMemcpy(&v->db, v->dbfd, sizeof(virg_db), cudaMemcpyDeviceToHost);
	VIRG_CHECK(r != cudaSuccess, "Corrupt database file")

	// allocate an area for the variable-size tablet tracking information
	// even if there are no tablets, alloced_tablets is set to non-zero when the
	// database is created
	size_t size = v->db.alloced_tablets * sizeof(virg_tablet_info);
	v->db.tablet_info = (virg_tablet_info*) malloc(size);
	VIRG_CHECK(v->db.tablet_info == NULL, "Out of memory")

	// read the meta-information from disk
	r = cudaMemcpy(v->db.tablet_info, v->dbfd + sizeof(virg_db), size, cudaMemcpyDeviceToHost);
	VIRG_CHECK(r != cudaSuccess, "Problem reading tablet info")

	return VIRG_SUCCESS;
}

