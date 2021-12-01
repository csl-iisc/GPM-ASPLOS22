#include "virginian.h"

const char *virg_opstring(int op) {
	static const char* ops[] = {
		/* 0, */   "Table",
		/* 1, */   "ResultColumn",
		/* 2, */   "Parallel",
		/* 3, */   "Finish",
		/* 4, */   "Column",
		/* 5, */   "Rowid",
		/* 6, */   "Result",
		/* 7, */   "Converge",
		/* 8, */   "Invalid",
		/* 9, */   "Cast",
		/* 10, */   "Integer",
		/* 11, */   "Float",
		/* 12, */   "Le",
		/* 13, */   "Lt",
		/* 14, */   "Ge",
		/* 15, */   "Gt",
		/* 16, */   "Eq",
		/* 17, */   "Neq",
		/* 18, */   "Add",
		/* 19, */   "Sub",
		/* 20, */   "Mul",
		/* 21, */   "Div",
		/* 22, */   "And",
		/* 23, */   "Or",
		/* 24, */   "Not",
		/* 25 */   "Nop",
	""};
	return ops[op];}

