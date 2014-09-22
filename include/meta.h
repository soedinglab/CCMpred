#ifdef JANSSON
#include "jansson.h"

json_t* meta_read_json(FILE *fmeta);
json_t* meta_create();
json_t* meta_add_step(json_t *meta, char *name);
json_t* meta_file_from_path(char* path);

#ifdef MSGPACK

#include <msgpack.h>
void json_to_msgpack(msgpack_packer *pk, json_t *o);
void meta_write_msgpack(msgpack_packer *pk, json_t *meta);
#endif

#endif


