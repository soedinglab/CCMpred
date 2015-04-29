#include <time.h>
#include <stdlib.h>

#ifdef UUID
#include <uuid/uuid.h>
#endif

#ifdef JANSSON
#include <jansson.h>
#endif

#ifdef MSGPACK
#include <msgpack.h>
#endif

#include "meta.h"

#ifdef JANSSON

json_t* meta_read_json(FILE *fmeta) {

	json_error_t *err;
	json_t *meta = json_loadf(fmeta, JSON_REJECT_DUPLICATES, err);
	if(meta == NULL) {
		return meta;
	}

	return meta;
}

json_t* meta_create() {
	json_t *meta = json_object();
	json_object_set(meta, "version", json_string("1.0.0"));
	json_object_set(meta, "workflow", json_array());

#ifdef UUID
	uuid_t uuid;
	uuid_generate(uuid);
	char s_uuid[37];
	uuid_unparse_lower(uuid, s_uuid);
	json_object_set(meta, "uuid", json_string(s_uuid));
#endif

	return meta;
}


json_t* meta_add_step(json_t *meta, char* name) {
	json_t *step = json_object();

	json_t *workflow = json_object_get(meta, "workflow");
	json_array_append(workflow, step);

	json_object_set(step, "step", json_string(name));

	time_t ltime = time(NULL);
	char timestamp[256];
	strftime(timestamp, 256, "%FT%TZ", gmtime(&ltime));
	json_object_set(step, "timestamp", json_string(timestamp));

	return step;
}


json_t* meta_file_from_path(char* path) {

	json_t *meta_file = json_object();
	json_object_set(meta_file, "path", json_string(path));

	char *rp = realpath(path, NULL);
	json_object_set(meta_file, "abspath", json_string(rp));

	char *bn = basename(rp);
	json_object_set(meta_file, "basename", json_string(bn));

	return meta_file;
}

#ifdef MSGPACK


void json_object_to_msgpack(msgpack_packer *pk, json_t *o) {
	msgpack_pack_map(pk, json_object_size(o));

	const char *key;
	json_t *value;
	json_object_foreach(o, key, value) {
		msgpack_pack_str(pk, strlen(key));
		msgpack_pack_str_body(pk, key, strlen(key));

		json_to_msgpack(pk, value);
	}
}

void json_array_to_msgpack(msgpack_packer *pk, json_t *o) {
	msgpack_pack_array(pk, json_array_size(o));

	size_t index;
	json_t *value;
	json_array_foreach(o, index, value) {
		json_to_msgpack(pk, value);
	}
}

void json_string_to_msgpack(msgpack_packer *pk, json_t *o) {
	const char *val = json_string_value(o);
	msgpack_pack_str(pk, strlen(val));
	msgpack_pack_str_body(pk, val, strlen(val));
}

void json_integer_to_msgpack(msgpack_packer *pk, json_t *o) {
	int val = json_integer_value(o);
	msgpack_pack_int(pk, val);
}

void json_real_to_msgpack(msgpack_packer *pk, json_t *o) {
	double val = json_real_value(o);
	msgpack_pack_double(pk, val);
}

void json_boolean_to_msgpack(msgpack_packer *pk, json_t *o) {
	if(json_is_true(o)) {
		msgpack_pack_true(pk);
	} else {
		msgpack_pack_false(pk);
	}
}

void json_to_msgpack(msgpack_packer *pk, json_t *o) {
	switch(json_typeof(o)) {

		case JSON_OBJECT:
			json_object_to_msgpack(pk, o);
			break;
		case JSON_ARRAY:
			json_array_to_msgpack(pk, o);
			break;
		case JSON_STRING:
			json_string_to_msgpack(pk, o);
			break;
		case JSON_INTEGER:
			json_integer_to_msgpack(pk, o);
			break;
		case JSON_REAL:
			json_real_to_msgpack(pk, o);
			break;
		case JSON_TRUE:
		case JSON_FALSE:
			json_boolean_to_msgpack(pk, o);
			break;
		default:
			printf("WARN: Unknown JSON type!\n");
	}
}

void meta_write_msgpack(msgpack_packer *pk, json_t *meta) {

	msgpack_pack_str(pk, 4);
	msgpack_pack_str_body(pk, "meta", 4);

	json_to_msgpack(pk, meta);
}

#endif


#endif
