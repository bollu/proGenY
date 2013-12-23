#include "uniqueNamer.h"
#include "../IO/Hash.h"

void initUniqueNames(int initialCapacity, UniqueNames &names){
	if(names.nameMap != NULL) return;

	names.nameMap = Hash::CreateHashmap(initialCapacity);
};

std::string genUniqueName(UniqueNames &names, const char *baseName){
	const Hash *nameHash = Hash::getHash(baseName);

	int *countPtr = (int*) hashmapGet(names.nameMap, (void*)nameHash);

	int count = 0;

	if(countPtr == NULL){
		countPtr = new int (0);
		hashmapPut(names.nameMap, (void*)nameHash, (void *)(countPtr) );	

	}else{
		count = *countPtr;
	}

	std::string uniqueName = (baseName + std::to_string(count));
	(*countPtr)++;

	return uniqueName;
};


bool destoyNameMap(void* key, void* value, void* context){
	delete ((int *)value);
	return false;
}


void destroyUniqueNames(UniqueNames &names){
	hashmapForEach(names.nameMap, &destoyNameMap, NULL);
	hashmapFree(names.nameMap);
};