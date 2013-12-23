#pragma once
#include "../IO/hashmap.h"
#include <string>

struct UniqueNames{
	Hashmap *nameMap = NULL;
};

void initUniqueNames(int initialCapacity, UniqueNames &names);
std::string genUniqueName(UniqueNames &names, const char *baseName);
void destroyUniqueNames(UniqueNames &names);