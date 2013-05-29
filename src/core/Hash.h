#pragma once
#include <map>
#include <string>

typedef unsigned long long uint64_t;


class Hash{

public:

	static const Hash* getHash(std::string str);

	//convert a given hash to a string. expensive, but
	//useful for logging
	static std::string Hash2Str(const Hash *hash);

	static void setSeed(unsigned int seed);
	
	uint64_t getVal() const;

	bool operator == (const Hash &other) const;
	bool operator<(const Hash& other) const;
	bool operator>(const Hash& other) const;



private:

	typedef std::map<std::string, Hash* > HashMap; 
	typedef HashMap::iterator hashMapIt;

	static HashMap hashMap;

	static unsigned int seed;
	uint64_t hashedVal;
	

	Hash(std::string &str, unsigned int seed);

	uint64_t _MurmurHash64B ( const void * key, int len, unsigned int seed );

};

