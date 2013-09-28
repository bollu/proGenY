#include "Hash.h"
#include "logObject.h"


std::map<std::string, Hash* > Hash::hashMap;
unsigned int Hash::seed;


Hash::Hash(std::string &str, unsigned int seed){
	this->seed = seed;
	this->hashedVal = _MurmurHash64B(str.c_str(), str.length(), seed);
}

uint64_t Hash::getVal() const{
	return this->hashedVal;
}


const Hash* Hash::getHash(const char* str){
	std::string stdString = std::string(str);

	hashMapIt it = hashMap.find(stdString);

		//the hash is not present in the map. time to create a new Hash
	if(it == hashMap.end()){
		Hash *h = new Hash(stdString, seed);
		hashMap[str] = h;

		return h;

	}else{
			//the map has the hash, so return it 
		return (it->second);
	}
}

const Hash* Hash::getHash(std::string &str){
	hashMapIt it = hashMap.find(str);

		//the hash is not present in the map. time to create a new Hash
	if(it == hashMap.end()){
		Hash *h = new Hash(str, seed);
		hashMap[str] = h;

		return h;

	}else{
			//the map has the hash, so return it 
		return (it->second);
	}

};


std::string Hash::Hash2Str(const Hash *hash){
	for(auto it = hashMap.begin(); it != hashMap.end(); ++it){
		if(it->second == hash){
			return it->first;
		}
	}

	IO::errorLog<<"unable to convert hash to string."<<IO::flush;
	return "NO HASH AVAILAIBLE";

};


void Hash::setSeed(unsigned int seed){
	Hash::seed = seed;
}

bool Hash::operator == (const Hash &other) const{
	return (this->seed == other.seed && this->hashedVal == other.hashedVal);
}

bool Hash::operator < (const Hash &other) const{
	return (this->hashedVal < other.hashedVal);
};

bool Hash::operator > (const Hash &other) const{
	return (this->hashedVal < other.hashedVal);
};


uint64_t Hash::_MurmurHash64B ( const void * key, int len, unsigned int seed ){
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	unsigned int h1 = seed ^ len;
	unsigned int h2 = 0;

	const unsigned int * data = (const unsigned int *)key;

	while(len >= 8)
	{
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;

		unsigned int k2 = *data++;
		k2 *= m; k2 ^= k2 >> r; k2 *= m;
		h2 *= m; h2 ^= k2;
		len -= 4;
	}

	if(len >= 4)
	{
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;
	}

	switch(len)
	{
		case 3: h2 ^= ((unsigned char*)data)[2] << 16;
		case 2: h2 ^= ((unsigned char*)data)[1] << 8;
		case 1: h2 ^= ((unsigned char*)data)[0];
		h2 *= m;
	};

	h1 ^= h2 >> 18; h1 *= m;
	h2 ^= h1 >> 22; h2 *= m;
	h1 ^= h2 >> 17; h1 *= m;
	h2 ^= h1 >> 19; h2 *= m;

	uint64_t h = h1;

	h = (h << 32) | h2;

	return h;
}   
