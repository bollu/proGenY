#pragma once
#include <map>
#include <string>

typedef unsigned long long uint64_t;

/*! a flyweight class to store string Hashes.  */
/*!
Hash uses the murmur hash to hash new strings. If it encounters
a string that it has hashes before, it simply returns the Hash object.
if it's a new string, then it creates a new Hash object and stores it.

Since the class consists of Flyweight objects, const Hash* can be directly used 
as a key for any associative container (such as std::map). This is because
no new Hash objects are created. They are simply reused. So, a direct pointer comparison
is sufficient to check whether two Hashes are the same. This makes it extremely powerful
to store objects that are constantly queried by name    
*/ 
class Hash{

public:

	/*! returns a flyweight Hash object created from the given string
	
	If the string has already been hashed, the function returns
	the previous Hash object. if the string is a new string,
	then a new Hash object is created, and is then stored. 

	@param [in] str the input string that is to be hashed.

	\return
	 The const Hash* that is a flyweight hash of the given string
	*/ 
	static const Hash* getHash(const char* str);

	static const Hash* getHash(std::string &str);

	/*static const Hash* getHash(std::string& str){
		Hash::getHash(str.c_str());
	};*/
	/*! converts a Hash object to the actual string
	
	The hash is searched for internally. Once found, the corresponding string 
	is returned. The process is quite expensive as it is a linear search, which is O(n)

	
	\return 
	the string that the Hash has been created from 
	*/
	static std::string Hash2Str(const Hash *hash);

	/*! sets the seed that is used for the hashing algorithm internally

	The seed is part of the MurmurHash algorithm. the seed is used to create
	the Hash value of the string. Two strings created from different seeds_
	will have different hashes_. So,_do not_ change the seed value
	in the middle of the program. this will screw up the flyweight implementation.

	 @param [in] seed The seed that is used to hash all the strings.
	 */
	static void setSeed(unsigned int seed);
	

	/*! returns the hashed value of the string

	\return
	The MurmurHash hash value of the string that this Hash was created from
	*/
	uint64_t getVal() const;

	/*! returns if one Hash value is equal to the other. 
	
	This function is usually not used. Since Hashes are flyweights, a direct pointer
	comparison will show if two hashes are equal to each other or not.
	
	\return
	whether this Hash is equal to the other Hash.
	*/
	bool operator == (const Hash &other) const;
	
	/*! returns if one Hash value is less than the other. 
	
	\return
	whether this Hash is less than the other Hash
	*/
	bool operator<(const Hash& other) const;

	/*! returns if one Hash value is greater than the other. 
	
	\return
	whether this Hash is greater than the other Hash
	*/
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

