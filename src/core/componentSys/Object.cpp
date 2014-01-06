
#include "Object.h"
#include <sstream>

UniqueNames Object::uniqueNames;

Object::Object(std::string name) : dead(false), parent(NULL){
	//create a unique name from the generic name given
	//if the name map is already initialized, this does nothing
	initUniqueNames(100, this->uniqueNames);
	this->name =  Hash::getHash( genUniqueName(this->uniqueNames, name.c_str()) );
	
	this->propMap = Hash::CreateHashmap(5);
	

	this->addProp(Hash::getHash("position"), new Prop <vector2>(vector2(0, 0)));
	this->addProp(Hash::getHash("facing"), new Prop <util::Angle>(util::Angle::Deg(0)));
};


bool destoyPropMap(void* key, void* value, void* context){
	delete ((baseProperty *)value);
	return false;
}

Object::~Object(){
	hashmapForEach(this->propMap, &destoyPropMap, NULL);
	hashmapFree(this->propMap);

	/*for(auto it = this->propertyMap.begin(); it != this->propertyMap.end(); propertyMap.erase(it++)){
		delete ((*it).second);	
	}*/

}


//public------------------------------------------------------------
//getters-----------------------------------------------------------

std::string Object::getName() const{
	return Hash::Hash2Str(this->name);
};


bool Object::hasProperty(const Hash *name) const{
	return this->_getBaseProp(name) == NULL ? false : true;
}

bool Object::hasProperty(const char *name) const{
	return this->hasProperty(Hash::getHash(name));
}

bool Object::requireProperty(const Hash *name) const{
	if(!this->hasProperty(name)){
		IO::errorLog<<"Object "<<this->getName()<<
		" does not have Property "<<name<<" as required"<<IO::flush;

		return false;
	}

	return true;
};

bool Object::requireProperty(const char *name) const{
	return this->requireProperty(Hash::getHash(name));
};

bool Object::isDead() const{
	return this->dead;
}

//setters-----------------------------------------------------------
void Object::addChild(Object *childObj){
	childObj->parent = this;
	this->children.push_back(childObj);
};

Object *Object::getParent(){
	return this->parent;
};


void Object::addProp(const Hash *name, baseProperty *value){
	
	//there was nothing here before
	if ((hashmapPut(this->propMap, (void*)name, value)) == NULL){
		return;
	}
	/*
	propertyIt it;
	if( ( it = propertyMap.find(name)) == propertyMap.end() ){
		propertyMap[name] = value;
		
		return;
	}*/


	

	IO::errorLog<<"trying to add property twice to object\n  \
	\nProperty: "<<name<<
	"\nObject Name: "<<this->getName()<<IO::flush;
}

void Object::addProp(const char *name, baseProperty *value){
	this->addProp(Hash::getHash(name), value);
};


baseProperty *Object::_getBaseProp(const Hash *name) const{

	
	return (baseProperty *)(hashmapGet(this->propMap, (void*)name));
	
	/*
	cPropertyIt it = propertyMap.find(name);

	if( it  != propertyMap.end() ){
		return it->second;
	}*/
	return NULL;
}


void Object::_printProperties() const{
	IO::infoLog<<"\n\n"<<"name: "<<this->name<<"\n";

	/*
	for(auto it =  propertyMap.begin(); it != propertyMap.end(); ++it){
		IO::infoLog<<it->first<<"\n";
	}*/		
}


void Object::Kill(){
	this->dead = true;
}
