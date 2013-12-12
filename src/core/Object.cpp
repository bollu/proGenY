#pragma once
#include "Object.h"
#include <sstream>
#include "../util/logObject.h"
#include "../util/mathUtil.h"

std::map<std::string, unsigned int> Object::nameMap;

Object::Object(std::string _name) : dead(false){
	//create a unique name from the generic name given
	this->_genUniqueName(_name, this->name);

	this->addProp(Hash::getHash("position"), new Prop <vector2>(vector2(0, 0)));
	this->addProp(Hash::getHash("facing"), new Prop <util::Angle>(util::Angle::Deg(0)));



};

Object::~Object(){
	for(auto it = this->propertyMap.begin(); it != this->propertyMap.end(); propertyMap.erase(it++)){
		delete ((*it).second);	
	}

}


//public-------------------------------------------
void Object::addProp(const Hash *name, baseProperty *value){
	
	auto it = propertyMap.find(name);
	if(it == propertyMap.end()){
		propertyMap[name] = value;
		
		return;
	}

	util::errorLog<<"trying to add property twice to object\n  \
		\nProperty: "<<name<<
		"\nObject Name: "<<this->getName();
}

baseProperty *Object::getBaseProp(const Hash *name){

	auto it = propertyMap.find(name);
	
	if( it != propertyMap.end() ){
		return it->second;
	}
	return NULL;
}

std::string Object::getName(){
	return this->name;
};



/*
//privates---------------------------------------------
void Object::_addData(const Hash *name, baseProperty* value){
	dataMap[name] = value;
}

baseProperty* Object::_getData(const Hash *name){

	dataIt it;
	if( (it = dataMap.find(name)) != dataMap.end() ){
		return it->second;
	}
	return NULL;
}
*/


void Object::_genUniqueName(std::string genericName, std::string &out){
	auto it = this->nameMap.find(genericName);
	std::stringstream sstm;

	//the name hasn't been stored
	if(it == this->nameMap.end()){
		
		this->nameMap[genericName] = 0;


		sstm << genericName << 0;
		out = sstm.str();
	}else{

		//it's our responsibilty to increment it
		this->nameMap[genericName]++;

		sstm << genericName << this->nameMap[genericName];
		out = sstm.str();
	}
}