#pragma once
#include "../Hash.h"
#include "../vector.h"


/*
class eventData{
private:
	union _eventDataUnion{
	float f;
	int i;
	const Hash *hash;
	};

	_eventDataUnion data;
public:
	eventData(){
		//use the least
		this->data.i = 0;
	}

	eventData(int i){
		this->data.i = i;
	}

	eventData(float f){
		this->data.f = f;
	}

	eventData(std::string str){
		this->data.hash = Hash::getHash(str);
	}

	eventData(const Hash *hash){
		this->data.hash = hash;
	}
	
	eventData(vector2 v2){
		this->data.v2 = v2;
	}

	int getInt(){
		return this->data.i;
	}

	float getFloat(){
		return this->data.f;
	}

	std::string getStr(){
		return Hash::Hash2Str(this->data.hash);
	}

	const Hash *getHash(){
		return this->data.hash;
	}

	vector2 getVector2(){
		return this->data.v2;
	}

	
};*/