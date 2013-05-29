
#pragma once
#include <string>
#include "vector.h"
#include <iostream>


class baseProperty{
protected:
	baseProperty(){};
public:
	virtual ~baseProperty(){};
};


template <typename T>
class Prop : public baseProperty{
	private:
		T val;
	public:

	Prop(T value){
		this->val = value;
	}

	~Prop(){
	}


	T getVal() const{
		return this->val;
	}

	void setVal(T val){
		this->val = val;
	}


	/*operator T(){
		return (this->val);
	}*/
};


template<typename T>
class managedProp : public baseProperty{
private:
	T *val;
public:
	managedProp(T *value){
		this->val = value;
	}

	~managedProp(){
		std::cout<<"deleted; val = "<<this->val<<std::endl;
		delete(this->val);
	};


	T* getVal(){
		return this->val;
	}

	operator T*(){
		return (this->val);
	}


	
};




typedef Prop<int> iProp;
typedef Prop<float> fProp;
typedef Prop<std::string> sProp;
typedef Prop<vector2> v2Prop;

//can be used as a dummy to "TAG" certain obects.
typedef Prop<char> dummyProp;